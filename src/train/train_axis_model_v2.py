
"""
train_axis_model_v2.py

Axis model 개선판.

변경점:
1. Best model 저장 기준을 val_loss가 아니라 실제 목표 지표인 angle score로 변경
2. axis 좌표 loss를 endpoint swap-invariant loss로 변경
3. geometry HVA/IMA loss 가중치 강화
4. HVA 구간 기준 WeightedRandomSampler 추가
5. best checkpoint를 score / geometry HVA / direct HVA 기준으로 각각 저장

실행:
  python src/train/train_axis_model_v2.py
"""

import csv
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import timm

from src.utils.config import (
    AXIS_TRAIN_CSV,
    AXIS_VAL_CSV,
    AXIS_TEST_CSV,
    AXIS_MODEL_DIR,
    AXIS_LOG_DIR,
    IMG_SIZE,
    NUM_WORKERS,
    SEED,
    DEVICE,
)


EPOCHS = 140
LR = 1.2e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 35
MIN_DELTA = 1e-4

BACKBONE = "efficientnet_b2"
BATCH_SIZE = 6

W_AXIS = 25.0
W_GEOM_HVA = 0.35
W_GEOM_IMA = 0.35
W_DIRECT_HVA = 0.08
W_DIRECT_IMA = 0.08
W_CONSISTENCY = 0.10

ANGLE_SCALE = 90.0

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def normalize_image(img_rgb: np.ndarray) -> torch.Tensor:
    img = img_rgb.astype(np.float32) / 255.0
    mean = np.array(MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(STD, dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).float()


def line_angle_torch(line_a: torch.Tensor, line_b: torch.Tensor) -> torch.Tensor:
    va = line_a[:, 2:4] - line_a[:, 0:2]
    vb = line_b[:, 2:4] - line_b[:, 0:2]
    dot = (va * vb).sum(dim=1)
    denom = torch.linalg.norm(va, dim=1) * torch.linalg.norm(vb, dim=1)
    cos = dot / (denom + 1e-8)
    cos = torch.clamp(cos, -1.0 + 1e-6, 1.0 - 1e-6)
    angle = torch.rad2deg(torch.acos(cos))
    angle = torch.where(angle > 90.0, 180.0 - angle, angle)
    return angle


def tensor_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred.detach() - target.detach())).item()


def split_axis(axis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return axis[:, 0:4], axis[:, 4:8], axis[:, 8:12]


def swap_line_endpoints(line: torch.Tensor) -> torch.Tensor:
    return torch.cat([line[:, 2:4], line[:, 0:2]], dim=1)


def swap_invariant_line_loss(pred_line: torch.Tensor, gt_line: torch.Tensor) -> torch.Tensor:
    loss_normal = F.smooth_l1_loss(pred_line, gt_line, reduction="none").mean(dim=1)
    loss_swapped = F.smooth_l1_loss(swap_line_endpoints(pred_line), gt_line, reduction="none").mean(dim=1)
    return torch.min(loss_normal, loss_swapped).mean()


def axis_loss_swap_invariant(pred_axis: torch.Tensor, gt_axis: torch.Tensor) -> torch.Tensor:
    pred_great, pred_first, pred_second = split_axis(pred_axis)
    gt_great, gt_first, gt_second = split_axis(gt_axis)
    return (
        swap_invariant_line_loss(pred_great, gt_great)
        + swap_invariant_line_loss(pred_first, gt_first)
        + swap_invariant_line_loss(pred_second, gt_second)
    ) / 3.0


def make_weighted_sampler(csv_path: Path) -> Optional[WeightedRandomSampler]:
    df = pd.read_csv(csv_path)
    if "HVA" not in df.columns:
        return None

    bins = [0, 15, 20, 40, 60, 200]
    labels = pd.cut(df["HVA"].astype(float), bins=bins, include_lowest=True)
    counts = labels.value_counts().to_dict()

    weights = []
    for item in labels:
        weights.append(1.0 / max(1, counts.get(item, 1)))

    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


class AxisDataset(Dataset):
    def __init__(self, csv_path: Path, augment: bool = False):
        if not csv_path.exists():
            raise FileNotFoundError(f"split csv가 없습니다: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.augment = augment

        self.axis_cols = [
            "great_toe_x1", "great_toe_y1", "great_toe_x2", "great_toe_y2",
            "first_metatarsal_x1", "first_metatarsal_y1", "first_metatarsal_x2", "first_metatarsal_y2",
            "second_metatarsal_x1", "second_metatarsal_y1", "second_metatarsal_x2", "second_metatarsal_y2",
        ]

        required = {"image_path", "HVA", "IMA", *self.axis_cols}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"{csv_path}에 필요한 컬럼이 없습니다: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def _augment_brightness_contrast(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() > 0.65:
            return img
        alpha = np.random.uniform(0.90, 1.10)
        beta = np.random.uniform(-8, 8)
        out = img.astype(np.float32) * alpha + beta
        return np.clip(out, 0, 255).astype(np.uint8)

    def _augment_noise(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() > 0.85:
            return img
        noise = np.random.normal(0, 3.0, img.shape).astype(np.float32)
        out = img.astype(np.float32) + noise
        return np.clip(out, 0, 255).astype(np.uint8)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img = read_image_rgb(str(row["image_path"]))

        if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        if self.augment:
            img = self._augment_brightness_contrast(img)
            img = self._augment_noise(img)

        img_t = normalize_image(img)

        axis = np.array([float(row[c]) for c in self.axis_cols], dtype=np.float32)

        return {
            "image": img_t,
            "axis": torch.from_numpy(axis),
            "hva": torch.tensor(float(row["HVA"]), dtype=torch.float32),
            "ima": torch.tensor(float(row["IMA"]), dtype=torch.float32),
        }


class AxisRegressionModel(nn.Module):
    def __init__(self, backbone_name: str = BACKBONE):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 768),
            nn.BatchNorm1d(768),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.15),
        )

        self.axis_head = nn.Sequential(nn.Linear(384, 12), nn.Sigmoid())
        self.angle_head = nn.Sequential(nn.Linear(384, 2), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        hidden = self.head(feat)
        pred_axis = self.axis_head(hidden)
        pred_angle = self.angle_head(hidden) * ANGLE_SCALE
        return pred_axis, pred_angle


def compute_losses(
    pred_axis: torch.Tensor,
    pred_angle: torch.Tensor,
    gt_axis: torch.Tensor,
    gt_hva: torch.Tensor,
    gt_ima: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pred_great, pred_first, pred_second = split_axis(pred_axis)

    pred_geom_hva = line_angle_torch(pred_great, pred_first)
    pred_geom_ima = line_angle_torch(pred_first, pred_second)

    pred_direct_hva = pred_angle[:, 0]
    pred_direct_ima = pred_angle[:, 1]

    gt_hva_norm = gt_hva / ANGLE_SCALE
    gt_ima_norm = gt_ima / ANGLE_SCALE

    pred_geom_hva_norm = pred_geom_hva / ANGLE_SCALE
    pred_geom_ima_norm = pred_geom_ima / ANGLE_SCALE
    pred_direct_hva_norm = pred_direct_hva / ANGLE_SCALE
    pred_direct_ima_norm = pred_direct_ima / ANGLE_SCALE

    axis_loss = axis_loss_swap_invariant(pred_axis, gt_axis)

    geom_hva_loss = F.smooth_l1_loss(pred_geom_hva_norm, gt_hva_norm)
    geom_ima_loss = F.smooth_l1_loss(pred_geom_ima_norm, gt_ima_norm)

    direct_hva_loss = F.smooth_l1_loss(pred_direct_hva_norm, gt_hva_norm)
    direct_ima_loss = F.smooth_l1_loss(pred_direct_ima_norm, gt_ima_norm)

    consistency_loss = (
        F.smooth_l1_loss(pred_geom_hva_norm, pred_direct_hva_norm)
        + F.smooth_l1_loss(pred_geom_ima_norm, pred_direct_ima_norm)
    )

    total_loss = (
        W_AXIS * axis_loss
        + W_GEOM_HVA * geom_hva_loss
        + W_GEOM_IMA * geom_ima_loss
        + W_DIRECT_HVA * direct_hva_loss
        + W_DIRECT_IMA * direct_ima_loss
        + W_CONSISTENCY * consistency_loss
    )

    return {
        "total": total_loss,
        "axis": axis_loss,
        "geom_hva": geom_hva_loss,
        "geom_ima": geom_ima_loss,
        "direct_hva": direct_hva_loss,
        "direct_ima": direct_ima_loss,
        "consistency": consistency_loss,
        "pred_geom_hva": pred_geom_hva,
        "pred_geom_ima": pred_geom_ima,
        "pred_direct_hva": pred_direct_hva,
        "pred_direct_ima": pred_direct_ima,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)

    sums = {
        "loss": 0.0,
        "axis": 0.0,
        "geom_hva_loss": 0.0,
        "geom_ima_loss": 0.0,
        "direct_hva_loss": 0.0,
        "direct_ima_loss": 0.0,
        "consistency": 0.0,
        "geom_hva_mae": 0.0,
        "geom_ima_mae": 0.0,
        "direct_hva_mae": 0.0,
        "direct_ima_mae": 0.0,
    }

    n_batches = 0

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        gt_axis = batch["axis"].to(device, non_blocking=True)
        gt_hva = batch["hva"].to(device, non_blocking=True)
        gt_ima = batch["ima"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            pred_axis, pred_angle = model(img)

            loss_dict = compute_losses(
                pred_axis=pred_axis,
                pred_angle=pred_angle,
                gt_axis=gt_axis,
                gt_hva=gt_hva,
                gt_ima=gt_ima,
            )

            loss = loss_dict["total"]

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        sums["loss"] += loss.item()
        sums["axis"] += loss_dict["axis"].item()
        sums["geom_hva_loss"] += loss_dict["geom_hva"].item()
        sums["geom_ima_loss"] += loss_dict["geom_ima"].item()
        sums["direct_hva_loss"] += loss_dict["direct_hva"].item()
        sums["direct_ima_loss"] += loss_dict["direct_ima"].item()
        sums["consistency"] += loss_dict["consistency"].item()

        sums["geom_hva_mae"] += tensor_mae(loss_dict["pred_geom_hva"], gt_hva)
        sums["geom_ima_mae"] += tensor_mae(loss_dict["pred_geom_ima"], gt_ima)
        sums["direct_hva_mae"] += tensor_mae(loss_dict["pred_direct_hva"], gt_hva)
        sums["direct_ima_mae"] += tensor_mae(loss_dict["pred_direct_ima"], gt_ima)

        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in sums.items()}


def angle_score(metrics: Dict[str, float]) -> float:
    return (
        metrics["geom_hva_mae"]
        + metrics["geom_ima_mae"]
        + 0.5 * metrics["direct_hva_mae"]
        + 0.5 * metrics["direct_ima_mae"]
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    score_value: float,
    metrics: Dict[str, float],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "score": score_value,
            "backbone": BACKBONE,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "loss_weights": {
                "W_AXIS": W_AXIS,
                "W_GEOM_HVA": W_GEOM_HVA,
                "W_GEOM_IMA": W_GEOM_IMA,
                "W_DIRECT_HVA": W_DIRECT_HVA,
                "W_DIRECT_IMA": W_DIRECT_IMA,
                "W_CONSISTENCY": W_CONSISTENCY,
            },
            "val_metrics": metrics,
        },
        path,
    )


def main() -> None:
    seed_everything(SEED)

    device = get_device()

    AXIS_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    AXIS_LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = AxisDataset(AXIS_TRAIN_CSV, augment=True)
    val_ds = AxisDataset(AXIS_VAL_CSV, augment=False)
    test_ds = AxisDataset(AXIS_TEST_CSV, augment=False)

    sampler = make_weighted_sampler(AXIS_TRAIN_CSV)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False if sampler is not None else True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = AxisRegressionModel(BACKBONE).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    best_score_path = AXIS_MODEL_DIR / "best_axis_model_v2_score.pth"
    best_geom_hva_path = AXIS_MODEL_DIR / "best_axis_model_v2_geom_hva.pth"
    best_direct_hva_path = AXIS_MODEL_DIR / "best_axis_model_v2_direct_hva.pth"
    last_path = AXIS_MODEL_DIR / "last_axis_model_v2.pth"

    log_path = AXIS_LOG_DIR / "train_axis_v2_log.csv"

    log_cols = [
        "epoch",
        "lr",
        "train_loss",
        "val_loss",
        "train_axis",
        "val_axis",
        "val_score",
        "val_geom_hva_mae",
        "val_geom_ima_mae",
        "val_direct_hva_mae",
        "val_direct_ima_mae",
    ]

    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=log_cols)
        writer.writeheader()

    print("Axis model v2 학습 시작")
    print("=" * 88)
    print(f"Device   : {device}")
    print(f"Backbone : {BACKBONE}")
    print(f"Train    : {len(train_ds)}")
    print(f"Val      : {len(val_ds)}")
    print(f"Test     : {len(test_ds)}")
    print(f"Batch    : {BATCH_SIZE}")
    print(f"Epochs   : {EPOCHS}")
    print(f"Sampler  : {'WeightedRandomSampler' if sampler is not None else 'shuffle'}")
    print(f"Save best score: {best_score_path}")
    print("=" * 88)
    print(
        f"Loss weights | axis={W_AXIS}, geom_hva={W_GEOM_HVA}, geom_ima={W_GEOM_IMA}, "
        f"direct_hva={W_DIRECT_HVA}, direct_ima={W_DIRECT_IMA}, consistency={W_CONSISTENCY}"
    )
    print("=" * 88)

    best_score = float("inf")
    best_geom_hva = float("inf")
    best_direct_hva = float("inf")
    best_epoch = 0
    no_improve = 0

    header = (
        f"{'Ep':>4} {'lr':>9} "
        f"{'tr_loss':>8} {'v_loss':>8} {'score':>8} "
        f"{'v_gHVA':>8} {'v_dHVA':>8} "
        f"{'v_gIMA':>8} {'v_dIMA':>8}"
    )

    print(header)
    print("-" * len(header))

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        train_m = run_epoch(model, train_loader, device, optimizer=optimizer)

        with torch.no_grad():
            val_m = run_epoch(model, val_loader, device, optimizer=None)

        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        score = angle_score(val_m)

        improved_score = score < best_score - MIN_DELTA
        improved_geom_hva = val_m["geom_hva_mae"] < best_geom_hva - MIN_DELTA
        improved_direct_hva = val_m["direct_hva_mae"] < best_direct_hva - MIN_DELTA

        if improved_score:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(best_score_path, model, epoch, score, val_m)
        else:
            no_improve += 1

        if improved_geom_hva:
            best_geom_hva = val_m["geom_hva_mae"]
            save_checkpoint(best_geom_hva_path, model, epoch, best_geom_hva, val_m)

        if improved_direct_hva:
            best_direct_hva = val_m["direct_hva_mae"]
            save_checkpoint(best_direct_hva_path, model, epoch, best_direct_hva, val_m)

        save_checkpoint(last_path, model, epoch, score, val_m)

        with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=log_cols)
            writer.writerow(
                {
                    "epoch": epoch,
                    "lr": lr,
                    "train_loss": train_m["loss"],
                    "val_loss": val_m["loss"],
                    "train_axis": train_m["axis"],
                    "val_axis": val_m["axis"],
                    "val_score": score,
                    "val_geom_hva_mae": val_m["geom_hva_mae"],
                    "val_geom_ima_mae": val_m["geom_ima_mae"],
                    "val_direct_hva_mae": val_m["direct_hva_mae"],
                    "val_direct_ima_mae": val_m["direct_ima_mae"],
                }
            )

        mark = "*" if improved_score else ""
        elapsed = time.time() - start

        print(
            f"{epoch:4d} {lr:9.2e} "
            f"{train_m['loss']:8.4f} {val_m['loss']:8.4f} {score:8.3f} "
            f"{val_m['geom_hva_mae']:8.3f} {val_m['direct_hva_mae']:8.3f} "
            f"{val_m['geom_ima_mae']:8.3f} {val_m['direct_ima_mae']:8.3f} "
            f"{mark} {elapsed:.1f}s"
        )

        if no_improve >= PATIENCE:
            print(f"\n조기 종료: {PATIENCE} epoch 동안 score 개선 없음")
            break

    print("\n학습 완료")
    print("=" * 88)
    print(f"Best score epoch: {best_epoch}")
    print(f"Best val score  : {best_score:.6f}")
    print(f"Best score model: {best_score_path}")
    print(f"Best geom HVA   : {best_geom_hva:.6f} -> {best_geom_hva_path}")
    print(f"Best direct HVA : {best_direct_hva:.6f} -> {best_direct_hva_path}")
    print(f"Last model      : {last_path}")
    print(f"Log             : {log_path}")

    checkpoint = torch.load(best_score_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        test_m = run_epoch(model, test_loader, device, optimizer=None)

    print("\nTest set 결과 - best score model")
    print("=" * 88)
    print(f"geometry HVA MAE : {test_m['geom_hva_mae']:.4f} deg")
    print(f"direct   HVA MAE : {test_m['direct_hva_mae']:.4f} deg")
    print(f"geometry IMA MAE : {test_m['geom_ima_mae']:.4f} deg")
    print(f"direct   IMA MAE : {test_m['direct_ima_mae']:.4f} deg")
    print(f"score            : {angle_score(test_m):.4f}")


if __name__ == "__main__":
    main()
