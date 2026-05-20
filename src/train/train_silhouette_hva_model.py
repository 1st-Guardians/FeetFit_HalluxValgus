
"""
train_silhouette_hva_model.py

X-ray silhouette 기반 HVA/IMA direct regression 모델 학습 코드입니다.

목적:
- 내부 뼈 구조가 제거된 silhouette 이미지만으로 HVA/IMA를 어느 정도 예측할 수 있는지 확인
- 일반 외형 발 사진으로 넘어가기 위한 bridge 실험

입력:
  data/splits/silhouette_train.csv
  data/splits/silhouette_val.csv
  data/splits/silhouette_test.csv

출력:
  models/silhouette_model/best_silhouette_model.pth
  models/silhouette_model/last_silhouette_model.pth
  outputs/silhouette_model/logs/train_silhouette_log.csv

실행:
  python src/train/train_silhouette_hva_model.py
"""

import csv
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import timm

from src.utils.config import (
    SILHOUETTE_TRAIN_CSV,
    SILHOUETTE_VAL_CSV,
    SILHOUETTE_TEST_CSV,
    SILHOUETTE_MODEL_DIR,
    BEST_SILHOUETTE_MODEL,
    LAST_SILHOUETTE_MODEL,
    SILHOUETTE_LOG_DIR,
    IMG_SIZE,
    NUM_WORKERS,
    SEED,
    DEVICE,
)


# ============================================================
# 학습 설정
# ============================================================

EPOCHS = 140
LR = 1.5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 30
MIN_DELTA = 1e-4

BACKBONE = "efficientnet_b0"
BATCH_SIZE = 8

ANGLE_SCALE = 90.0

# silhouette는 흑백에 가까우나 pretrained backbone 사용을 위해 3채널 유지
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# HVA가 메인 목표라 HVA에 조금 더 가중치
W_HVA = 1.0
W_IMA = 0.7


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


class SilhouetteDataset(Dataset):
    def __init__(self, csv_path: Path, augment: bool = False):
        if not csv_path.exists():
            raise FileNotFoundError(f"split csv가 없습니다: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.augment = augment

        required = {"silhouette_image_path", "HVA", "IMA"}
        missing = required - set(self.df.columns)

        if missing:
            raise ValueError(f"{csv_path}에 필요한 컬럼이 없습니다: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def _augment_mask(self, img: np.ndarray) -> np.ndarray:
        """
        silhouette는 실제 발 사진 mask로 넘어갈 때 contour가 조금 흔들릴 수 있으므로
        약한 erosion/dilation과 blur를 섞어 robustness를 줍니다.
        """
        if np.random.rand() < 0.25:
            k = int(np.random.choice([3, 5]))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            if np.random.rand() < 0.5:
                img = cv2.dilate(img, kernel, iterations=1)
            else:
                img = cv2.erode(img, kernel, iterations=1)

        if np.random.rand() < 0.15:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        img = read_image_rgb(str(row["silhouette_image_path"]))

        if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        if self.augment:
            img = self._augment_mask(img)

        img_t = normalize_image(img)

        hva = torch.tensor(float(row["HVA"]), dtype=torch.float32)
        ima = torch.tensor(float(row["IMA"]), dtype=torch.float32)

        return {
            "image": img_t,
            "hva": hva,
            "ima": ima,
        }


class SilhouetteRegressionModel(nn.Module):
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
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.30),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.20),

            nn.Linear(256, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        angle = self.head(feat) * ANGLE_SCALE
        return angle


def compute_loss(pred_angle: torch.Tensor, gt_hva: torch.Tensor, gt_ima: torch.Tensor) -> Dict[str, torch.Tensor]:
    pred_hva = pred_angle[:, 0]
    pred_ima = pred_angle[:, 1]

    pred_hva_norm = pred_hva / ANGLE_SCALE
    pred_ima_norm = pred_ima / ANGLE_SCALE
    gt_hva_norm = gt_hva / ANGLE_SCALE
    gt_ima_norm = gt_ima / ANGLE_SCALE

    hva_loss = F.smooth_l1_loss(pred_hva_norm, gt_hva_norm)
    ima_loss = F.smooth_l1_loss(pred_ima_norm, gt_ima_norm)

    total = W_HVA * hva_loss + W_IMA * ima_loss

    return {
        "total": total,
        "hva_loss": hva_loss,
        "ima_loss": ima_loss,
        "pred_hva": pred_hva,
        "pred_ima": pred_ima,
    }


def tensor_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred.detach() - target.detach())).item()


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
        "hva_loss": 0.0,
        "ima_loss": 0.0,
        "hva_mae": 0.0,
        "ima_mae": 0.0,
    }

    n_batches = 0

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        gt_hva = batch["hva"].to(device, non_blocking=True)
        gt_ima = batch["ima"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            pred_angle = model(img)

            loss_dict = compute_loss(pred_angle, gt_hva, gt_ima)
            loss = loss_dict["total"]

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        sums["loss"] += loss.item()
        sums["hva_loss"] += loss_dict["hva_loss"].item()
        sums["ima_loss"] += loss_dict["ima_loss"].item()
        sums["hva_mae"] += tensor_mae(loss_dict["pred_hva"], gt_hva)
        sums["ima_mae"] += tensor_mae(loss_dict["pred_ima"], gt_ima)

        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in sums.items()}


def score(metrics: Dict[str, float]) -> float:
    return metrics["hva_mae"] + 0.7 * metrics["ima_mae"]


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
            "val_metrics": metrics,
            "loss_weights": {
                "W_HVA": W_HVA,
                "W_IMA": W_IMA,
            },
        },
        path,
    )


def main() -> None:
    seed_everything(SEED)

    device = get_device()

    SILHOUETTE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SILHOUETTE_LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = SilhouetteDataset(SILHOUETTE_TRAIN_CSV, augment=True)
    val_ds = SilhouetteDataset(SILHOUETTE_VAL_CSV, augment=False)
    test_ds = SilhouetteDataset(SILHOUETTE_TEST_CSV, augment=False)

    sampler = make_weighted_sampler(SILHOUETTE_TRAIN_CSV)

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

    model = SilhouetteRegressionModel(BACKBONE).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    log_path = SILHOUETTE_LOG_DIR / "train_silhouette_log.csv"

    log_cols = [
        "epoch",
        "lr",
        "train_loss",
        "val_loss",
        "val_score",
        "val_hva_mae",
        "val_ima_mae",
    ]

    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=log_cols)
        writer.writeheader()

    print("Silhouette HVA/IMA model 학습 시작")
    print("=" * 80)
    print(f"Device   : {device}")
    print(f"Backbone : {BACKBONE}")
    print(f"Train    : {len(train_ds)}")
    print(f"Val      : {len(val_ds)}")
    print(f"Test     : {len(test_ds)}")
    print(f"Batch    : {BATCH_SIZE}")
    print(f"Epochs   : {EPOCHS}")
    print(f"Sampler  : {'WeightedRandomSampler' if sampler is not None else 'shuffle'}")
    print(f"Save best: {BEST_SILHOUETTE_MODEL}")
    print("=" * 80)

    best_score = float("inf")
    best_epoch = 0
    no_improve = 0

    header = f"{'Ep':>4} {'lr':>9} {'tr_loss':>8} {'v_loss':>8} {'score':>8} {'v_HVA':>8} {'v_IMA':>8}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        train_m = run_epoch(model, train_loader, device, optimizer=optimizer)

        with torch.no_grad():
            val_m = run_epoch(model, val_loader, device, optimizer=None)

        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        val_score = score(val_m)

        improved = val_score < best_score - MIN_DELTA

        if improved:
            best_score = val_score
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(BEST_SILHOUETTE_MODEL, model, epoch, val_score, val_m)
        else:
            no_improve += 1

        save_checkpoint(LAST_SILHOUETTE_MODEL, model, epoch, val_score, val_m)

        with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=log_cols)
            writer.writerow(
                {
                    "epoch": epoch,
                    "lr": lr,
                    "train_loss": train_m["loss"],
                    "val_loss": val_m["loss"],
                    "val_score": val_score,
                    "val_hva_mae": val_m["hva_mae"],
                    "val_ima_mae": val_m["ima_mae"],
                }
            )

        mark = "*" if improved else ""
        elapsed = time.time() - start

        print(
            f"{epoch:4d} {lr:9.2e} "
            f"{train_m['loss']:8.4f} {val_m['loss']:8.4f} {val_score:8.3f} "
            f"{val_m['hva_mae']:8.3f} {val_m['ima_mae']:8.3f} "
            f"{mark} {elapsed:.1f}s"
        )

        if no_improve >= PATIENCE:
            print(f"\n조기 종료: {PATIENCE} epoch 동안 score 개선 없음")
            break

    print("\n학습 완료")
    print("=" * 80)
    print(f"Best epoch: {best_epoch}")
    print(f"Best val score: {best_score:.6f}")
    print(f"Best model: {BEST_SILHOUETTE_MODEL}")
    print(f"Last model: {LAST_SILHOUETTE_MODEL}")
    print(f"Log: {log_path}")

    checkpoint = torch.load(BEST_SILHOUETTE_MODEL, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        test_m = run_epoch(model, test_loader, device, optimizer=None)

    print("\nTest set 결과")
    print("=" * 80)
    print(f"HVA MAE : {test_m['hva_mae']:.4f} deg")
    print(f"IMA MAE : {test_m['ima_mae']:.4f} deg")
    print(f"Score   : {score(test_m):.4f}")


if __name__ == "__main__":
    main()
