
"""
predict_real_foot_with_silhouette_model.py

학습된 silhouette HVA/IMA 모델을 이용해 실제 외형 발 사진 silhouette에 대해
외형 기반 HVA/IMA 의심각을 예측합니다.

입력:
  data/processed/real_foot_silhouette/real_foot_silhouette.csv
  models/silhouette_model/best_silhouette_model.pth

출력:
  outputs/real_foot_test/predictions/real_foot_predictions.csv
  outputs/real_foot_test/overlays

주의:
- 이 예측값은 X-ray 기반 병원식 HVA 확정값이 아닙니다.
- 실제 발 외형 silhouette로 추정한 screening 지표입니다.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import timm

from src.utils.config import (
    REAL_FOOT_SILHOUETTE_CSV,
    BEST_SILHOUETTE_MODEL,
    REAL_FOOT_PREDICTION_DIR,
    REAL_FOOT_OUTPUT_OVERLAY_DIR,
    IMG_SIZE,
    DEVICE,
)


ANGLE_SCALE = 90.0
BACKBONE_FALLBACK = "efficientnet_b0"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class SilhouetteRegressionModel(nn.Module):
    def __init__(self, backbone_name: str = BACKBONE_FALLBACK):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
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


def get_device() -> torch.device:
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def normalize_image(img_rgb: np.ndarray) -> torch.Tensor:
    img = img_rgb.astype(np.float32) / 255.0
    mean = np.array(MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(STD, dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).float()


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"silhouette model checkpoint가 없습니다: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    backbone = checkpoint.get("backbone", BACKBONE_FALLBACK)
    model = SilhouetteRegressionModel(backbone).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, checkpoint


def grade_hva(hva: float) -> str:
    if hva < 15:
        return "정상 범위 의심"
    if hva < 20:
        return "경도 무지외반 의심"
    if hva < 40:
        return "중등도 무지외반 의심"
    return "고도 무지외반 의심"


def predict_one(model: nn.Module, image_path: str, device: torch.device) -> Tuple[float, float]:
    img = read_image_rgb(image_path)

    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    x = normalize_image(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)[0].detach().cpu().numpy()

    pred_hva = float(pred[0])
    pred_ima = float(pred[1])

    return pred_hva, pred_ima


def make_prediction_overlay(row, pred_hva: float, pred_ima: float, grade: str, out_path: Path) -> None:
    overlay_src = str(row.get("overlay_path", ""))

    if overlay_src and Path(overlay_src).exists():
        img = cv2.imread(overlay_src, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(row["silhouette_image_path"]), cv2.IMREAD_COLOR)

    if img is None:
        return

    text1 = f"HVA={pred_hva:.2f} deg | IMA={pred_ima:.2f} deg"
    text2 = f"{grade}"

    cv2.rectangle(img, (0, img.shape[0] - 72), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, text1, (12, img.shape[0] - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text2, (12, img.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(REAL_FOOT_SILHOUETTE_CSV))
    parser.add_argument("--model", type=str, default=str(BEST_SILHOUETTE_MODEL))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    model_path = Path(args.model)

    if not csv_path.exists():
        raise FileNotFoundError(f"real_foot_silhouette.csv가 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"sample_id", "silhouette_image_path"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"real foot csv에 필요한 컬럼이 없습니다: {sorted(missing)}")

    device = get_device()
    model, checkpoint = load_model(model_path, device)

    REAL_FOOT_PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    REAL_FOOT_OUTPUT_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    print("실제 발 이미지 HVA/IMA 예측 시작")
    print("=" * 80)
    print(f"Device  : {device}")
    print(f"CSV     : {csv_path}")
    print(f"Model   : {model_path}")
    print(f"Backbone: {checkpoint.get('backbone', BACKBONE_FALLBACK)}")
    print(f"Images  : {len(df)}")
    print("=" * 80)

    records = []

    for idx, row in df.iterrows():
        sample_id = str(row["sample_id"])
        image_path = str(row["silhouette_image_path"])

        pred_hva, pred_ima = predict_one(model, image_path, device)
        grade = grade_hva(pred_hva)

        pred_overlay_path = REAL_FOOT_OUTPUT_OVERLAY_DIR / f"{sample_id}_prediction_overlay.png"
        make_prediction_overlay(row, pred_hva, pred_ima, grade, pred_overlay_path)

        rec = dict(row)
        rec.update(
            {
                "pred_HVA": pred_hva,
                "pred_IMA": pred_ima,
                "hva_grade": grade,
                "prediction_overlay_path": str(pred_overlay_path),
                "note": "외형 기반 screening 지표이며, X-ray 기반 확정 HVA가 아닙니다.",
            }
        )
        records.append(rec)

        print(f"[{idx + 1}/{len(df)}] {sample_id}: HVA={pred_hva:.2f}, IMA={pred_ima:.2f}, {grade}")

    out_csv = REAL_FOOT_PREDICTION_DIR / "real_foot_predictions.csv"
    out_df = pd.DataFrame(records)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n완료")
    print("=" * 80)
    print(f"예측 CSV 저장: {out_csv}")
    print(f"오버레이 저장: {REAL_FOOT_OUTPUT_OVERLAY_DIR}")

    if len(out_df) > 0:
        print("\n예측 요약")
        print("-" * 80)
        print(f"HVA mean: {out_df['pred_HVA'].mean():.2f}")
        print(f"HVA min : {out_df['pred_HVA'].min():.2f}")
        print(f"HVA max : {out_df['pred_HVA'].max():.2f}")
        print(f"IMA mean: {out_df['pred_IMA'].mean():.2f}")
        print("\n등급 분포")
        print(out_df["hva_grade"].value_counts().to_string())


if __name__ == "__main__":
    main()
