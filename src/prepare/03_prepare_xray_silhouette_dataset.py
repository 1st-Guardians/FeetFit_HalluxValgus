
"""
03_prepare_xray_silhouette_dataset.py

X-ray 원본 기반 axis dataset에서 생성된 crop 이미지를 이용해
X-ray silhouette/mask dataset을 생성합니다.

목적:
- 일반 발 외형 사진과의 도메인 차이를 줄이기 위해
  X-ray 내부 뼈 구조를 제거하고 발 외곽선 형태만 남긴 silhouette 이미지를 생성합니다.

입력:
  data/processed/axis_dataset/processed_axis_dataset.csv
  data/processed/axis_dataset/images/*.png

출력:
  data/processed/xray_silhouette_dataset/images
  data/processed/xray_silhouette_dataset/masks
  data/processed/xray_silhouette_dataset/overlays
  data/processed/xray_silhouette_dataset/processed_silhouette_dataset.csv

실행:
  python src/prepare/03_prepare_xray_silhouette_dataset.py
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.utils.config import (
    PROCESSED_AXIS_CSV,
    XRAY_SILHOUETTE_IMAGE_DIR,
    XRAY_SILHOUETTE_MASK_DIR,
    XRAY_SILHOUETTE_OVERLAY_DIR,
    PROCESSED_SILHOUETTE_CSV,
    IMG_SIZE,
    make_dirs,
)


def read_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")

    return img


def normalize_to_uint8(gray: np.ndarray) -> np.ndarray:
    gray = gray.astype(np.float32)
    min_v = float(np.percentile(gray, 1))
    max_v = float(np.percentile(gray, 99))

    if max_v - min_v < 1e-6:
        return np.zeros_like(gray, dtype=np.uint8)

    norm = (gray - min_v) / (max_v - min_v)
    norm = np.clip(norm, 0.0, 1.0)
    return (norm * 255).astype(np.uint8)


def get_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[0]


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_small, iterations=1)

    contour = get_largest_contour(cleaned)

    if contour is None:
        return cleaned

    largest = np.zeros_like(cleaned)
    cv2.drawContours(largest, [contour], -1, 255, thickness=-1)
    largest = cv2.morphologyEx(largest, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    return largest


def create_mask_by_otsu(gray: np.ndarray) -> np.ndarray:
    """
    Otsu threshold 기반으로 발 영역 mask를 추출합니다.

    X-ray는 이미지마다 밝기 방향이 달라질 수 있어
    threshold 결과와 inverse 결과 중 더 그럴듯한 mask를 선택합니다.
    """
    norm = normalize_to_uint8(gray)
    blur = cv2.GaussianBlur(norm, (5, 5), 0)

    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(th)

    th_clean = clean_mask(th)
    inv_clean = clean_mask(inv)

    h, w = gray.shape[:2]
    total_area = h * w

    def score_mask(mask: np.ndarray) -> float:
        area = float(np.count_nonzero(mask)) / float(total_area)

        contour = get_largest_contour(mask)
        if contour is None:
            return -1e9

        contour_area = cv2.contourArea(contour) / float(total_area)

        area_score = -abs(area - 0.45)
        contour_score = contour_area

        edge_pixels = np.concatenate([
            mask[0, :],
            mask[-1, :],
            mask[:, 0],
            mask[:, -1],
        ])
        edge_ratio = np.count_nonzero(edge_pixels) / max(1, edge_pixels.size)
        edge_penalty = -0.5 * edge_ratio

        return area_score + contour_score + edge_penalty

    th_score = score_mask(th_clean)
    inv_score = score_mask(inv_clean)

    if inv_score > th_score:
        return inv_clean

    return th_clean


def create_silhouette_image(mask: np.ndarray) -> np.ndarray:
    silhouette = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    silhouette[mask > 0] = (255, 255, 255)
    return silhouette


def create_overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = img.copy()

    contour = get_largest_contour(mask)
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, (0, 255, 255), 2)

    color_mask = np.zeros_like(overlay)
    color_mask[mask > 0] = (0, 255, 0)

    overlay = cv2.addWeighted(overlay, 0.75, color_mask, 0.25, 0)
    return overlay


def mask_quality(mask: np.ndarray) -> Tuple[float, float]:
    h, w = mask.shape[:2]
    total = h * w

    area_ratio = float(np.count_nonzero(mask)) / float(total)

    edge_pixels = np.concatenate([
        mask[0, :],
        mask[-1, :],
        mask[:, 0],
        mask[:, -1],
    ])

    edge_ratio = float(np.count_nonzero(edge_pixels)) / float(edge_pixels.size)

    return area_ratio, edge_ratio


def process_one(row, index: int):
    image_path = Path(str(row["image_path"]))

    if not image_path.exists():
        return None, f"image_not_found:{image_path}"

    img = read_image_bgr(image_path)

    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = create_mask_by_otsu(gray)
    silhouette = create_silhouette_image(mask)
    overlay = create_overlay(img, mask)

    area_ratio, edge_ratio = mask_quality(mask)

    sample_id = str(row["sample_id"])

    out_img_path = XRAY_SILHOUETTE_IMAGE_DIR / f"{sample_id}_silhouette.png"
    out_mask_path = XRAY_SILHOUETTE_MASK_DIR / f"{sample_id}_mask.png"
    out_overlay_path = XRAY_SILHOUETTE_OVERLAY_DIR / f"{sample_id}_overlay.png"

    XRAY_SILHOUETTE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    XRAY_SILHOUETTE_MASK_DIR.mkdir(parents=True, exist_ok=True)
    XRAY_SILHOUETTE_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_img_path), silhouette)
    cv2.imwrite(str(out_mask_path), mask)
    cv2.imwrite(str(out_overlay_path), overlay)

    record = {
        "sample_id": sample_id,
        "silhouette_image_path": str(out_img_path),
        "mask_path": str(out_mask_path),
        "overlay_path": str(out_overlay_path),
        "source_axis_image_path": str(image_path),
        "source_filename": row.get("source_filename", ""),
        "side_original": row.get("side_original", ""),
        "flipped_to_left": row.get("flipped_to_left", False),
        "rotated_180": row.get("rotated_180", False),
        "properties": row.get("properties", ""),
        "image_width": IMG_SIZE,
        "image_height": IMG_SIZE,
        "mask_area_ratio": area_ratio,
        "mask_edge_ratio": edge_ratio,
        "HVA": float(row["HVA"]),
        "IMA": float(row["IMA"]),
    }

    axis_cols = [
        "great_toe_x1", "great_toe_y1", "great_toe_x2", "great_toe_y2",
        "first_metatarsal_x1", "first_metatarsal_y1", "first_metatarsal_x2", "first_metatarsal_y2",
        "second_metatarsal_x1", "second_metatarsal_y1", "second_metatarsal_x2", "second_metatarsal_y2",
    ]

    for col in axis_cols:
        if col in row:
            record[col] = float(row[col])

    return record, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(PROCESSED_AXIS_CSV))
    parser.add_argument("--area-min", type=float, default=0.05)
    parser.add_argument("--area-max", type=float, default=0.95)
    args = parser.parse_args()

    make_dirs()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"processed_axis_dataset.csv가 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"sample_id", "image_path", "HVA", "IMA"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"processed axis csv에 필요한 컬럼이 없습니다: {sorted(missing)}")

    print("X-ray silhouette dataset 생성 시작")
    print("=" * 80)
    print(f"Input CSV  : {csv_path}")
    print(f"Image out  : {XRAY_SILHOUETTE_IMAGE_DIR}")
    print(f"Mask out   : {XRAY_SILHOUETTE_MASK_DIR}")
    print(f"Overlay out: {XRAY_SILHOUETTE_OVERLAY_DIR}")
    print(f"Output CSV : {PROCESSED_SILHOUETTE_CSV}")
    print(f"Rows       : {len(df)}")
    print("=" * 80)

    records = []
    errors = []

    for idx, row in df.iterrows():
        record, error = process_one(row, idx)

        if error is not None:
            errors.append({"row_index": idx, "sample_id": row.get("sample_id", ""), "error": error})
            print(f"[SKIP] idx={idx} sample={row.get('sample_id', '')} reason={error}")
            continue

        assert record is not None
        records.append(record)

        if len(records) % 50 == 0:
            print(f"processed: {len(records)} / {len(df)}")

    out_df = pd.DataFrame(records)

    out_df["mask_quality_warning"] = (
        (out_df["mask_area_ratio"] < args.area_min)
        | (out_df["mask_area_ratio"] > args.area_max)
    )

    PROCESSED_SILHOUETTE_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(PROCESSED_SILHOUETTE_CSV, index=False, encoding="utf-8-sig")

    if errors:
        error_csv = PROCESSED_SILHOUETTE_CSV.parent / "prepare_silhouette_errors.csv"
        pd.DataFrame(errors).to_csv(error_csv, index=False, encoding="utf-8-sig")
    else:
        error_csv = None

    print("\n완료")
    print("=" * 80)
    print(f"생성 샘플 수: {len(out_df)}")
    print(f"오류/스킵 수: {len(errors)}")
    print(f"CSV 저장    : {PROCESSED_SILHOUETTE_CSV}")
    print(f"이미지 저장 : {XRAY_SILHOUETTE_IMAGE_DIR}")
    print(f"마스크 저장 : {XRAY_SILHOUETTE_MASK_DIR}")
    print(f"오버레이    : {XRAY_SILHOUETTE_OVERLAY_DIR}")

    if error_csv is not None:
        print(f"오류 CSV    : {error_csv}")

    if len(out_df) > 0:
        print("\nMask 품질 요약")
        print("-" * 80)
        print(f"mask_area_ratio mean: {out_df['mask_area_ratio'].mean():.4f}")
        print(f"mask_area_ratio min : {out_df['mask_area_ratio'].min():.4f}")
        print(f"mask_area_ratio max : {out_df['mask_area_ratio'].max():.4f}")
        print(f"mask_edge_ratio mean: {out_df['mask_edge_ratio'].mean():.4f}")
        print(f"quality warning 개수: {int(out_df['mask_quality_warning'].sum())}")

        print("\nmask_area_ratio 작은 상위 10개")
        print(
            out_df[["sample_id", "mask_area_ratio", "mask_edge_ratio", "overlay_path"]]
            .sort_values("mask_area_ratio", ascending=True)
            .head(10)
            .to_string(index=False)
        )

        print("\nmask_area_ratio 큰 상위 10개")
        print(
            out_df[["sample_id", "mask_area_ratio", "mask_edge_ratio", "overlay_path"]]
            .sort_values("mask_area_ratio", ascending=False)
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
