
"""
03_prepare_xray_silhouette_dataset_v2.py

X-ray crop 이미지에서 foot silhouette/mask dataset을 생성합니다.

v1 문제:
- 일부 이미지에서 inverse threshold가 선택되어 배경 전체가 mask로 잡힘
- 이 경우 mask_area_ratio와 mask_edge_ratio가 매우 큼
- 특히 mask_edge_ratio가 0.9 이상이면 테두리 배경까지 mask로 칠해진 상태일 가능성이 큼

v2 개선:
1. edge_ratio가 높은 candidate mask에 강한 penalty 적용
2. 이미지 테두리에 붙은 background component를 제거
3. 여러 threshold candidate를 만들고, area/edge/contour score 기준으로 best 선택
4. mask_quality_warning 조건에 edge_ratio 기준 추가
5. high edge mask가 생성되면 fallback 방식으로 재시도

실행:
  python src/prepare/03_prepare_xray_silhouette_dataset.py
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

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


def mask_area_ratio(mask: np.ndarray) -> float:
    h, w = mask.shape[:2]
    return float(np.count_nonzero(mask)) / float(h * w)


def mask_edge_ratio(mask: np.ndarray) -> float:
    edge_pixels = np.concatenate([
        mask[0, :],
        mask[-1, :],
        mask[:, 0],
        mask[:, -1],
    ])
    return float(np.count_nonzero(edge_pixels)) / float(edge_pixels.size)


def remove_border_components(mask: np.ndarray) -> np.ndarray:
    """
    이미지 테두리에 붙은 connected component를 제거합니다.
    배경이 mask로 잡히는 경우를 줄이기 위한 단계입니다.
    """
    binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return mask

    h, w = mask.shape[:2]
    keep = np.zeros_like(binary)

    for label_id in range(1, num_labels):
        x, y, bw, bh, area = stats[label_id]

        touches_border = (
            x <= 1
            or y <= 1
            or x + bw >= w - 2
            or y + bh >= h - 2
        )

        # 테두리에 붙은 거대한 component는 배경일 가능성이 큼
        if touches_border and area > 0.10 * h * w:
            continue

        keep[labels == label_id] = 1

    return (keep * 255).astype(np.uint8)


def clean_mask(mask: np.ndarray, remove_border: bool = True) -> np.ndarray:
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    cleaned = mask.copy()

    if remove_border:
        cleaned = remove_border_components(cleaned)

    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_small, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_mid, iterations=1)

    contour = get_largest_contour(cleaned)

    if contour is None:
        return cleaned

    largest = np.zeros_like(cleaned)
    cv2.drawContours(largest, [contour], -1, 255, thickness=-1)
    largest = cv2.morphologyEx(largest, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    return largest


def score_mask(mask: np.ndarray) -> float:
    """
    발 silhouette candidate 점수.
    높을수록 좋음.

    핵심:
    - edge_ratio가 높으면 거의 무조건 감점
    - 배경 전체가 선택된 mask는 edge_ratio가 매우 높음
    - 발 영역은 너무 작지도, 너무 크지도 않아야 함
    """
    h, w = mask.shape[:2]
    total_area = h * w

    area = mask_area_ratio(mask)
    edge = mask_edge_ratio(mask)

    contour = get_largest_contour(mask)
    if contour is None:
        return -1e9

    contour_area = cv2.contourArea(contour) / float(total_area)

    x, y, bw, bh = cv2.boundingRect(contour)
    bbox_area = (bw * bh) / float(total_area)

    # 일반적으로 foot crop에서 foot silhouette은 전체 캔버스의 0.20~0.75 정도가 정상인 경우가 많음.
    # 너무 작은 mask와 너무 큰 mask를 동시에 감점.
    area_target = 0.42
    area_score = -abs(area - area_target)

    # edge가 0.35 이상이면 배경까지 잡혔을 가능성이 큼.
    edge_penalty = -2.0 * edge

    # contour와 bbox가 어느 정도 큰 것은 좋지만, bbox가 너무 화면 전체면 감점.
    contour_score = 0.8 * contour_area
    bbox_penalty = -0.5 * max(0.0, bbox_area - 0.80)

    # 극단적 후보는 강한 penalty
    hard_penalty = 0.0
    if area > 0.80:
        hard_penalty -= 5.0
    if edge > 0.45:
        hard_penalty -= 5.0
    if area < 0.12:
        hard_penalty -= 2.0

    return area_score + contour_score + bbox_penalty + edge_penalty + hard_penalty


def build_candidates(norm: np.ndarray) -> List[np.ndarray]:
    """
    여러 threshold 후보를 생성합니다.
    """
    candidates: List[np.ndarray] = []

    blur = cv2.GaussianBlur(norm, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(norm)
    clahe_blur = cv2.GaussianBlur(clahe, (5, 5), 0)

    for src in [blur, clahe_blur]:
        # Otsu
        _, th = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(th)
        candidates.append(cv2.bitwise_not(th))

        # percentile 기반 bright-object 후보
        for p in [20, 25, 30, 35, 40, 45]:
            threshold = np.percentile(src, p)
            candidate = np.where(src >= threshold, 255, 0).astype(np.uint8)
            candidates.append(candidate)

        # percentile 기반 dark-object 후보도 후보로 만들되, edge penalty로 걸러지게 함
        for p in [55, 60, 65, 70, 75]:
            threshold = np.percentile(src, p)
            candidate = np.where(src <= threshold, 255, 0).astype(np.uint8)
            candidates.append(candidate)

    return candidates


def create_mask(gray: np.ndarray) -> Tuple[np.ndarray, str, float]:
    """
    최적 mask를 생성합니다.
    """
    norm = normalize_to_uint8(gray)

    raw_candidates = build_candidates(norm)

    best_mask = None
    best_score = -1e18
    best_name = "none"

    for i, raw in enumerate(raw_candidates):
        cleaned = clean_mask(raw, remove_border=True)
        s = score_mask(cleaned)

        if s > best_score:
            best_score = s
            best_mask = cleaned
            best_name = f"candidate_{i}"

    if best_mask is None:
        best_mask = np.zeros_like(gray, dtype=np.uint8)

    # fallback:
    # 여전히 edge_ratio가 너무 높으면 border component 제거를 한 번 더 강하게 수행
    if mask_edge_ratio(best_mask) > 0.45 or mask_area_ratio(best_mask) > 0.80:
        stronger = remove_border_components(best_mask)
        stronger = clean_mask(stronger, remove_border=True)

        if score_mask(stronger) > score_mask(best_mask):
            best_mask = stronger
            best_name += "_fallback_border_removed"
            best_score = score_mask(best_mask)

    return best_mask, best_name, float(best_score)


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
    return mask_area_ratio(mask), mask_edge_ratio(mask)


def process_one(row, index: int):
    image_path = Path(str(row["image_path"]))

    if not image_path.exists():
        return None, f"image_not_found:{image_path}"

    img = read_image_bgr(image_path)

    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask, method, score = create_mask(gray)
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
        "mask_method": method,
        "mask_score": score,
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
    parser.add_argument("--area-min", type=float, default=0.08)
    parser.add_argument("--area-max", type=float, default=0.80)
    parser.add_argument("--edge-max", type=float, default=0.35)
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

    print("X-ray silhouette dataset 생성 시작 v2")
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
        | (out_df["mask_edge_ratio"] > args.edge_max)
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
        print(f"mask_edge_ratio max : {out_df['mask_edge_ratio'].max():.4f}")
        print(f"quality warning 개수: {int(out_df['mask_quality_warning'].sum())}")

        print("\nmask_area_ratio 큰 상위 10개")
        print(
            out_df[["sample_id", "mask_area_ratio", "mask_edge_ratio", "mask_method", "overlay_path"]]
            .sort_values("mask_area_ratio", ascending=False)
            .head(10)
            .to_string(index=False)
        )

        print("\nmask_edge_ratio 큰 상위 10개")
        print(
            out_df[["sample_id", "mask_area_ratio", "mask_edge_ratio", "mask_method", "overlay_path"]]
            .sort_values("mask_edge_ratio", ascending=False)
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
