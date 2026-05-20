"""
03_prepare_xray_silhouette_dataset.py

X-ray crop 이미지에서 foot silhouette/mask dataset을 생성합니다.

v3 핵심:
1. processed_axis_dataset.csv의 letterbox_pad_x/y/new_w/new_h를 사용해
   실제 X-ray crop 영역만 ROI로 사용
2. ROI 내부에서만 threshold 수행
3. 발이 아래쪽 테두리에 닿는 것은 허용
4. top/left/right 테두리까지 mask가 퍼지는 경우는 배경 선택으로 판단해 감점
5. 발 전체 soft tissue silhouette을 우선 잡도록 bright-object 후보 중심으로 선택
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
    return sorted(contours, key=cv2.contourArea, reverse=True)[0]


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    contour = get_largest_contour(mask)
    if contour is None:
        return mask

    out = np.zeros_like(mask)
    cv2.drawContours(out, [contour], -1, 255, thickness=-1)
    return out


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))

    out = mask.copy()
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel_small, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel_mid, iterations=1)

    out = keep_largest_component(out)

    # 내부 구멍 채우기
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    return out


def edge_ratios(mask: np.ndarray) -> Tuple[float, float, float, float, float]:
    h, w = mask.shape[:2]

    top = np.count_nonzero(mask[0, :]) / max(1, w)
    bottom = np.count_nonzero(mask[-1, :]) / max(1, w)
    left = np.count_nonzero(mask[:, 0]) / max(1, h)
    right = np.count_nonzero(mask[:, -1]) / max(1, h)

    all_edge = np.count_nonzero(
        np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    ) / max(1, (2 * w + 2 * h))

    return float(top), float(bottom), float(left), float(right), float(all_edge)


def score_mask(mask: np.ndarray) -> float:
    h, w = mask.shape[:2]
    total = h * w

    area = np.count_nonzero(mask) / max(1, total)
    contour = get_largest_contour(mask)

    if contour is None:
        return -1e9

    contour_area = cv2.contourArea(contour) / max(1, total)
    x, y, bw, bh = cv2.boundingRect(contour)
    bbox_area = (bw * bh) / max(1, total)

    top, bottom, left, right, all_edge = edge_ratios(mask)

    # 발목/뒤꿈치가 아래쪽에 닿는 것은 허용
    # 대신 top, left, right까지 넓게 닿으면 배경을 잡았을 가능성이 큼
    bad_edge = top + 0.7 * left + 0.7 * right + 0.15 * bottom

    area_score = -abs(area - 0.42)
    contour_score = 0.8 * contour_area
    bbox_score = -0.25 * max(0.0, bbox_area - 0.85)
    edge_penalty = -2.2 * bad_edge

    hard_penalty = 0.0
    if area > 0.82:
        hard_penalty -= 5.0
    if top > 0.45 and left > 0.45 and right > 0.45:
        hard_penalty -= 8.0
    if all_edge > 0.55:
        hard_penalty -= 5.0
    if area < 0.12:
        hard_penalty -= 2.0

    return area_score + contour_score + bbox_score + edge_penalty + hard_penalty


def build_candidates(roi_gray: np.ndarray) -> List[np.ndarray]:
    norm = normalize_to_uint8(roi_gray)
    blur = cv2.GaussianBlur(norm, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(norm)
    clahe_blur = cv2.GaussianBlur(clahe, (5, 5), 0)

    candidates = []

    for src in [blur, clahe_blur]:
        # Otsu bright-object
        _, th = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(th)

        # percentile bright-object 후보
        # 발 soft tissue가 배경보다 밝은 경우를 우선
        for p in [15, 20, 25, 30, 35, 40, 45, 50]:
            t = np.percentile(src, p)
            candidates.append(np.where(src >= t, 255, 0).astype(np.uint8))

        # inverse 후보는 제한적으로만 사용
        # 배경이 밝고 발이 어두운 케이스 대비
        _, inv_th = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        candidates.append(inv_th)

    return candidates


def extract_roi_from_letterbox(img: np.ndarray, row) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    letterbox padding을 제외한 실제 X-ray crop 영역만 반환합니다.
    processed_axis_dataset.csv에 letterbox 값이 없으면 전체 이미지를 사용합니다.
    """
    h, w = img.shape[:2]

    if all(c in row.index for c in ["letterbox_pad_x", "letterbox_pad_y", "letterbox_new_w", "letterbox_new_h"]):
        pad_x = int(row["letterbox_pad_x"])
        pad_y = int(row["letterbox_pad_y"])
        new_w = int(row["letterbox_new_w"])
        new_h = int(row["letterbox_new_h"])

        x1 = max(0, min(w - 1, pad_x))
        y1 = max(0, min(h - 1, pad_y))
        x2 = max(0, min(w, pad_x + new_w))
        y2 = max(0, min(h, pad_y + new_h))

        if x2 > x1 and y2 > y1:
            return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

    return img.copy(), (0, 0, w, h)


def create_mask_for_roi(roi_bgr: np.ndarray) -> Tuple[np.ndarray, str, float]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    candidates = build_candidates(gray)

    best_mask = None
    best_score = -1e18
    best_name = "none"

    for i, raw in enumerate(candidates):
        cleaned = clean_mask(raw)
        s = score_mask(cleaned)

        if s > best_score:
            best_score = s
            best_mask = cleaned
            best_name = f"candidate_{i}"

    if best_mask is None:
        best_mask = np.zeros_like(gray, dtype=np.uint8)

    return best_mask, best_name, float(best_score)


def create_full_mask(img: np.ndarray, row) -> Tuple[np.ndarray, str, float]:
    roi, (x1, y1, x2, y2) = extract_roi_from_letterbox(img, row)
    roi_mask, method, score = create_mask_for_roi(roi)

    full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = roi_mask

    return full_mask, method, score


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

    return cv2.addWeighted(overlay, 0.75, color_mask, 0.25, 0)


def mask_quality(mask: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    h, w = mask.shape[:2]
    area = np.count_nonzero(mask) / max(1, h * w)
    top, bottom, left, right, all_edge = edge_ratios(mask)
    return float(area), top, bottom, left, right, all_edge


def process_one(row, index: int):
    image_path = Path(str(row["image_path"]))

    if not image_path.exists():
        return None, f"image_not_found:{image_path}"

    img = read_image_bgr(image_path)

    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    mask, method, score = create_full_mask(img, row)
    silhouette = create_silhouette_image(mask)
    overlay = create_overlay(img, mask)

    area_ratio, edge_top, edge_bottom, edge_left, edge_right, edge_all = mask_quality(mask)

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
        "mask_edge_top": edge_top,
        "mask_edge_bottom": edge_bottom,
        "mask_edge_left": edge_left,
        "mask_edge_right": edge_right,
        "mask_edge_ratio": edge_all,
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
    parser.add_argument("--area-max", type=float, default=0.82)
    parser.add_argument("--edge-max", type=float, default=0.55)
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

    print("X-ray silhouette dataset 생성 시작 v3")
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

        print("\nmask_area_ratio 작은 상위 10개")
        print(
            out_df[["sample_id", "mask_area_ratio", "mask_edge_ratio", "mask_method", "overlay_path"]]
            .sort_values("mask_area_ratio", ascending=True)
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()