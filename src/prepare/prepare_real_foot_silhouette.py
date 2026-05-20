
"""
prepare_real_foot_silhouette.py

실제 외형 발 사진을 silhouette model 입력 형태로 변환하는 전처리 코드입니다.

입력:
  data/raw/real_foot_images

출력:
  data/processed/real_foot_silhouette/images
  data/processed/real_foot_silhouette/masks
  data/processed/real_foot_silhouette/overlays
  data/processed/real_foot_silhouette/real_foot_silhouette.csv

역할:
1. 실제 발 사진 읽기
2. 발 영역 segmentation
3. 가장 큰 발 contour 선택
4. contour bounding box 기준 crop
5. padding 후 512x512 letterbox resize
6. 발가락 방향이 위쪽을 향하도록 180도 보정
7. 파일명에 right/r이 포함되면 좌우 반전하여 left-foot 기준으로 통일
8. silhouette image, mask, overlay, CSV 저장
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.utils.config import (
    REAL_FOOT_IMAGE_DIR,
    REAL_FOOT_SILHOUETTE_IMAGE_DIR,
    REAL_FOOT_SILHOUETTE_MASK_DIR,
    REAL_FOOT_SILHOUETTE_OVERLAY_DIR,
    REAL_FOOT_SILHOUETTE_CSV,
    IMG_SIZE,
    IMAGE_EXTENSIONS,
    make_dirs,
)


def list_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"실제 발 이미지 폴더가 없습니다: {image_dir}")

    return [
        p for p in sorted(image_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def read_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def normalize_to_uint8(gray: np.ndarray) -> np.ndarray:
    gray = gray.astype(np.float32)
    p1 = float(np.percentile(gray, 1))
    p99 = float(np.percentile(gray, 99))

    if p99 - p1 < 1e-6:
        return np.zeros_like(gray, dtype=np.uint8)

    out = (gray - p1) / (p99 - p1)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)


def get_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
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
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    out = mask.copy()
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, large, iterations=2)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, small, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, mid, iterations=1)
    out = keep_largest_component(out)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, large, iterations=2)
    return out


def mask_edge_ratio(mask: np.ndarray) -> float:
    edge = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return float(np.count_nonzero(edge)) / max(1, edge.size)


def score_mask(mask: np.ndarray) -> float:
    h, w = mask.shape[:2]
    total = h * w
    area_ratio = np.count_nonzero(mask) / max(1, total)
    edge_ratio = mask_edge_ratio(mask)

    contour = get_largest_contour(mask)
    if contour is None:
        return -1e9

    contour_area = cv2.contourArea(contour) / max(1, total)
    x, y, bw, bh = cv2.boundingRect(contour)
    bbox_area = (bw * bh) / max(1, total)

    area_score = -abs(area_ratio - 0.35)
    contour_score = 0.8 * contour_area
    bbox_penalty = -0.25 * max(0.0, bbox_area - 0.85)
    edge_penalty = -0.6 * edge_ratio

    hard_penalty = 0.0
    if area_ratio < 0.03:
        hard_penalty -= 5.0
    if area_ratio > 0.90:
        hard_penalty -= 5.0

    return area_score + contour_score + bbox_penalty + edge_penalty + hard_penalty


def candidate_masks(img_bgr: np.ndarray) -> List[np.ndarray]:
    candidates: List[np.ndarray] = []
    h, w = img_bgr.shape[:2]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_norm = normalize_to_uint8(gray)
    blur = cv2.GaussianBlur(gray_norm, (5, 5), 0)

    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(th)
    candidates.append(cv2.bitwise_not(th))

    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        2,
    )
    candidates.append(adaptive)
    candidates.append(cv2.bitwise_not(adaptive))

    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    candidates.append(cv2.inRange(ycrcb, lower, upper))

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, s_ch, v_ch = cv2.split(hsv)
    sat_candidate = np.where((s_ch > 25) & (v_ch > 40), 255, 0).astype(np.uint8)
    candidates.append(sat_candidate)

    rect_margin_x = int(w * 0.08)
    rect_margin_y = int(h * 0.04)
    rect = (
        rect_margin_x,
        rect_margin_y,
        max(1, w - 2 * rect_margin_x),
        max(1, h - 2 * rect_margin_y),
    )

    try:
        gc_mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(img_bgr, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        gc_out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        candidates.append(gc_out)
    except cv2.error:
        pass

    return candidates


def segment_foot(img_bgr: np.ndarray) -> Tuple[np.ndarray, str, float]:
    candidates = candidate_masks(img_bgr)

    best_mask = None
    best_score = -1e18
    best_method = "none"

    for i, raw in enumerate(candidates):
        cleaned = clean_mask(raw)
        s = score_mask(cleaned)

        if s > best_score:
            best_score = s
            best_mask = cleaned
            best_method = f"candidate_{i}"

    if best_mask is None:
        best_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    return best_mask, best_method, float(best_score)


def crop_by_mask(img: np.ndarray, mask: np.ndarray, padding_ratio: float = 0.12) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    contour = get_largest_contour(mask)
    h, w = mask.shape[:2]

    if contour is None:
        return img.copy(), mask.copy(), (0, 0, w, h)

    x, y, bw, bh = cv2.boundingRect(contour)
    pad = int(max(bw, bh) * padding_ratio)

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)

    return img[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def resize_with_padding(img: np.ndarray, mask: np.ndarray, size: int = IMG_SIZE) -> Tuple[np.ndarray, np.ndarray, float, int, int, int, int]:
    h, w = img.shape[:2]
    scale = min(size / max(1, w), size / max(1, h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas_img = np.zeros((size, size, 3), dtype=img.dtype)
    canvas_mask = np.zeros((size, size), dtype=np.uint8)

    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2

    canvas_img[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_resized
    canvas_mask[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = mask_resized

    return canvas_img, canvas_mask, scale, pad_x, pad_y, new_w, new_h


def width_profile_orientation(mask: np.ndarray) -> Tuple[bool, float, float]:
    h, _ = mask.shape[:2]
    widths = []

    for y in range(h):
        xs = np.where(mask[y, :] > 0)[0]
        widths.append((xs.max() - xs.min() + 1) if len(xs) > 0 else 0)

    widths = np.array(widths, dtype=np.float32)
    top = widths[int(h * 0.10):int(h * 0.35)]
    bottom = widths[int(h * 0.65):int(h * 0.90)]

    top_width = float(np.mean(top)) if len(top) > 0 else 0.0
    bottom_width = float(np.mean(bottom)) if len(bottom) > 0 else 0.0

    should_rotate = bottom_width > top_width * 1.15
    return bool(should_rotate), top_width, bottom_width


def guess_side_from_filename(path: Path) -> str:
    name = path.stem.lower()

    right_tokens = ["right", "_r", "-r", " rt", "rt_", "오른", "우측"]
    left_tokens = ["left", "_l", "-l", " lt", "lt_", "왼", "좌측"]

    for t in right_tokens:
        if t in name:
            return "right"

    for t in left_tokens:
        if t in name:
            return "left"

    return "unknown"


def make_silhouette(mask: np.ndarray) -> np.ndarray:
    sil = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    sil[mask > 0] = (255, 255, 255)
    return sil


def make_overlay(img: np.ndarray, mask: np.ndarray, text: str) -> np.ndarray:
    overlay = img.copy()
    color = np.zeros_like(overlay)
    color[mask > 0] = (0, 255, 0)
    overlay = cv2.addWeighted(overlay, 0.75, color, 0.25, 0)

    contour = get_largest_contour(mask)
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, (0, 255, 255), 2)

    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(overlay, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    return overlay


def process_image(path: Path) -> Dict[str, object]:
    original = read_image_bgr(path)
    raw_mask, method, mask_score = segment_foot(original)
    cropped_img, cropped_mask, bbox = crop_by_mask(original, raw_mask, padding_ratio=0.12)

    proc_img, proc_mask, scale, pad_x, pad_y, new_w, new_h = resize_with_padding(cropped_img, cropped_mask, IMG_SIZE)

    rotated_180, top_width, bottom_width = width_profile_orientation(proc_mask)

    if rotated_180:
        proc_img = cv2.rotate(proc_img, cv2.ROTATE_180)
        proc_mask = cv2.rotate(proc_mask, cv2.ROTATE_180)

    side_guess = guess_side_from_filename(path)
    flipped_to_left = False

    if side_guess == "right":
        proc_img = cv2.flip(proc_img, 1)
        proc_mask = cv2.flip(proc_mask, 1)
        flipped_to_left = True

    silhouette = make_silhouette(proc_mask)

    area_ratio = float(np.count_nonzero(proc_mask)) / float(IMG_SIZE * IMG_SIZE)
    edge_ratio = mask_edge_ratio(proc_mask)

    sample_id = path.stem
    out_img_path = REAL_FOOT_SILHOUETTE_IMAGE_DIR / f"{sample_id}_silhouette.png"
    out_mask_path = REAL_FOOT_SILHOUETTE_MASK_DIR / f"{sample_id}_mask.png"
    out_overlay_path = REAL_FOOT_SILHOUETTE_OVERLAY_DIR / f"{sample_id}_overlay.png"

    REAL_FOOT_SILHOUETTE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    REAL_FOOT_SILHOUETTE_MASK_DIR.mkdir(parents=True, exist_ok=True)
    REAL_FOOT_SILHOUETTE_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_img_path), silhouette)
    cv2.imwrite(str(out_mask_path), proc_mask)

    overlay = make_overlay(
        proc_img,
        proc_mask,
        text=f"{sample_id} side={side_guess} flip={flipped_to_left} rot180={rotated_180}",
    )
    cv2.imwrite(str(out_overlay_path), overlay)

    x1, y1, x2, y2 = bbox

    return {
        "sample_id": sample_id,
        "source_image_path": str(path),
        "silhouette_image_path": str(out_img_path),
        "mask_path": str(out_mask_path),
        "overlay_path": str(out_overlay_path),
        "side_guess": side_guess,
        "flipped_to_left": flipped_to_left,
        "rotated_180": rotated_180,
        "orientation_top_width": top_width,
        "orientation_bottom_width": bottom_width,
        "segmentation_method": method,
        "segmentation_score": mask_score,
        "mask_area_ratio": area_ratio,
        "mask_edge_ratio": edge_ratio,
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "letterbox_scale": scale,
        "letterbox_pad_x": pad_x,
        "letterbox_pad_y": pad_y,
        "letterbox_new_w": new_w,
        "letterbox_new_h": new_h,
        "image_width": IMG_SIZE,
        "image_height": IMG_SIZE,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, default=str(REAL_FOOT_IMAGE_DIR))
    args = parser.parse_args()

    make_dirs()

    image_dir = Path(args.image_dir)
    images = list_images(image_dir)

    print("실제 발 이미지 silhouette 생성 시작")
    print("=" * 80)
    print(f"Image dir : {image_dir}")
    print(f"Images    : {len(images)}")
    print(f"Output CSV: {REAL_FOOT_SILHOUETTE_CSV}")
    print("=" * 80)

    records = []
    errors = []

    for idx, path in enumerate(images, start=1):
        try:
            rec = process_image(path)
            records.append(rec)
            print(f"[{idx}/{len(images)}] OK  {path.name}")
        except Exception as e:
            errors.append({"filename": path.name, "path": str(path), "error": str(e)})
            print(f"[{idx}/{len(images)}] ERR {path.name}: {e}")

    out_df = pd.DataFrame(records)
    REAL_FOOT_SILHOUETTE_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(REAL_FOOT_SILHOUETTE_CSV, index=False, encoding="utf-8-sig")

    if errors:
        err_path = REAL_FOOT_SILHOUETTE_CSV.parent / "real_foot_silhouette_errors.csv"
        pd.DataFrame(errors).to_csv(err_path, index=False, encoding="utf-8-sig")
    else:
        err_path = None

    print("\n완료")
    print("=" * 80)
    print(f"성공: {len(records)}")
    print(f"실패: {len(errors)}")
    print(f"CSV 저장    : {REAL_FOOT_SILHOUETTE_CSV}")
    print(f"이미지 저장 : {REAL_FOOT_SILHOUETTE_IMAGE_DIR}")
    print(f"마스크 저장 : {REAL_FOOT_SILHOUETTE_MASK_DIR}")
    print(f"오버레이    : {REAL_FOOT_SILHOUETTE_OVERLAY_DIR}")

    if err_path is not None:
        print(f"오류 CSV    : {err_path}")

    if len(out_df) > 0:
        print("\nMask 품질 요약")
        print("-" * 80)
        print(f"mask_area_ratio mean: {out_df['mask_area_ratio'].mean():.4f}")
        print(f"mask_area_ratio min : {out_df['mask_area_ratio'].min():.4f}")
        print(f"mask_area_ratio max : {out_df['mask_area_ratio'].max():.4f}")
        print(f"mask_edge_ratio mean: {out_df['mask_edge_ratio'].mean():.4f}")
        print("\nside_guess 분포")
        print(out_df["side_guess"].value_counts().to_string())


if __name__ == "__main__":
    main()
