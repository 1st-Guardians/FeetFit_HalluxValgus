
"""
prepare_real_foot_silhouette_v4.py

실제 외형 발 사진을 silhouette model 입력 형태로 변환하는 전처리 코드입니다.

v4 수정 핵심:
1. v3의 bilateral_split 기능을 제거했습니다.
   - 단일 오른발/왼발을 억지로 좌우 분리하면서 발의 옆부분, 뒤꿈치, 새끼발가락 일부를 별도 발로 오인하는 문제가 있었기 때문입니다.
2. 양발 판별 조건을 매우 보수적으로 변경했습니다.
   - 실제로 분리된 큰 발 component 2개가 있을 때만 bilateral로 처리합니다.
   - 한 발 안의 발가락 일부/발 옆면/뒤꿈치 조각은 별도 발로 보지 않습니다.
3. 단일 발은 무조건 하나의 발 샘플로 crop합니다.
4. crop 이후 발가락 방향을 먼저 위로 정렬하고, 그 다음 엄지 방향으로 왼발/오른발을 자동 추정합니다.
5. 오른발로 추정되면 left-foot 기준으로 좌우반전합니다.

입력:
  data/raw/real_foot_images

출력:
  data/processed/real_foot_silhouette/images
  data/processed/real_foot_silhouette/masks
  data/processed/real_foot_silhouette/overlays
  data/processed/real_foot_silhouette/real_foot_silhouette.csv

실행:
  python src/prepare/prepare_real_foot_silhouette.py
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


MIN_COMPONENT_AREA_RATIO = 0.012
TWO_FEET_MIN_AREA_RATIO = 0.045
LEFT_FOOT_BIG_TOE_SIDE = "right"


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


def clean_mask(mask: np.ndarray, keep_only_largest: bool = False) -> np.ndarray:
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

    out = mask.copy()
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, large, iterations=2)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, small, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, mid, iterations=2)

    if keep_only_largest:
        out = keep_largest_component(out)

    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, large, iterations=1)
    return out


def mask_edge_ratio(mask: np.ndarray) -> float:
    edge = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return float(np.count_nonzero(edge)) / max(1, edge.size)


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
    skin = cv2.inRange(ycrcb, lower, upper)
    candidates.append(skin)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, s_ch, v_ch = cv2.split(hsv)
    sat_candidate = np.where((s_ch > 25) & (v_ch > 40), 255, 0).astype(np.uint8)
    candidates.append(sat_candidate)

    rect_margin_x = int(w * 0.04)
    rect_margin_y = int(h * 0.03)
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


def get_components(mask: np.ndarray) -> List[Dict[str, object]]:
    h, w = mask.shape[:2]
    total = h * w

    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    components = []

    for label_id in range(1, num_labels):
        x, y, bw, bh, area = stats[label_id]
        area_ratio = float(area) / float(total)

        if area_ratio < MIN_COMPONENT_AREA_RATIO:
            continue

        comp_mask = np.zeros_like(mask)
        comp_mask[labels == label_id] = 255
        comp_mask = clean_mask(comp_mask, keep_only_largest=True)

        contour = get_largest_contour(comp_mask)
        if contour is None:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        cx, cy = centroids[label_id]
        bbox_area_ratio = float(bw * bh) / float(total)

        components.append(
            {
                "label_id": label_id,
                "mask": comp_mask,
                "area": int(area),
                "area_ratio": area_ratio,
                "bbox_area_ratio": bbox_area_ratio,
                "bbox": (int(x), int(y), int(x + bw), int(y + bh)),
                "bbox_width": int(bw),
                "bbox_height": int(bh),
                "centroid": (float(cx), float(cy)),
            }
        )

    components = sorted(components, key=lambda c: c["area"], reverse=True)
    return components


def component_vertical_overlap(c1: Dict[str, object], c2: Dict[str, object]) -> float:
    _, y1a, _, y2a = c1["bbox"]
    _, y1b, _, y2b = c2["bbox"]

    inter = max(0, min(y2a, y2b) - max(y1a, y1b))
    h1 = max(1, y2a - y1a)
    h2 = max(1, y2b - y1b)

    return inter / float(min(h1, h2))


def component_horizontal_gap_ratio(c1: Dict[str, object], c2: Dict[str, object], image_width: int) -> float:
    cx1 = c1["centroid"][0]
    cx2 = c2["centroid"][0]
    return abs(cx2 - cx1) / max(1.0, float(image_width))


def is_valid_single_foot_component(component: Dict[str, object], image_shape: Tuple[int, int]) -> bool:
    h, w = image_shape
    bw = component["bbox_width"]
    bh = component["bbox_height"]
    area_ratio = component["area_ratio"]

    if area_ratio < TWO_FEET_MIN_AREA_RATIO:
        return False

    if bh < h * 0.25 and bw < w * 0.25:
        return False

    aspect = bw / max(1, bh)

    # 너무 납작한 조각, 너무 얇은 조각은 발 하나로 보기 어려움
    if aspect > 2.3:
        return False
    if aspect < 0.12:
        return False

    return True


def score_mask(mask: np.ndarray) -> float:
    h, w = mask.shape[:2]
    total = h * w
    area_ratio = np.count_nonzero(mask) / max(1, total)
    edge_ratio = mask_edge_ratio(mask)
    components = get_components(mask)

    if not components:
        return -1e9

    largest = components[0]
    contour = get_largest_contour(largest["mask"])

    if contour is None:
        return -1e9

    contour_area = cv2.contourArea(contour) / max(1, total)
    bbox_area = largest["bbox_area_ratio"]

    # 큰 component 1개 또는 보수적 양발 2개가 좋은 후보
    comp_bonus = 0.0
    if len(components) >= 2:
        c1, c2 = components[0], components[1]
        overlap = component_vertical_overlap(c1, c2)
        sep = component_horizontal_gap_ratio(c1, c2, w)
        if (
            c1["area_ratio"] >= TWO_FEET_MIN_AREA_RATIO
            and c2["area_ratio"] >= TWO_FEET_MIN_AREA_RATIO
            and c2["area_ratio"] >= c1["area_ratio"] * 0.45
            and overlap >= 0.35
            and sep >= 0.22
        ):
            comp_bonus += 0.25

    area_score = -abs(area_ratio - 0.35)
    contour_score = 0.8 * contour_area
    bbox_penalty = -0.25 * max(0.0, bbox_area - 0.90)
    edge_penalty = -0.55 * edge_ratio

    hard_penalty = 0.0
    if area_ratio < 0.03:
        hard_penalty -= 5.0
    if area_ratio > 0.92:
        hard_penalty -= 5.0

    return area_score + contour_score + bbox_penalty + edge_penalty + comp_bonus + hard_penalty


def segment_foot_candidates(img_bgr: np.ndarray) -> Tuple[np.ndarray, str, float]:
    candidates = candidate_masks(img_bgr)

    best_mask = None
    best_score = -1e18
    best_method = "none"

    for i, raw in enumerate(candidates):
        cleaned = clean_mask(raw, keep_only_largest=False)
        s = score_mask(cleaned)

        if s > best_score:
            best_score = s
            best_mask = cleaned
            best_method = f"candidate_{i}"

    if best_mask is None:
        best_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    return best_mask, best_method, float(best_score)


def select_foot_components(mask: np.ndarray) -> Tuple[List[Dict[str, object]], str]:
    """
    v4:
    - 강제 bilateral_split 제거.
    - 실제로 분리된 큰 component 2개가 있을 때만 bilateral.
    - 한 발의 발가락 조각/뒤꿈치/옆부분 조각은 single foot로 분리하지 않음.
    """
    h, w = mask.shape[:2]
    components = get_components(mask)

    if len(components) == 0:
        return [], "none"

    valid = [c for c in components if is_valid_single_foot_component(c, (h, w))]

    if len(valid) >= 2:
        c1, c2 = valid[0], valid[1]

        area1 = float(c1["area_ratio"])
        area2 = float(c2["area_ratio"])
        overlap = component_vertical_overlap(c1, c2)
        sep = component_horizontal_gap_ratio(c1, c2, w)

        # 매우 보수적으로 양발 인정
        if (
            area1 >= TWO_FEET_MIN_AREA_RATIO
            and area2 >= TWO_FEET_MIN_AREA_RATIO
            and area2 >= area1 * 0.45
            and overlap >= 0.35
            and sep >= 0.22
        ):
            selected = sorted([c1, c2], key=lambda c: c["centroid"][0])
            return selected, "bilateral"

    # 양발 조건이 명확하지 않으면 하나의 발로 처리
    return [components[0]], "single"


def crop_by_bbox(img: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int], padding_ratio: float = 0.14):
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1
    pad = int(max(bw, bh) * padding_ratio)

    nx1 = max(0, x1 - pad)
    ny1 = max(0, y1 - pad)
    nx2 = min(w, x2 + pad)
    ny2 = min(h, y2 + pad)

    return img[ny1:ny2, nx1:nx2].copy(), mask[ny1:ny2, nx1:nx2].copy(), (nx1, ny1, nx2, ny2)


def resize_with_padding(img: np.ndarray, mask: np.ndarray, size: int = IMG_SIZE):
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


def estimate_big_toe_side_from_shape(mask: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
    contour = get_largest_contour(mask)
    if contour is None:
        return "unknown", 0.0, {}

    x, y, w, h = cv2.boundingRect(contour)

    if w <= 0 or h <= 0:
        return "unknown", 0.0, {}

    toe_y1 = y
    toe_y2 = min(mask.shape[0], y + int(h * 0.45))
    toe_x1 = x
    toe_x2 = x + w

    toe = mask[toe_y1:toe_y2, toe_x1:toe_x2]

    if toe.size == 0 or np.count_nonzero(toe) == 0:
        return "unknown", 0.0, {}

    mid = toe.shape[1] // 2
    left = toe[:, :mid]
    right = toe[:, mid:]

    left_area = float(np.count_nonzero(left))
    right_area = float(np.count_nonzero(right))

    top_part = toe[:max(1, int(toe.shape[0] * 0.60)), :]
    top_left = top_part[:, :mid]
    top_right = top_part[:, mid:]

    top_left_area = float(np.count_nonzero(top_left))
    top_right_area = float(np.count_nonzero(top_right))

    col_mass = toe.sum(axis=0).astype(np.float32) / 255.0
    if len(col_mass) >= 9:
        col_mass = cv2.GaussianBlur(col_mass.reshape(1, -1), (1, 9), 0).ravel()

    left_mass_peak = float(col_mass[:mid].max()) if mid > 0 else 0.0
    right_mass_peak = float(col_mass[mid:].max()) if len(col_mass[mid:]) > 0 else 0.0

    eps = 1e-6
    area_score = (right_area - left_area) / max(eps, left_area + right_area)
    top_score = (top_right_area - top_left_area) / max(eps, top_left_area + top_right_area)
    peak_score = (right_mass_peak - left_mass_peak) / max(eps, left_mass_peak + right_mass_peak)

    combined = 0.58 * area_score + 0.28 * top_score + 0.14 * peak_score
    confidence = min(1.0, abs(combined) * 3.0)

    if combined > 0.05:
        side = "right"
    elif combined < -0.05:
        side = "left"
    else:
        side = "unknown"

    debug = {
        "toe_left_area": left_area,
        "toe_right_area": right_area,
        "toe_top_left_area": top_left_area,
        "toe_top_right_area": top_right_area,
        "toe_left_mass_peak": left_mass_peak,
        "toe_right_mass_peak": right_mass_peak,
        "big_toe_shape_score": float(combined),
    }

    return side, float(confidence), debug


def estimate_big_toe_side_from_nail(img_bgr: np.ndarray, mask: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
    contour = get_largest_contour(mask)
    if contour is None:
        return "unknown", 0.0, {}

    x, y, w, h = cv2.boundingRect(contour)

    toe_y1 = y
    toe_y2 = min(mask.shape[0], y + int(h * 0.42))
    toe_x1 = x
    toe_x2 = x + w

    roi_img = img_bgr[toe_y1:toe_y2, toe_x1:toe_x2]
    roi_mask = mask[toe_y1:toe_y2, toe_x1:toe_x2]

    if roi_img.size == 0 or np.count_nonzero(roi_mask) == 0:
        return "unknown", 0.0, {}

    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    _, s_ch, v_ch = cv2.split(hsv)

    masked_v = v_ch[roi_mask > 0]
    masked_s = s_ch[roi_mask > 0]

    if len(masked_v) < 20:
        return "unknown", 0.0, {}

    v_thr = np.percentile(masked_v, 65)
    s_thr = np.percentile(masked_s, 55)

    nail_candidate = ((v_ch >= v_thr) & (s_ch <= s_thr) & (roi_mask > 0)).astype(np.uint8)

    mid = nail_candidate.shape[1] // 2
    left_count = float(np.count_nonzero(nail_candidate[:, :mid]))
    right_count = float(np.count_nonzero(nail_candidate[:, mid:]))

    eps = 1e-6
    score = (right_count - left_count) / max(eps, left_count + right_count)
    confidence = min(1.0, abs(score) * 2.0)

    if score > 0.10:
        side = "right"
    elif score < -0.10:
        side = "left"
    else:
        side = "unknown"

    debug = {
        "nail_left_count": left_count,
        "nail_right_count": right_count,
        "big_toe_nail_score": float(score),
    }

    return side, float(confidence), debug


def estimate_big_toe_side(img_bgr: np.ndarray, mask: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
    shape_side, shape_conf, shape_debug = estimate_big_toe_side_from_shape(mask)
    nail_side, nail_conf, nail_debug = estimate_big_toe_side_from_nail(img_bgr, mask)

    shape_score = shape_debug.get("big_toe_shape_score", 0.0)
    nail_score = nail_debug.get("big_toe_nail_score", 0.0)

    combined_score = 0.86 * shape_score + 0.14 * nail_score
    confidence = min(1.0, abs(combined_score) * 3.0)

    if combined_score > 0.05:
        side = "right"
    elif combined_score < -0.05:
        side = "left"
    else:
        side = "unknown"

    debug = {}
    debug.update(shape_debug)
    debug.update(nail_debug)
    debug["big_toe_combined_score"] = float(combined_score)
    debug["big_toe_shape_side"] = shape_side
    debug["big_toe_shape_confidence"] = shape_conf
    debug["big_toe_nail_side"] = nail_side
    debug["big_toe_nail_confidence"] = nail_conf

    return side, float(confidence), debug


def side_from_big_toe_side(big_toe_side: str, fallback_side: str = "unknown") -> Tuple[str, str]:
    if big_toe_side == "right":
        return "left", "big_toe_shape"
    if big_toe_side == "left":
        return "right", "big_toe_shape"
    return fallback_side, "fallback"


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

    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 72), (0, 0, 0), -1)
    y0 = 22
    for line in text.split("|"):
        cv2.putText(overlay, line.strip(), (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        y0 += 22

    return overlay


def process_component(
    source_path: Path,
    img_bgr: np.ndarray,
    component: Dict[str, object],
    component_index: int,
    mode: str,
    method: str,
    segmentation_score: float,
    total_components: int,
) -> Dict[str, object]:
    comp_mask = component["mask"]
    bbox = component["bbox"]

    cropped_img, cropped_mask, crop_bbox = crop_by_bbox(img_bgr, comp_mask, bbox, padding_ratio=0.14)
    proc_img, proc_mask, scale, pad_x, pad_y, new_w, new_h = resize_with_padding(cropped_img, cropped_mask, IMG_SIZE)

    rotated_180, top_width, bottom_width = width_profile_orientation(proc_mask)

    if rotated_180:
        proc_img = cv2.rotate(proc_img, cv2.ROTATE_180)
        proc_mask = cv2.rotate(proc_mask, cv2.ROTATE_180)

    big_toe_side, big_toe_confidence, big_toe_debug = estimate_big_toe_side(proc_img, proc_mask)

    fallback_side = "unknown"
    if mode == "bilateral":
        fallback_side = "left" if component_index == 0 else "right"

    side_guess, side_method = side_from_big_toe_side(big_toe_side, fallback_side=fallback_side)

    flipped_to_left = False

    if side_guess == "right":
        proc_img = cv2.flip(proc_img, 1)
        proc_mask = cv2.flip(proc_mask, 1)
        flipped_to_left = True

    silhouette = make_silhouette(proc_mask)

    source_stem = source_path.stem

    if mode == "bilateral":
        suffix = "left" if component_index == 0 else "right"
        sample_id = f"{source_stem}_{suffix}"
    else:
        sample_id = f"{source_stem}_{side_guess}"

    out_img_path = REAL_FOOT_SILHOUETTE_IMAGE_DIR / f"{sample_id}_silhouette.png"
    out_mask_path = REAL_FOOT_SILHOUETTE_MASK_DIR / f"{sample_id}_mask.png"
    out_overlay_path = REAL_FOOT_SILHOUETTE_OVERLAY_DIR / f"{sample_id}_overlay.png"

    REAL_FOOT_SILHOUETTE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    REAL_FOOT_SILHOUETTE_MASK_DIR.mkdir(parents=True, exist_ok=True)
    REAL_FOOT_SILHOUETTE_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_img_path), silhouette)
    cv2.imwrite(str(out_mask_path), proc_mask)

    area_ratio = float(np.count_nonzero(proc_mask)) / float(IMG_SIZE * IMG_SIZE)
    edge_ratio = mask_edge_ratio(proc_mask)

    overlay = make_overlay(
        proc_img,
        proc_mask,
        text=(
            f"{sample_id} mode={mode} side={side_guess} method={side_method} | "
            f"big_toe={big_toe_side} conf={big_toe_confidence:.2f} flip={flipped_to_left} rot={rotated_180} | "
            f"shape={big_toe_debug.get('big_toe_shape_score', 0.0):.3f} "
            f"nail={big_toe_debug.get('big_toe_nail_score', 0.0):.3f}"
        ),
    )
    cv2.imwrite(str(out_overlay_path), overlay)

    x1, y1, x2, y2 = crop_bbox
    bx1, by1, bx2, by2 = bbox
    cx, cy = component["centroid"]

    record = {
        "sample_id": sample_id,
        "source_image_path": str(source_path),
        "silhouette_image_path": str(out_img_path),
        "mask_path": str(out_mask_path),
        "overlay_path": str(out_overlay_path),

        "source_mode": mode,
        "component_index": component_index,
        "total_selected_components": total_components,

        "side_guess": side_guess,
        "side_method": side_method,
        "big_toe_side": big_toe_side,
        "big_toe_confidence": big_toe_confidence,
        "flipped_to_left": flipped_to_left,
        "rotated_180": rotated_180,

        "orientation_top_width": top_width,
        "orientation_bottom_width": bottom_width,

        "segmentation_method": method,
        "segmentation_score": segmentation_score,
        "component_area_ratio": component["area_ratio"],
        "component_bbox_area_ratio": component["bbox_area_ratio"],

        "mask_area_ratio": area_ratio,
        "mask_edge_ratio": edge_ratio,

        "component_bbox_x1": bx1,
        "component_bbox_y1": by1,
        "component_bbox_x2": bx2,
        "component_bbox_y2": by2,
        "component_centroid_x": cx,
        "component_centroid_y": cy,

        "crop_bbox_x1": x1,
        "crop_bbox_y1": y1,
        "crop_bbox_x2": x2,
        "crop_bbox_y2": y2,

        "letterbox_scale": scale,
        "letterbox_pad_x": pad_x,
        "letterbox_pad_y": pad_y,
        "letterbox_new_w": new_w,
        "letterbox_new_h": new_h,

        "image_width": IMG_SIZE,
        "image_height": IMG_SIZE,
    }

    for k, v in big_toe_debug.items():
        record[k] = v

    return record


def process_image(path: Path) -> List[Dict[str, object]]:
    img_bgr = read_image_bgr(path)
    full_mask, method, segmentation_score = segment_foot_candidates(img_bgr)

    selected_components, mode = select_foot_components(full_mask)

    if len(selected_components) == 0:
        raise RuntimeError("발 component를 찾지 못했습니다.")

    selected_components = sorted(selected_components, key=lambda c: c["centroid"][0])
    records = []

    for component_index, component in enumerate(selected_components):
        rec = process_component(
            source_path=path,
            img_bgr=img_bgr,
            component=component,
            component_index=component_index,
            mode=mode,
            method=method,
            segmentation_score=segmentation_score,
            total_components=len(selected_components),
        )
        records.append(rec)

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, default=str(REAL_FOOT_IMAGE_DIR))
    args = parser.parse_args()

    make_dirs()

    image_dir = Path(args.image_dir)
    images = list_images(image_dir)

    print("실제 발 이미지 silhouette 생성 시작 v4")
    print("=" * 80)
    print(f"Image dir : {image_dir}")
    print(f"Images    : {len(images)}")
    print(f"Output CSV: {REAL_FOOT_SILHOUETTE_CSV}")
    print("=" * 80)

    records: List[Dict[str, object]] = []
    errors = []

    for idx, path in enumerate(images, start=1):
        try:
            recs = process_image(path)
            records.extend(recs)
            print(f"[{idx}/{len(images)}] OK  {path.name} -> {len(recs)} sample(s)")
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
    print(f"원본 이미지 수 : {len(images)}")
    print(f"생성 샘플 수   : {len(records)}")
    print(f"실패 이미지 수 : {len(errors)}")
    print(f"CSV 저장      : {REAL_FOOT_SILHOUETTE_CSV}")
    print(f"이미지 저장   : {REAL_FOOT_SILHOUETTE_IMAGE_DIR}")
    print(f"마스크 저장   : {REAL_FOOT_SILHOUETTE_MASK_DIR}")
    print(f"오버레이      : {REAL_FOOT_SILHOUETTE_OVERLAY_DIR}")

    if err_path is not None:
        print(f"오류 CSV      : {err_path}")

    if len(out_df) > 0:
        print("\n샘플 유형 분포")
        print("-" * 80)
        print(out_df["source_mode"].value_counts().to_string())

        print("\nside_guess 분포")
        print(out_df["side_guess"].value_counts().to_string())

        print("\nside_method 분포")
        print(out_df["side_method"].value_counts().to_string())

        print("\nbig_toe_side 분포")
        print(out_df["big_toe_side"].value_counts().to_string())

        print("\nMask 품질 요약")
        print("-" * 80)
        print(f"mask_area_ratio mean: {out_df['mask_area_ratio'].mean():.4f}")
        print(f"mask_area_ratio min : {out_df['mask_area_ratio'].min():.4f}")
        print(f"mask_area_ratio max : {out_df['mask_area_ratio'].max():.4f}")
        print(f"mask_edge_ratio mean: {out_df['mask_edge_ratio'].mean():.4f}")

        print("\n확인 필요 샘플 상위 10개")
        print(
            out_df[[
                "sample_id",
                "source_mode",
                "side_guess",
                "side_method",
                "big_toe_side",
                "big_toe_confidence",
                "mask_area_ratio",
                "mask_edge_ratio",
                "overlay_path",
            ]]
            .sort_values("big_toe_confidence", ascending=True)
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
