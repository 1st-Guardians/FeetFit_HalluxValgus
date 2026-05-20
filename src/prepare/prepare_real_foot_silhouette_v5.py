"""
prepare_real_foot_silhouette_v5.py

v4 대비 핵심 수정:
  1. pre_merge_mask : component 분석 전에 큰 kernel(35px) morphological close로
                     발 안의 조각들(발가락 틈새·뒤꿈치 분리)을 사전 병합
  2. 양발 판별 기준 대폭 강화 (모든 조건 동시 충족 필요)
     - 각 component area >= 12% (기존 4.5%)
     - 크기 비율 >= 65% (기존 45%)
     - 각 bbox 세로 스팬 >= 40% 이미지 높이  ← 신규: 발 조각 차단
     - bw/bh <= 1.6 (가로로 납작한 조각 차단)  ← 신규
     - centroid 수평 거리 >= 15% 이미지 폭
     - 수직 bbox 중첩 >= 30%
  3. 작은 fragment 흡수: 가장 큰 component의 20% 미만이고 근접하면 주 component에 병합
  4. 발가락 방향 판별 강화: width-profile mean + std 결합

실행: python src/prepare/prepare_real_foot_silhouette_v5.py
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

# ── 상수 ───────────────────────────────────────────────────────────────────────
MIN_COMP_AREA_RATIO = 0.012        # 컴포넌트 최소 면적 (전체 이미지 대비)

# 양발 판별 – 모두 충족해야 bilateral 인정
BI_MIN_AREA_RATIO  = 0.12          # 각 발 면적 >= 12%
BI_SIZE_RATIO      = 0.65          # 작은발 / 큰발 >= 65%
BI_MIN_HEIGHT_SPAN = 0.40          # 각 발 bbox 높이 >= 이미지 높이의 40%
BI_MAX_ASPECT      = 1.60          # bw/bh <= 1.6  (가로로 납작한 조각 제외)
BI_MIN_HOR_SEP     = 0.15          # 무게중심 수평 거리 >= 이미지 폭의 15%
BI_MIN_VERT_OVL    = 0.30          # 세로 bbox 중첩 >= 30%

PRE_CLOSE_KSIZE    = 35            # 사전 병합용 morphological close kernel
FRAG_MERGE_RATIO   = 0.20          # 주 발의 20% 미만인 nearby fragment는 병합
FRAG_MERGE_DIST_R  = 0.08          # 병합 거리 임계 (이미지 폭의 8%)

LEFT_FOOT_BIG_TOE_SIDE = "right"   # 학습 기준: 왼발 기준, 엄지가 이미지 오른쪽


# ── I/O 헬퍼 ──────────────────────────────────────────────────────────────────
def list_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"폴더 없음: {image_dir}")
    return [p for p in sorted(image_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def norm_uint8(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    p1, p99 = float(np.percentile(g, 1)), float(np.percentile(g, 99))
    if p99 - p1 < 1e-6:
        return np.zeros_like(g, dtype=np.uint8)
    return np.clip((g - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)


# ── Segmentation ──────────────────────────────────────────────────────────────
def _candidate_masks(img: np.ndarray) -> List[np.ndarray]:
    h, w = img.shape[:2]
    gray = norm_uint8(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cands: List[np.ndarray] = []

    # Otsu
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cands += [th, cv2.bitwise_not(th)]

    # Adaptive
    ad = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 51, 2)
    cands += [ad, cv2.bitwise_not(ad)]

    # YCrCb 피부색
    ycr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycr, np.array([0, 133, 77]), np.array([255, 173, 127]))
    cands.append(skin)

    # HSV saturation + value
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    cands.append(((s > 20) & (v > 35)).astype(np.uint8) * 255)

    # GrabCut
    mx = int(w * 0.04); my = int(h * 0.03)
    rect = (mx, my, max(1, w - 2*mx), max(1, h - 2*my))
    try:
        gc = np.zeros((h, w), np.uint8)
        cv2.grabCut(img, gc, rect,
                    np.zeros((1,65), np.float64), np.zeros((1,65), np.float64),
                    5, cv2.GC_INIT_WITH_RECT)
        cands.append(np.where((gc==cv2.GC_FGD)|(gc==cv2.GC_PR_FGD), 255, 0).astype(np.uint8))
    except cv2.error:
        pass

    return cands


def _clean(mask: np.ndarray, keep_largest: bool = False) -> np.ndarray:
    k5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    k25 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    m = mask.copy()
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k25, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k5,  iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k13, iterations=2)
    if keep_largest:
        m = _keep_largest(m)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k25, iterations=1)
    return m


def _keep_largest(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    out = np.zeros_like(mask)
    cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return out


def _edge_ratio(mask: np.ndarray) -> float:
    edge = np.concatenate([mask[0,:], mask[-1,:], mask[:,0], mask[:,-1]])
    return float(np.count_nonzero(edge)) / max(1, edge.size)


def _score_mask(mask: np.ndarray) -> float:
    h, w = mask.shape[:2]
    total = h * w
    ar = np.count_nonzero(mask) / max(1, total)
    er = _edge_ratio(mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return -1e9
    ca = cv2.contourArea(max(cnts, key=cv2.contourArea)) / max(1, total)
    hard = 0.0
    if ar < 0.03 or ar > 0.92:
        hard -= 5.0
    return -abs(ar - 0.35) + 0.8*ca - 0.55*er + hard


def segment_foot(img: np.ndarray) -> Tuple[np.ndarray, str, float]:
    """여러 후보 mask 중 점수가 가장 높은 것 반환"""
    best_mask, best_score, best_method = None, -1e18, "none"
    for i, raw in enumerate(_candidate_masks(img)):
        cleaned = _clean(raw, keep_largest=False)
        s = _score_mask(cleaned)
        if s > best_score:
            best_score, best_mask, best_method = s, cleaned, f"cand_{i}"
    if best_mask is None:
        best_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    return best_mask, best_method, float(best_score)


# ── 사전 병합 (pre-merge) ─────────────────────────────────────────────────────
def pre_merge_mask(mask: np.ndarray) -> np.ndarray:
    """
    큰 kernel morphological close로 발 내부 조각들을 미리 병합.
    발가락 사이 틈, 발 옆면 분리 등을 하나의 blob으로 만든다.
    양발 이미지에서도 두 발 사이 간격(보통 5~15%)은 유지된다.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (PRE_CLOSE_KSIZE, PRE_CLOSE_KSIZE))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)


# ── Component 분석 ────────────────────────────────────────────────────────────
def _get_components(mask: np.ndarray) -> List[Dict]:
    h, w = mask.shape[:2]
    total = h * w
    binary = (mask > 0).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    comps = []
    for lid in range(1, n):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area / total < MIN_COMP_AREA_RATIO:
            continue
        cm = np.zeros_like(mask)
        cm[labels == lid] = 255
        cm = _clean(cm, keep_largest=True)
        cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        comps.append({
            "mask":            cm,
            "area":            area,
            "area_ratio":      area / total,
            "bbox":            (x, y, x+bw, y+bh),
            "bbox_w":          bw,
            "bbox_h":          bh,
            "centroid":        (float(centroids[lid][0]), float(centroids[lid][1])),
        })

    return sorted(comps, key=lambda c: c["area"], reverse=True)


def _merge_fragments(comps: List[Dict], img_shape: Tuple[int, int]) -> List[Dict]:
    """
    주 component(가장 큰 것)의 FRAG_MERGE_RATIO 미만이고
    충분히 가까운 작은 조각은 주 component mask에 병합한다.
    """
    if len(comps) <= 1:
        return comps

    h, w = img_shape
    dist_thr = w * FRAG_MERGE_DIST_R
    main = comps[0]
    merged_mask = main["mask"].copy()
    survivors = [main]

    for c in comps[1:]:
        if c["area_ratio"] < main["area_ratio"] * FRAG_MERGE_RATIO:
            # 거리 확인: centroid 간 유클리드 거리
            dx = c["centroid"][0] - main["centroid"][0]
            dy = c["centroid"][1] - main["centroid"][1]
            dist = (dx**2 + dy**2) ** 0.5
            if dist < dist_thr:
                merged_mask = cv2.bitwise_or(merged_mask, c["mask"])
                continue
        survivors.append(c)

    # 주 component mask 업데이트
    if survivors[0] is main:
        # 병합된 마스크로 교체
        km = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PRE_CLOSE_KSIZE, PRE_CLOSE_KSIZE))
        survivors[0] = {**main,
                        "mask": cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, km)}
    return survivors


# ── 양발/단일발 선택 ──────────────────────────────────────────────────────────
def _is_bilateral_pair(c1: Dict, c2: Dict, img_h: int, img_w: int) -> bool:
    """두 component가 진짜 양발인지 엄격하게 판별"""
    a1, a2 = c1["area_ratio"], c2["area_ratio"]

    # 면적 기준
    if a1 < BI_MIN_AREA_RATIO or a2 < BI_MIN_AREA_RATIO:
        return False
    # 크기 비율
    if a2 / max(a1, 1e-6) < BI_SIZE_RATIO:
        return False

    # 세로 스팬 (각 발이 이미지 높이의 40% 이상을 차지해야 함)
    h_span1 = c1["bbox_h"] / max(1, img_h)
    h_span2 = c2["bbox_h"] / max(1, img_h)
    if h_span1 < BI_MIN_HEIGHT_SPAN or h_span2 < BI_MIN_HEIGHT_SPAN:
        return False

    # 종횡비 – 가로로 납작한 조각(뒤꿈치 조각 등) 제외
    asp1 = c1["bbox_w"] / max(1, c1["bbox_h"])
    asp2 = c2["bbox_w"] / max(1, c2["bbox_h"])
    if asp1 > BI_MAX_ASPECT or asp2 > BI_MAX_ASPECT:
        return False

    # 수평 간격 (무게중심)
    h_sep = abs(c1["centroid"][0] - c2["centroid"][0]) / max(1, img_w)
    if h_sep < BI_MIN_HOR_SEP:
        return False

    # 수직 중첩 (두 발은 비슷한 높이에 있어야 함)
    _, y1a, _, y2a = c1["bbox"]
    _, y1b, _, y2b = c2["bbox"]
    inter = max(0, min(y2a, y2b) - max(y1a, y1b))
    overlap = inter / max(1, min(y2a-y1a, y2b-y1b))
    if overlap < BI_MIN_VERT_OVL:
        return False

    return True


def select_foot_components(mask: np.ndarray) -> Tuple[List[Dict], str]:
    """
    1. pre_merge_mask 로 조각 사전 병합
    2. component 추출 + 소형 인근 fragment 흡수
    3. 엄격한 기준으로 양발/단일발 결정
    """
    h, w = mask.shape[:2]

    merged_mask = pre_merge_mask(mask)
    comps = _get_components(merged_mask)

    if not comps:
        # fallback: 원본 mask 재시도
        comps = _get_components(mask)
    if not comps:
        return [], "none"

    comps = _merge_fragments(comps, (h, w))

    if len(comps) >= 2:
        c1, c2 = comps[0], comps[1]
        if _is_bilateral_pair(c1, c2, h, w):
            selected = sorted([c1, c2], key=lambda c: c["centroid"][0])
            return selected, "bilateral"

    # 단일 발: 가장 큰 component만 사용
    return [comps[0]], "single"


# ── Crop / Resize ─────────────────────────────────────────────────────────────
def crop_padded(img: np.ndarray, mask: np.ndarray,
                bbox: Tuple[int,int,int,int],
                pad_ratio: float = 0.14):
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = bbox
    pad = int(max(x2-x1, y2-y1) * pad_ratio)
    nx1, ny1 = max(0, x1-pad), max(0, y1-pad)
    nx2, ny2 = min(w, x2+pad), min(h, y2+pad)
    return (img[ny1:ny2, nx1:nx2].copy(),
            mask[ny1:ny2, nx1:nx2].copy(),
            (nx1, ny1, nx2, ny2))


def letterbox(img: np.ndarray, mask: np.ndarray, size: int = IMG_SIZE):
    h, w = img.shape[:2]
    scale = min(size / max(w, 1), size / max(h, 1))
    nw, nh = int(round(w*scale)), int(round(h*scale))
    img_r  = cv2.resize(img,  (nw, nh), interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas_i = np.zeros((size, size, 3), dtype=img.dtype)
    canvas_m = np.zeros((size, size),   dtype=np.uint8)
    px, py = (size-nw)//2, (size-nh)//2
    canvas_i[py:py+nh, px:px+nw] = img_r
    canvas_m[py:py+nh, px:px+nw] = mask_r
    return canvas_i, canvas_m, scale, px, py, nw, nh


# ── 발가락 방향 판별 ──────────────────────────────────────────────────────────
def detect_toe_orientation(mask: np.ndarray) -> Tuple[bool, float, float]:
    """
    발가락이 아래쪽에 있으면 rotate=True 반환.

    판별 기준:
      1. 상/하 구간 평균 폭 비교 (기존)
      2. 상/하 구간 폭 표준편차 비교 (발가락 쪽이 더 불규칙)

    발가락 위쪽 기준 (정상):
      - top 구간 (10-40%) 이 ball-of-foot → 비교적 넓고 불규칙
      - bottom 구간 (60-90%) 이 heel → 좁고 규칙적
    """
    h, _ = mask.shape[:2]
    widths = np.array([
        (int(np.where(mask[y,:] > 0)[0].max()) - int(np.where(mask[y,:] > 0)[0].min()) + 1
         if np.count_nonzero(mask[y,:]) > 1 else 0)
        for y in range(h)
    ], dtype=np.float32)

    top    = widths[int(h*0.10):int(h*0.40)]
    bottom = widths[int(h*0.60):int(h*0.90)]

    mean_top = float(top.mean())   if len(top)    else 0.
    mean_bot = float(bottom.mean()) if len(bottom) else 0.
    std_top  = float(top.std())    if len(top)    else 0.
    std_bot  = float(bottom.std()) if len(bottom) else 0.

    # 점수: 음수 → bottom이 더 넓거나 불규칙 → 발가락이 아래 → 회전 필요
    mean_score = mean_top - mean_bot          # >0: top 넓음 (정상)
    std_score  = std_top  - std_bot           # >0: top 불규칙 (정상)
    combined   = 0.65 * mean_score + 0.35 * std_score

    should_rotate = combined < 0.0
    return bool(should_rotate), mean_top, mean_bot


# ── 엄지 방향 추정 ────────────────────────────────────────────────────────────
def _contour_of(mask: np.ndarray) -> Optional[np.ndarray]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def _big_toe_from_shape(mask: np.ndarray) -> Tuple[str, float, Dict]:
    cnt = _contour_of(mask)
    if cnt is None:
        return "unknown", 0., {}
    x, y, w, h = cv2.boundingRect(cnt)
    if w <= 0 or h <= 0:
        return "unknown", 0., {}

    ty1, ty2 = y, min(mask.shape[0], y + int(h*0.45))
    toe = mask[ty1:ty2, x:x+w]
    if toe.size == 0 or not np.any(toe):
        return "unknown", 0., {}

    mid = toe.shape[1] // 2
    l_area = float(np.count_nonzero(toe[:, :mid]))
    r_area = float(np.count_nonzero(toe[:, mid:]))

    top  = toe[:max(1, int(toe.shape[0]*0.60)), :]
    tl   = float(np.count_nonzero(top[:, :mid]))
    tr   = float(np.count_nonzero(top[:, mid:]))

    col = toe.sum(axis=0).astype(np.float32) / 255.
    if len(col) >= 9:
        col = cv2.GaussianBlur(col.reshape(1,-1), (1,9), 0).ravel()
    lp = float(col[:mid].max()) if mid > 0 else 0.
    rp = float(col[mid:].max()) if len(col[mid:]) > 0 else 0.

    eps = 1e-6
    s  = 0.58*(r_area-l_area)/max(eps, l_area+r_area) \
       + 0.28*(tr-tl)/max(eps, tl+tr)                \
       + 0.14*(rp-lp)/max(eps, lp+rp)
    conf = min(1., abs(s)*3.)

    side = "right" if s > 0.05 else ("left" if s < -0.05 else "unknown")
    debug = {"shape_score": float(s),
             "toe_l_area": l_area, "toe_r_area": r_area}
    return side, conf, debug


def _big_toe_from_nail(img: np.ndarray, mask: np.ndarray) -> Tuple[str, float, Dict]:
    cnt = _contour_of(mask)
    if cnt is None:
        return "unknown", 0., {}
    x, y, w, h = cv2.boundingRect(cnt)
    ty2 = min(mask.shape[0], y + int(h*0.42))
    roi = img[y:ty2, x:x+w]
    rm  = mask[y:ty2, x:x+w]
    if roi.size == 0 or not np.any(rm):
        return "unknown", 0., {}

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mv = np.percentile(v[rm>0], 65) if np.count_nonzero(rm) >= 20 else 128
    ms = np.percentile(s[rm>0], 55) if np.count_nonzero(rm) >= 20 else 50
    nail = ((v>=mv) & (s<=ms) & (rm>0)).astype(np.uint8)

    mid  = nail.shape[1]//2
    lc   = float(np.count_nonzero(nail[:, :mid]))
    rc   = float(np.count_nonzero(nail[:, mid:]))
    eps  = 1e-6
    score = (rc-lc)/max(eps, lc+rc)
    conf  = min(1., abs(score)*2.)
    side  = "right" if score>0.10 else ("left" if score<-0.10 else "unknown")
    return side, conf, {"nail_score": float(score)}


def estimate_big_toe_side(img: np.ndarray, mask: np.ndarray) -> Tuple[str, float, Dict]:
    ss, sc, sd = _big_toe_from_shape(mask)
    ns, nc, nd = _big_toe_from_nail(img, mask)
    combined = 0.86*sd.get("shape_score",0.) + 0.14*nd.get("nail_score",0.)
    conf = min(1., abs(combined)*3.)
    side = "right" if combined>0.05 else ("left" if combined<-0.05 else "unknown")
    debug = {**sd, **nd,
             "combined_score":  combined,
             "shape_side":      ss, "shape_conf": sc,
             "nail_side":       ns, "nail_conf":  nc}
    return side, conf, debug


# ── Silhouette / Overlay ──────────────────────────────────────────────────────
def make_silhouette(mask: np.ndarray) -> np.ndarray:
    sil = np.zeros((*mask.shape, 3), dtype=np.uint8)
    sil[mask > 0] = (255, 255, 255)
    return sil


def make_overlay(img: np.ndarray, mask: np.ndarray, text: str) -> np.ndarray:
    ov = img.copy()
    color = np.zeros_like(ov); color[mask>0] = (0,255,0)
    ov = cv2.addWeighted(ov, 0.75, color, 0.25, 0)
    cnt = _contour_of(mask)
    if cnt is not None:
        cv2.drawContours(ov, [cnt], -1, (0,255,255), 2)
    cv2.rectangle(ov, (0,0), (ov.shape[1], 72), (0,0,0), -1)
    y0 = 22
    for line in text.split("|"):
        cv2.putText(ov, line.strip(), (8, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,255,255), 1, cv2.LINE_AA)
        y0 += 22
    return ov


# ── 단일 component 처리 ───────────────────────────────────────────────────────
def process_component(
    src_path: Path,
    img_bgr: np.ndarray,
    comp: Dict,
    comp_idx: int,
    mode: str,
    seg_method: str,
    seg_score: float,
    n_total: int,
) -> Dict:
    # crop + letterbox
    crop_img, crop_mask, crop_bbox = crop_padded(img_bgr, comp["mask"], comp["bbox"])
    lb_img, lb_mask, scale, px, py, nw, nh = letterbox(crop_img, crop_mask, IMG_SIZE)

    # 발가락 방향 정렬
    rotated, top_w, bot_w = detect_toe_orientation(lb_mask)
    if rotated:
        lb_img  = cv2.rotate(lb_img,  cv2.ROTATE_180)
        lb_mask = cv2.rotate(lb_mask, cv2.ROTATE_180)

    # 엄지 방향 → 왼/오른발 추정
    big_toe_side, bt_conf, bt_debug = estimate_big_toe_side(lb_img, lb_mask)

    fallback = ("left" if comp_idx==0 else "right") if mode=="bilateral" else "unknown"
    if big_toe_side == "right":
        side_guess, side_method = "left", "big_toe"
    elif big_toe_side == "left":
        side_guess, side_method = "right", "big_toe"
    else:
        side_guess, side_method = fallback, "fallback"

    # 오른발 → 좌우반전
    flipped = side_guess == "right"
    if flipped:
        lb_img  = cv2.flip(lb_img,  1)
        lb_mask = cv2.flip(lb_mask, 1)

    silhouette = make_silhouette(lb_mask)

    # 파일명
    stem = src_path.stem
    suffix = ("left" if comp_idx==0 else "right") if mode=="bilateral" else side_guess
    sample_id = f"{stem}_{suffix}"

    REAL_FOOT_SILHOUETTE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    REAL_FOOT_SILHOUETTE_MASK_DIR.mkdir(parents=True, exist_ok=True)
    REAL_FOOT_SILHOUETTE_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    p_img = REAL_FOOT_SILHOUETTE_IMAGE_DIR / f"{sample_id}_silhouette.png"
    p_msk = REAL_FOOT_SILHOUETTE_MASK_DIR  / f"{sample_id}_mask.png"
    p_ov  = REAL_FOOT_SILHOUETTE_OVERLAY_DIR / f"{sample_id}_overlay.png"

    cv2.imwrite(str(p_img), silhouette)
    cv2.imwrite(str(p_msk), lb_mask)

    area_r = float(np.count_nonzero(lb_mask)) / (IMG_SIZE*IMG_SIZE)
    edge_r = _edge_ratio(lb_mask)

    ov = make_overlay(
        lb_img, lb_mask,
        f"{sample_id} mode={mode} side={side_guess} method={side_method} | "
        f"big_toe={big_toe_side} conf={bt_conf:.2f} flip={flipped} rot={rotated} | "
        f"shape={bt_debug.get('shape_score',0.):.3f} "
        f"nail={bt_debug.get('nail_score',0.):.3f}"
    )
    cv2.imwrite(str(p_ov), ov)

    x1c, y1c, x2c, y2c = comp["bbox"]
    bx1, by1, bx2, by2 = crop_bbox
    cx, cy = comp["centroid"]

    rec = {
        "sample_id":              sample_id,
        "source_image_path":      str(src_path),
        "silhouette_image_path":  str(p_img),
        "mask_path":              str(p_msk),
        "overlay_path":           str(p_ov),
        "source_mode":            mode,
        "component_index":        comp_idx,
        "total_selected":         n_total,
        "side_guess":             side_guess,
        "side_method":            side_method,
        "big_toe_side":           big_toe_side,
        "big_toe_confidence":     bt_conf,
        "flipped_to_left":        flipped,
        "rotated_180":            rotated,
        "orientation_top_w":      top_w,
        "orientation_bot_w":      bot_w,
        "segmentation_method":    seg_method,
        "segmentation_score":     seg_score,
        "component_area_ratio":   comp["area_ratio"],
        "mask_area_ratio":        area_r,
        "mask_edge_ratio":        edge_r,
        "component_bbox_x1":      x1c, "component_bbox_y1": y1c,
        "component_bbox_x2":      x2c, "component_bbox_y2": y2c,
        "component_centroid_x":   cx,  "component_centroid_y": cy,
        "crop_bbox_x1":           bx1, "crop_bbox_y1": by1,
        "crop_bbox_x2":           bx2, "crop_bbox_y2": by2,
        "letterbox_scale":        scale,
        "letterbox_pad_x":        px,  "letterbox_pad_y": py,
        "letterbox_new_w":        nw,  "letterbox_new_h": nh,
        "image_width":            IMG_SIZE, "image_height": IMG_SIZE,
    }
    rec.update(bt_debug)
    return rec


# ── 이미지 1장 처리 ───────────────────────────────────────────────────────────
def process_image(path: Path) -> List[Dict]:
    img = read_bgr(path)
    raw_mask, method, score = segment_foot(img)
    comps, mode = select_foot_components(raw_mask)

    if not comps:
        raise RuntimeError("발 component를 찾지 못했습니다.")

    comps = sorted(comps, key=lambda c: c["centroid"][0])
    return [
        process_component(path, img, c, i, mode, method, score, len(comps))
        for i, c in enumerate(comps)
    ]


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=str(REAL_FOOT_IMAGE_DIR))
    args = parser.parse_args()

    make_dirs()
    images = list_images(Path(args.image_dir))

    print("실제 발 이미지 silhouette 생성 v5")
    print("=" * 70)
    print(f"입력: {args.image_dir}  ({len(images)} 장)")
    print(f"출력: {REAL_FOOT_SILHOUETTE_CSV}")
    print("=" * 70)

    records, errors = [], []
    for i, p in enumerate(images, 1):
        try:
            recs = process_image(p)
            records.extend(recs)
            modes = "+".join(r["source_mode"] for r in recs)
            sides = "+".join(r["side_guess"]  for r in recs)
            print(f"[{i:3d}/{len(images)}] OK  {p.name:<28} "
                  f"mode={modes}  side={sides}  n={len(recs)}")
        except Exception as e:
            errors.append({"filename": p.name, "path": str(p), "error": str(e)})
            print(f"[{i:3d}/{len(images)}] ERR {p.name}: {e}")

    df = pd.DataFrame(records)
    REAL_FOOT_SILHOUETTE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REAL_FOOT_SILHOUETTE_CSV, index=False, encoding="utf-8-sig")

    if errors:
        ep = REAL_FOOT_SILHOUETTE_CSV.parent / "errors.csv"
        pd.DataFrame(errors).to_csv(ep, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(f"원본 이미지  : {len(images)}")
    print(f"생성 샘플    : {len(records)}")
    print(f"실패 이미지  : {len(errors)}")

    if len(df) > 0:
        print(f"\nsource_mode  :\n{df['source_mode'].value_counts().to_string()}")
        print(f"\nside_guess   :\n{df['side_guess'].value_counts().to_string()}")
        print(f"\nbig_toe_side :\n{df['big_toe_side'].value_counts().to_string()}")
        print(f"\nmask_area_ratio: "
              f"mean={df['mask_area_ratio'].mean():.3f}  "
              f"min={df['mask_area_ratio'].min():.3f}  "
              f"max={df['mask_area_ratio'].max():.3f}")

        low_conf = df[df["big_toe_confidence"] < 0.30].sort_values("big_toe_confidence")
        if len(low_conf) > 0:
            print(f"\n[주의] 엄지 방향 신뢰도 낮은 샘플 ({len(low_conf)}개):")
            print(low_conf[["sample_id","side_guess","big_toe_side",
                             "big_toe_confidence","overlay_path"]]
                  .head(10).to_string(index=False))


if __name__ == "__main__":
    main()
