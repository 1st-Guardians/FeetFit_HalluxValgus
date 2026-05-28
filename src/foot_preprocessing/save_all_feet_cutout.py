from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# 설정
# =========================
MODEL_PATH = "models/foot_seg_yolo11n_best.pt"
SOURCE_DIR = "data/raw/foot_photos/images"
OUTPUT_DIR = "data/processed/foot_cutout_all"

CONF = 0.25
PADDING = 10
MIN_COMPONENT_AREA = 2000   # 너무 작은 잡음 제거용
IOU_DUP_THRESHOLD = 0.6     # 중복 발 제거 기준
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================
# 유틸 함수
# =========================
def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    if union == 0:
        return 0.0
    return inter / union


def split_connected_components(binary_mask: np.ndarray, min_area: int):
    """
    하나의 binary mask 안에 연결이 끊어진 영역이 여러 개 있으면
    각각 분리해서 반환
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    components = []

    # 0번은 background
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        comp_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        comp_mask[labels == label_idx] = 1
        components.append(comp_mask)

    return components


def crop_rgba_with_mask(orig_img: np.ndarray, binary_mask: np.ndarray, padding: int):
    """
    원본 BGR 이미지 + binary mask -> 배경 제거된 RGBA crop 반환
    """
    h, w = orig_img.shape[:2]

    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w - 1, x_max + padding)
    y_max = min(h - 1, y_max + padding)

    alpha = (binary_mask * 255).astype(np.uint8)
    b, g, r = cv2.split(orig_img)
    rgba = cv2.merge([b, g, r, alpha])

    cropped = rgba[y_min:y_max + 1, x_min:x_max + 1]
    return cropped


def deduplicate_masks(mask_items, iou_threshold=0.6):
    """
    같은 발이 중복 검출된 경우 제거
    mask_items: [{"mask": ..., "score": ...}, ...]
    score 높은 순으로 남김
    """
    mask_items = sorted(mask_items, key=lambda x: x["score"], reverse=True)
    kept = []

    for item in mask_items:
        current_mask = item["mask"]
        is_duplicate = False

        for kept_item in kept:
            iou = mask_iou(current_mask, kept_item["mask"])
            if iou >= iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(item)

    return kept


# =========================
# 메인 처리 함수
# =========================
def save_all_feet_from_image(image_path: Path, model: YOLO, output_dir: Path):
    results = model.predict(
        source=str(image_path),
        conf=CONF,
        save=False,
        verbose=False
    )

    if not results:
        print(f"[SKIP] 결과 없음: {image_path.name}")
        return

    result = results[0]
    orig_img = result.orig_img
    h, w = orig_img.shape[:2]

    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        print(f"[SKIP] 마스크 없음: {image_path.name}")
        return

    raw_masks = result.masks.data.cpu().numpy()

    # confidence score 가져오기
    # masks 개수와 boxes 개수는 보통 동일
    if result.boxes is not None and result.boxes.conf is not None:
        scores = result.boxes.conf.cpu().numpy()
    else:
        scores = np.ones(len(raw_masks), dtype=np.float32)

    all_components = []

    # 1) 예측된 모든 마스크 순회
    for mask_idx, raw_mask in enumerate(raw_masks):
        score = float(scores[mask_idx]) if mask_idx < len(scores) else 1.0

        # 원본 크기로 복원
        mask_resized = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary_mask = (mask_resized > 0.5).astype(np.uint8)

        # 2) 한 마스크 안에 떨어진 영역이 여러 개면 분리
        components = split_connected_components(binary_mask, min_area=MIN_COMPONENT_AREA)

        # component가 없으면 skip
        for comp in components:
            all_components.append({
                "mask": comp,
                "score": score
            })

    if not all_components:
        print(f"[SKIP] 유효한 발 마스크 없음: {image_path.name}")
        return

    # 3) 중복 제거
    final_components = deduplicate_masks(all_components, iou_threshold=IOU_DUP_THRESHOLD)

    # 4) 발마다 따로 저장
    save_count = 0
    for idx, item in enumerate(final_components, start=1):
        foot_mask = item["mask"]
        cropped_rgba = crop_rgba_with_mask(orig_img, foot_mask, padding=PADDING)

        if cropped_rgba is None:
            continue

        out_path = output_dir / f"{image_path.stem}_foot{idx}.png"
        cv2.imwrite(str(out_path), cropped_rgba)
        save_count += 1
        print(f"[SAVE] {out_path}")

    if save_count == 0:
        print(f"[SKIP] 저장된 발 없음: {image_path.name}")
    else:
        print(f"[DONE] {image_path.name}: {save_count}개 발 저장")


def main():
    model = YOLO(MODEL_PATH)

    source_dir = Path(SOURCE_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in source_dir.iterdir() if p.suffix.lower() in IMG_EXTS]

    if not image_paths:
        print("[ERROR] 처리할 이미지가 없습니다.")
        return

    print(f"[INFO] 총 {len(image_paths)}장 처리 시작")

    for image_path in image_paths:
        save_all_feet_from_image(image_path, model, output_dir)

    print("[DONE] 전체 처리 완료")


if __name__ == "__main__":
    main()