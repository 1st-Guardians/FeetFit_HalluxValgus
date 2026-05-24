# =============================================================================
# [파일명] predict_foot_mask_filtered.py
# [역할]   학습된 YOLO Segmentation 모델로 발 마스크를 추론(predict)한 뒤,
#          중복 마스크를 IoU 기반으로 제거하고, 결과를 시각화(overlay)하여 저장한다.
#          모델 성능을 눈으로 확인하거나 검증할 때 사용한다.
#
# [파이프라인 순서] 3단계 - 추론 결과 시각화/검증
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │ (1) cvat_xml_to_yolo_seg.py        (학습 데이터 변환)              │
#   │ (2) YOLO 모델 학습                                                │
#   │ (3) predict_foot_mask_filtered.py  ← 현재 파일 (추론 시각화)       │
#   │ (4) save_foot_cutout.py / save_foot_cutout_cropped.py              │
#   │     / save_all_feet_cutout.py      (발 영역 배경 제거 및 저장)      │
#   │ (5) save_foot_contours.py          (발 외곽선 추출)                │
#   │ (6) extract_forefoot_regions.py    (전족부 영역 추출 - 미구현)      │
#   └─────────────────────────────────────────────────────────────────────┘
#
# [사전 준비]
#   - 학습 완료된 YOLO 모델 가중치 파일 (.pt)
#   - 추론할 이미지가 들어있는 폴더
#   - 필요 패키지: pip install ultralytics opencv-python numpy
#
# [실행 방법]
#   코드 하단 if __name__ == "__main__" 블록의 경로를 수정한 후 실행:
#   python src/foot_preprocessing/predict_foot_mask_filtered.py
#
# [수정이 필요한 변수] (코드 하단 __main__ 블록)
#   model_path         : 학습된 YOLO 모델 가중치 경로 (.pt)
#   source_path        : 추론할 이미지 폴더 경로
#   output_dir         : 결과 overlay 이미지 저장 경로
#   conf               : confidence 임계값 (기본: 0.25)
#   mask_iou_threshold : 중복 마스크 제거 IoU 기준 (기본: 0.3)
#
# [출력]
#   output_dir에 마스크가 초록색으로 오버레이된 이미지가 저장됨.
#   각 마스크 위에 "foot 0.95" 형태로 confidence가 표시됨.
# =============================================================================

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def calculate_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return intersection / union


def remove_overlapping_masks(
    masks: list[np.ndarray],
    scores: list[float],
    iou_threshold: float = 0.3
) -> list[int]:
    sorted_indices = sorted(
        range(len(masks)),
        key=lambda i: scores[i],
        reverse=True
    )

    keep_indices = []

    for current_idx in sorted_indices:
        current_mask = masks[current_idx]
        should_keep = True

        for kept_idx in keep_indices:
            kept_mask = masks[kept_idx]
            iou = calculate_mask_iou(current_mask, kept_mask)

            if iou >= iou_threshold:
                should_keep = False
                break

        if should_keep:
            keep_indices.append(current_idx)

    return keep_indices


def predict_with_mask_overlap_filter(
    model_path: str,
    source_path: str,
    output_dir: str,
    conf: float = 0.25,
    imgsz: int = 960,
    mask_iou_threshold: float = 0.3
):
    model = YOLO(model_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=source_path,
        imgsz=imgsz,
        conf=conf,
        device=0,
        save=False,
        verbose=True
    )

    for result in results:
        image_path = Path(result.path)
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[SKIP] image read failed: {image_path}")
            continue

        original_h, original_w = image.shape[:2]

        if result.masks is None or result.boxes is None:
            print(f"[NO MASK] {image_path.name}")
            continue

        # YOLO mask: [N, H, W]
        mask_data = result.masks.data.cpu().numpy()

        # confidence scores
        scores = result.boxes.conf.cpu().numpy().tolist()

        resized_masks = []

        for mask in mask_data:
            mask = (mask > 0.5).astype(np.uint8)

            # result mask 크기가 원본과 다를 수 있으므로 원본 크기로 resize
            mask = cv2.resize(
                mask,
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST
            )

            resized_masks.append(mask)

        keep_indices = remove_overlapping_masks(
            masks=resized_masks,
            scores=scores,
            iou_threshold=mask_iou_threshold
        )

        print(
            f"[{image_path.name}] before={len(resized_masks)}, "
            f"after={len(keep_indices)}, keep={keep_indices}"
        )

        # 최종 마스크 overlay 이미지 생성
        overlay = image.copy()

        for idx in keep_indices:
            mask = resized_masks[idx]

            # 마스크 영역 반투명 표시
            colored = np.zeros_like(image)
            colored[:, :, 1] = 255

            overlay = np.where(
                mask[:, :, None].astype(bool),
                cv2.addWeighted(overlay, 0.6, colored, 0.4, 0),
                overlay
            )

            # 외곽선 그리기
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(
                overlay,
                contours,
                -1,
                (0, 255, 0),
                2
            )

            # confidence 표시
            if contours:
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                cv2.putText(
                    overlay,
                    f"foot {scores[idx]:.2f}",
                    (x, max(y - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        save_path = output_dir / image_path.name
        cv2.imwrite(str(save_path), overlay)

    print(f"Done. Saved to: {output_dir}")


if __name__ == "__main__":
    predict_with_mask_overlap_filter(
        model_path="runs/segment/foot_yolo11n_seg_img960/weights/best.pt",
        source_path="data/foot_yolo_seg/images/val",
        output_dir="outputs/foot_mask_filtered",
        conf=0.25,
        imgsz=960,
        mask_iou_threshold=0.3
    )