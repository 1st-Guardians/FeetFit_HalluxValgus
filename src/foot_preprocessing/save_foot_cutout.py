# =============================================================================
# [파일명] save_foot_cutout.py
# [역할]   YOLO 모델로 발 영역을 세그멘테이션하고, 배경을 투명(알파 채널)으로
#          제거한 뒤 원본 크기 그대로 PNG로 저장한다.
#          이미지 한 장에 발이 여러 개 있어도 가장 큰 마스크 1개만 사용한다.
#          (이미지를 crop하지 않음 - crop이 필요하면 save_foot_cutout_cropped.py 사용)
#
# [파이프라인 순서] 4단계 옵션 A - 발 배경 제거 (원본 크기 유지)
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │ (1) cvat_xml_to_yolo_seg.py        (학습 데이터 변환)              │
#   │ (2) YOLO 모델 학습                                                │
#   │ (3) predict_foot_mask_filtered.py  (추론 결과 시각화/검증)          │
#   │ (4) 아래 3개 중 하나 선택:                                         │
#   │     ► save_foot_cutout.py          ← 현재 파일 (1발, 원본 크기)    │
#   │       save_foot_cutout_cropped.py  (1발, 발 영역만 crop)           │
#   │       save_all_feet_cutout.py      (모든 발, 개별 crop)            │
#   │ (5) save_foot_contours.py          (발 외곽선 추출)                │
#   │ (6) extract_forefoot_regions.py    (전족부 영역 추출 - 미구현)      │
#   └─────────────────────────────────────────────────────────────────────┘
#
# [사전 준비]
#   - 학습 완료된 YOLO 모델 가중치 파일 (.pt)
#   - 원본 발 사진이 들어있는 폴더
#   - 필요 패키지: pip install ultralytics opencv-python numpy
#
# [실행 방법]
#   아래 "설정" 섹션의 경로를 본인 환경에 맞게 수정한 후 실행:
#   python src/foot_preprocessing/save_foot_cutout.py
#
# [수정이 필요한 변수]
#   MODEL_PATH : 학습된 YOLO 모델 가중치 경로
#   SOURCE_DIR : 원본 이미지 폴더 경로
#   OUTPUT_DIR : 결과 저장 폴더 경로
#
# [출력]
#   OUTPUT_DIR에 배경이 투명하게 제거된 PNG 이미지가 저장됨.
#   (원본 이미지와 동일한 크기, 발 영역만 불투명)
# =============================================================================

from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# 설정
# =========================
MODEL_PATH = "runs/segment/roboflow_foot_yolo11n_seg-2/weights/best.pt"
SOURCE_DIR = "data/raw/foot_photos/images"
OUTPUT_DIR = "data/processed/foot_cutout"

CONF = 0.25
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =========================
# 함수
# =========================
def save_cutout_image(image_path: Path, model: YOLO, output_dir: Path):
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
    orig_img = result.orig_img  # BGR 이미지 (numpy)
    h, w = orig_img.shape[:2]

    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        print(f"[SKIP] 마스크 없음: {image_path.name}")
        return

    # 마스크가 여러 개면 가장 큰 마스크 1개만 선택
    masks = result.masks.data.cpu().numpy()  # shape: (N, Hm, Wm)

    best_mask = None
    best_area = -1

    for m in masks:
        # 원본 이미지 크기로 맞춤
        m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m_bin = (m_resized > 0.5).astype(np.uint8)
        area = int(m_bin.sum())

        if area > best_area:
            best_area = area
            best_mask = m_bin

    if best_mask is None:
        print(f"[SKIP] 유효한 마스크 없음: {image_path.name}")
        return

    # BGRA 이미지 생성 (A=알파 채널)
    b, g, r = cv2.split(orig_img)
    alpha = (best_mask * 255).astype(np.uint8)
    cutout = cv2.merge([b, g, r, alpha])

    # 저장 파일명: png로 저장
    output_path = output_dir / f"{image_path.stem}.png"
    cv2.imwrite(str(output_path), cutout)
    print(f"[SAVE] {output_path}")


def main():
    model = YOLO(MODEL_PATH)

    source_dir = Path(SOURCE_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in source_dir.iterdir() if p.suffix.lower() in IMG_EXTS]

    if not image_paths:
        print("[ERROR] 이미지가 없습니다.")
        return

    print(f"[INFO] 총 {len(image_paths)}장 처리 시작")

    for image_path in image_paths:
        save_cutout_image(image_path, model, output_dir)

    print("[DONE] 배경 제거 완료")


if __name__ == "__main__":
    main()