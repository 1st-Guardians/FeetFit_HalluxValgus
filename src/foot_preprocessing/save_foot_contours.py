# =============================================================================
# [파일명] save_foot_contours.py
# [역할]   배경이 제거된 발 cutout 이미지(PNG, 알파 채널 포함)에서 발의
#          외곽선(contour)만 추출하여 투명 배경 위에 검은색 선으로 저장한다.
#          이전 단계에서 생성된 cutout PNG의 알파 채널을 이용하여 발 윤곽을 검출한다.
#
# [파이프라인 순서] 5단계 - 발 외곽선 추출
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │ (1) cvat_xml_to_yolo_seg.py        (학습 데이터 변환)              │
#   │ (2) YOLO 모델 학습                                                │
#   │ (3) predict_foot_mask_filtered.py  (추론 결과 시각화/검증)          │
#   │ (4) save_foot_cutout.py / save_foot_cutout_cropped.py              │
#   │     / save_all_feet_cutout.py      (발 영역 배경 제거 및 저장)      │
#   │ (5) save_foot_contours.py          ← 현재 파일 (외곽선 추출)       │
#   │ (6) extract_forefoot_regions.py    (전족부 영역 추출 - 미구현)      │
#   └─────────────────────────────────────────────────────────────────────┘
#
# [사전 준비]
#   - 4단계에서 생성된 투명 배경 발 cutout PNG 이미지 폴더
#     (기본: data/processed/foot_cutout_all)
#   - 필요 패키지: pip install opencv-python numpy
#     (YOLO 모델 불필요 - 이미 cutout된 이미지만 사용)
#
# [실행 방법]
#   아래 "설정" 섹션의 경로를 본인 환경에 맞게 수정한 후 실행:
#   python src/foot_preprocessing/save_foot_contours.py
#
# [수정이 필요한 변수]
#   SOURCE_DIR        : 입력 cutout PNG 이미지 폴더 경로
#   OUTPUT_DIR        : 결과 저장 폴더 경로
#   LINE_THICKNESS    : 외곽선 두께 (px, 기본: 3)
#   MIN_CONTOUR_AREA  : 무시할 최소 컨투어 면적 (px², 기본: 500)
#
# [출력]
#   OUTPUT_DIR에 발 외곽선만 있는 투명 배경 PNG 이미지가 저장됨.
#   파일명 예시: IMG001_foot1_contour.png
# =============================================================================

from pathlib import Path
import cv2
import numpy as np

# =========================
# 설정
# =========================
SOURCE_DIR = "data/processed/foot_cutout_all"
OUTPUT_DIR = "data/processed/foot_contours"

LINE_THICKNESS = 3
MIN_CONTOUR_AREA = 500

IMG_EXTS = {".png"}


# =========================
# 함수
# =========================
def save_foot_contour(image_path: Path, output_dir: Path):
    # PNG를 alpha 포함해서 읽기
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"[SKIP] 이미지 읽기 실패: {image_path.name}")
        return

    # BGRA인지 확인
    if img.ndim != 3 or img.shape[2] != 4:
        print(f"[SKIP] alpha 채널 없음: {image_path.name}")
        return

    h, w = img.shape[:2]

    # alpha 채널 추출
    alpha = img[:, :, 3]

    # alpha가 있는 부분 = 발 영역
    binary_mask = (alpha > 0).astype(np.uint8) * 255

    # 외곽선 찾기
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print(f"[SKIP] 외곽선 없음: {image_path.name}")
        return

    # 너무 작은 외곽선 제거
    valid_contours = [
        cnt for cnt in contours
        if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA
    ]

    if not valid_contours:
        print(f"[SKIP] 유효한 외곽선 없음: {image_path.name}")
        return

    # 투명 배경 캔버스 생성
    contour_img = np.zeros((h, w, 4), dtype=np.uint8)

    # 외곽선 색상: 검정색 + 불투명
    contour_color = (0, 0, 0, 255)  # BGRA

    # 외곽선 그리기
    cv2.drawContours(
        contour_img,
        valid_contours,
        contourIdx=-1,
        color=contour_color,
        thickness=LINE_THICKNESS
    )

    # 저장
    output_path = output_dir / f"{image_path.stem}_contour.png"
    cv2.imwrite(str(output_path), contour_img)

    print(f"[SAVE] {output_path}")


def main():
    source_dir = Path(SOURCE_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in source_dir.iterdir()
        if p.suffix.lower() in IMG_EXTS
    ]

    if not image_paths:
        print("[ERROR] 처리할 PNG 이미지가 없습니다.")
        return

    print(f"[INFO] 총 {len(image_paths)}장 처리 시작")

    for image_path in image_paths:
        save_foot_contour(image_path, output_dir)

    print("[DONE] 외곽선 저장 완료")


if __name__ == "__main__":
    main()