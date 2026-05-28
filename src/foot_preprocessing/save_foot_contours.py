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