"""
datasets.csv를 기반으로 X-ray 한 발 단위 axis dataset을 생성합니다.

수행 내용:
1. data/raw/annotations/datasets.csv 읽기
2. boxes 기준으로 left/right 발 crop
3. great_toe / first_metatarsal / second_metatarsal 축 좌표를 crop 기준 0~1 좌표로 변환
4. right 발은 horizontal flip하여 left-foot 기준으로 통일
5. 발가락이 아래를 향한 경우 180도 회전
6. 512x512 resize 이미지 저장
7. overlay 저장
8. processed_axis_dataset.csv 저장

실행:
  python src/prepare/01_prepare_axis_dataset.py
"""

import argparse
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd

from src.utils.config import DATASETS_CSV, XRAY_IMAGE_DIR, AXIS_IMAGE_DIR, AXIS_OVERLAY_DIR, PROCESSED_AXIS_CSV, IMG_SIZE, make_dirs
from src.utils.geometry import angle_between_lines_deg, flip_line_horizontal, line_points_in_unit_box, parse_four_floats, rotate_line_180, should_rotate_180, transform_line_to_crop
from src.utils.visualization import save_axis_overlay


def read_image(image_path: Path):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

    return img


def find_image_path(filename: str) -> Optional[Path]:
    candidate = XRAY_IMAGE_DIR / filename

    if candidate.exists():
        return candidate

    stem = Path(filename).stem
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = XRAY_IMAGE_DIR / f"{stem}{ext}"
        if p.exists():
            return p

    return None


def crop_by_box(img, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box

    px1 = int(round(x1 * w))
    py1 = int(round(y1 * h))
    px2 = int(round(x2 * w))
    py2 = int(round(y2 * h))

    px1 = max(0, min(w - 1, px1))
    px2 = max(0, min(w, px2))
    py1 = max(0, min(h - 1, py1))
    py2 = max(0, min(h, py2))

    if px2 <= px1 or py2 <= py1:
        raise ValueError(f"잘못된 crop box pixel 좌표입니다: {(px1, py1, px2, py2)}")

    return img[py1:py2, px1:px2].copy()


def resize_square(img, size: int = IMG_SIZE):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def process_row(row, index: int, exclude_truncated: bool = False):
    filename = str(row["filename"])
    side = str(row["labels"]).strip().lower()
    properties = "" if pd.isna(row.get("properties", "")) else str(row.get("properties", ""))

    if exclude_truncated and "truncated" in properties.lower():
        return None, "excluded_truncated"

    image_path = find_image_path(filename)
    if image_path is None:
        return None, f"image_not_found:{filename}"

    img = read_image(image_path)

    box = parse_four_floats(row["boxes"])
    great_toe = parse_four_floats(row["great_toe"])
    first_met = parse_four_floats(row["first_metatarsal"])
    second_met = parse_four_floats(row["second_metatarsal"])

    crop = crop_by_box(img, box)

    great_toe_c = transform_line_to_crop(great_toe, box)
    first_met_c = transform_line_to_crop(first_met, box)
    second_met_c = transform_line_to_crop(second_met, box)

    flipped_to_left = False

    if side == "right":
        crop = cv2.flip(crop, 1)
        great_toe_c = flip_line_horizontal(great_toe_c)
        first_met_c = flip_line_horizontal(first_met_c)
        second_met_c = flip_line_horizontal(second_met_c)
        flipped_to_left = True

    rotated_180 = False
    if should_rotate_180(great_toe_c, first_met_c):
        crop = cv2.rotate(crop, cv2.ROTATE_180)
        great_toe_c = rotate_line_180(great_toe_c)
        first_met_c = rotate_line_180(first_met_c)
        second_met_c = rotate_line_180(second_met_c)
        rotated_180 = True

    crop_resized = resize_square(crop, IMG_SIZE)

    valid_axis = (
        line_points_in_unit_box(great_toe_c)
        and line_points_in_unit_box(first_met_c)
        and line_points_in_unit_box(second_met_c)
    )

    geom_hva = angle_between_lines_deg(great_toe_c, first_met_c)
    geom_ima = angle_between_lines_deg(first_met_c, second_met_c)

    hva = float(row["HVA"])
    ima = float(row["IMA"])

    sample_id = f"{Path(filename).stem}_{side}_{index:05d}"
    if flipped_to_left:
        sample_id += "_flipped"
    if rotated_180:
        sample_id += "_rot180"

    out_img_path = AXIS_IMAGE_DIR / f"{sample_id}.png"
    out_overlay_path = AXIS_OVERLAY_DIR / f"{sample_id}_overlay.png"

    AXIS_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    AXIS_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_img_path), crop_resized)

    save_axis_overlay(
        image_bgr=crop_resized,
        great_toe=great_toe_c,
        first_metatarsal=first_met_c,
        second_metatarsal=second_met_c,
        hva=hva,
        ima=ima,
        out_path=out_overlay_path,
        title=sample_id,
    )

    record = {
        "sample_id": sample_id,
        "image_path": str(out_img_path),
        "overlay_path": str(out_overlay_path),
        "source_filename": filename,
        "source_path": str(image_path),
        "side_original": side,
        "flipped_to_left": flipped_to_left,
        "rotated_180": rotated_180,
        "properties": properties,
        "image_width": IMG_SIZE,
        "image_height": IMG_SIZE,

        "great_toe_x1": great_toe_c[0],
        "great_toe_y1": great_toe_c[1],
        "great_toe_x2": great_toe_c[2],
        "great_toe_y2": great_toe_c[3],

        "first_metatarsal_x1": first_met_c[0],
        "first_metatarsal_y1": first_met_c[1],
        "first_metatarsal_x2": first_met_c[2],
        "first_metatarsal_y2": first_met_c[3],

        "second_metatarsal_x1": second_met_c[0],
        "second_metatarsal_y1": second_met_c[1],
        "second_metatarsal_x2": second_met_c[2],
        "second_metatarsal_y2": second_met_c[3],

        "HVA": hva,
        "IMA": ima,
        "geom_HVA": geom_hva,
        "geom_IMA": geom_ima,
        "hva_diff_from_csv": abs(geom_hva - hva),
        "ima_diff_from_csv": abs(geom_ima - ima),
        "valid_axis": valid_axis,
    }

    return record, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(DATASETS_CSV))
    parser.add_argument("--exclude-truncated", action="store_true")
    args = parser.parse_args()

    make_dirs()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"datasets.csv가 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {
        "filename",
        "image_width",
        "image_height",
        "boxes",
        "labels",
        "great_toe",
        "first_metatarsal",
        "second_metatarsal",
        "HVA",
        "IMA",
    }
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"datasets.csv에 필요한 컬럼이 없습니다: {sorted(missing)}")

    records = []
    errors = []

    print("Axis dataset 생성 시작")
    print("=" * 70)
    print(f"CSV        : {csv_path}")
    print(f"Image dir  : {XRAY_IMAGE_DIR}")
    print(f"Output img : {AXIS_IMAGE_DIR}")
    print(f"Output csv : {PROCESSED_AXIS_CSV}")
    print(f"Rows       : {len(df)}")
    print("=" * 70)

    for idx, row in df.iterrows():
        record, error = process_row(row, idx, exclude_truncated=args.exclude_truncated)

        if error is not None:
            errors.append({"row_index": idx, "filename": row.get("filename", ""), "error": error})
            print(f"[SKIP] idx={idx} file={row.get('filename', '')} reason={error}")
            continue

        assert record is not None
        records.append(record)

        if len(records) % 50 == 0:
            print(f"processed: {len(records)} / {len(df)}")

    out_df = pd.DataFrame(records)
    PROCESSED_AXIS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(PROCESSED_AXIS_CSV, index=False, encoding="utf-8-sig")

    if errors:
        error_csv = PROCESSED_AXIS_CSV.parent / "prepare_axis_errors.csv"
        pd.DataFrame(errors).to_csv(error_csv, index=False, encoding="utf-8-sig")
    else:
        error_csv = None

    print("\n완료")
    print("=" * 70)
    print(f"생성 샘플 수: {len(out_df)}")
    print(f"오류/스킵 수: {len(errors)}")
    print(f"CSV 저장    : {PROCESSED_AXIS_CSV}")
    print(f"이미지 저장 : {AXIS_IMAGE_DIR}")
    print(f"오버레이    : {AXIS_OVERLAY_DIR}")
    if error_csv is not None:
        print(f"오류 CSV    : {error_csv}")

    if len(out_df) > 0:
        print("\nHVA/IMA 재계산 검증")
        print(f"HVA diff mean: {out_df['hva_diff_from_csv'].mean():.6f}")
        print(f"HVA diff max : {out_df['hva_diff_from_csv'].max():.6f}")
        print(f"IMA diff mean: {out_df['ima_diff_from_csv'].mean():.6f}")
        print(f"IMA diff max : {out_df['ima_diff_from_csv'].max():.6f}")
        print("\nside_original 분포")
        print(out_df["side_original"].value_counts().to_string())
        print("\nrotated_180 분포")
        print(out_df["rotated_180"].value_counts().to_string())
        print("\nflipped_to_left 분포")
        print(out_df["flipped_to_left"].value_counts().to_string())


if __name__ == "__main__":
    main()