
"""
01_prepare_axis_dataset.py

datasets.csv를 기반으로 X-ray 한 발 단위 axis dataset을 생성합니다.

수행 내용:
1. data/raw/annotations/datasets.csv 읽기
2. boxes 기준으로 left/right 발 crop
3. great_toe / first_metatarsal / second_metatarsal 축 좌표를 crop 기준 0~1 좌표로 변환
4. right 발은 horizontal flip하여 left-foot 기준으로 통일
5. 발가락이 아래를 향한 경우 180도 회전
6. 원본 비율을 유지한 채 letterbox padding 방식으로 512x512 저장
7. padding 적용 후 축 좌표도 최종 512x512 기준 0~1 좌표로 재변환
8. overlay 저장
9. processed_axis_dataset.csv 저장

중요:
- 기존 코드처럼 crop 이미지를 512x512로 강제 resize하면 가로/세로 비율이 깨져 HVA/IMA 각도가 왜곡됩니다.
- 이 버전은 비율 유지 resize + padding을 사용합니다.

실행:
  python src/prepare/01_prepare_axis_dataset.py
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.utils.config import (
    DATASETS_CSV,
    XRAY_IMAGE_DIR,
    AXIS_IMAGE_DIR,
    AXIS_OVERLAY_DIR,
    PROCESSED_AXIS_CSV,
    IMG_SIZE,
    make_dirs,
)
from src.utils.geometry import (
    angle_between_lines_deg,
    flip_line_horizontal,
    line_points_in_unit_box,
    parse_four_floats,
    rotate_line_180,
    should_rotate_180,
    transform_line_to_crop,
)
from src.utils.visualization import save_axis_overlay


Line = Tuple[float, float, float, float]


def read_image(image_path: Path) -> np.ndarray:
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


def crop_by_box(img: np.ndarray, box: Line) -> np.ndarray:
    """
    원본 이미지와 normalized box(x1,y1,x2,y2)를 받아 해당 영역을 crop합니다.
    """
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


def resize_with_padding(
    img: np.ndarray,
    size: int = IMG_SIZE,
) -> Tuple[np.ndarray, float, int, int, int, int]:
    """
    이미지 비율을 유지한 채 size x size canvas에 padding하여 넣습니다.

    반환:
      canvas
      scale
      pad_x
      pad_y
      new_w
      new_h
    """
    h, w = img.shape[:2]

    if h <= 0 or w <= 0:
        raise ValueError(f"잘못된 이미지 크기입니다: width={w}, height={h}")

    scale = min(size / w, size / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size, 3), dtype=resized.dtype)

    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2

    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas, scale, pad_x, pad_y, new_w, new_h


def apply_letterbox_to_point(
    x: float,
    y: float,
    crop_w: int,
    crop_h: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    size: int = IMG_SIZE,
) -> Tuple[float, float]:
    """
    crop 기준 normalized 좌표를 최종 letterbox image 기준 normalized 좌표로 변환합니다.

    crop normalized 좌표:
      x, y in [0, 1]

    crop pixel 좌표:
      px = x * crop_w
      py = y * crop_h

    letterbox pixel 좌표:
      px_new = px * scale + pad_x
      py_new = py * scale + pad_y

    최종 normalized:
      px_new / size
      py_new / size
    """
    px = float(x) * float(crop_w)
    py = float(y) * float(crop_h)

    px_new = px * float(scale) + float(pad_x)
    py_new = py * float(scale) + float(pad_y)

    nx = px_new / float(size)
    ny = py_new / float(size)

    return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))


def apply_letterbox_to_line(
    line: Line,
    crop_w: int,
    crop_h: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    size: int = IMG_SIZE,
) -> Line:
    x1, y1, x2, y2 = line

    nx1, ny1 = apply_letterbox_to_point(
        x1,
        y1,
        crop_w,
        crop_h,
        scale,
        pad_x,
        pad_y,
        size,
    )
    nx2, ny2 = apply_letterbox_to_point(
        x2,
        y2,
        crop_w,
        crop_h,
        scale,
        pad_x,
        pad_y,
        size,
    )

    return nx1, ny1, nx2, ny2


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

    # 1) box 기준 crop
    crop = crop_by_box(img, box)

    # 2) 원본 normalized axis 좌표를 crop 기준 normalized 좌표로 변환
    great_toe_c = transform_line_to_crop(great_toe, box)
    first_met_c = transform_line_to_crop(first_met, box)
    second_met_c = transform_line_to_crop(second_met, box)

    flipped_to_left = False

    # 3) right 발이면 좌우 반전해서 left-foot 기준으로 통일
    if side == "right":
        crop = cv2.flip(crop, 1)
        great_toe_c = flip_line_horizontal(great_toe_c)
        first_met_c = flip_line_horizontal(first_met_c)
        second_met_c = flip_line_horizontal(second_met_c)
        flipped_to_left = True

    # 4) 발가락이 아래 방향이면 180도 회전
    rotated_180 = False
    if should_rotate_180(great_toe_c, first_met_c):
        crop = cv2.rotate(crop, cv2.ROTATE_180)
        great_toe_c = rotate_line_180(great_toe_c)
        first_met_c = rotate_line_180(first_met_c)
        second_met_c = rotate_line_180(second_met_c)
        rotated_180 = True

    # 5) 비율 유지 resize + padding
    crop_h, crop_w = crop.shape[:2]

    (
        crop_resized,
        scale,
        pad_x,
        pad_y,
        new_w,
        new_h,
    ) = resize_with_padding(crop, IMG_SIZE)

    # 6) 좌표도 letterbox padding 기준으로 변환
    great_toe_final = apply_letterbox_to_line(
        great_toe_c,
        crop_w,
        crop_h,
        scale,
        pad_x,
        pad_y,
        IMG_SIZE,
    )
    first_met_final = apply_letterbox_to_line(
        first_met_c,
        crop_w,
        crop_h,
        scale,
        pad_x,
        pad_y,
        IMG_SIZE,
    )
    second_met_final = apply_letterbox_to_line(
        second_met_c,
        crop_w,
        crop_h,
        scale,
        pad_x,
        pad_y,
        IMG_SIZE,
    )

    # 7) 최종 저장 좌표가 unit box 안에 있는지 확인
    valid_axis = (
        line_points_in_unit_box(great_toe_final)
        and line_points_in_unit_box(first_met_final)
        and line_points_in_unit_box(second_met_final)
    )

    # 8) 최종 저장 좌표 기준으로 geometry HVA/IMA 재계산
    # 비율 유지 + padding이면 CSV의 HVA/IMA와 거의 일치해야 함.
    geom_hva = angle_between_lines_deg(great_toe_final, first_met_final)
    geom_ima = angle_between_lines_deg(first_met_final, second_met_final)

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
        great_toe=great_toe_final,
        first_metatarsal=first_met_final,
        second_metatarsal=second_met_final,
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

        "original_crop_width": crop_w,
        "original_crop_height": crop_h,
        "letterbox_scale": scale,
        "letterbox_pad_x": pad_x,
        "letterbox_pad_y": pad_y,
        "letterbox_new_w": new_w,
        "letterbox_new_h": new_h,

        "image_width": IMG_SIZE,
        "image_height": IMG_SIZE,

        "great_toe_x1": great_toe_final[0],
        "great_toe_y1": great_toe_final[1],
        "great_toe_x2": great_toe_final[2],
        "great_toe_y2": great_toe_final[3],

        "first_metatarsal_x1": first_met_final[0],
        "first_metatarsal_y1": first_met_final[1],
        "first_metatarsal_x2": first_met_final[2],
        "first_metatarsal_y2": first_met_final[3],

        "second_metatarsal_x1": second_met_final[0],
        "second_metatarsal_y1": second_met_final[1],
        "second_metatarsal_x2": second_met_final[2],
        "second_metatarsal_y2": second_met_final[3],

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
            errors.append(
                {
                    "row_index": idx,
                    "filename": row.get("filename", ""),
                    "error": error,
                }
            )
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

        print("\nHVA diff 큰 상위 10개")
        print(
            out_df[
                [
                    "sample_id",
                    "source_filename",
                    "side_original",
                    "HVA",
                    "geom_HVA",
                    "hva_diff_from_csv",
                    "IMA",
                    "geom_IMA",
                    "ima_diff_from_csv",
                    "overlay_path",
                ]
            ]
            .sort_values("hva_diff_from_csv", ascending=False)
            .head(10)
            .to_string(index=False)
        )

        print("\nside_original 분포")
        print(out_df["side_original"].value_counts().to_string())

        print("\nrotated_180 분포")
        print(out_df["rotated_180"].value_counts().to_string())

        print("\nflipped_to_left 분포")
        print(out_df["flipped_to_left"].value_counts().to_string())


if __name__ == "__main__":
    main()
