# =============================================================================
# [파일명] cvat_xml_to_yolo_seg.py
# [역할]   CVAT에서 내보낸 XML 어노테이션 파일을 YOLO Segmentation 학습용
#          데이터셋 구조(images/labels + train/val 분할)로 변환한다.
#          CVAT의 RLE(Run-Length Encoding) 마스크를 디코딩하여
#          YOLO polygon 형식(.txt)으로 저장하고, data.yaml도 자동 생성한다.
#
# [파이프라인 순서] 1단계 - 학습 데이터 준비
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │ (1) cvat_xml_to_yolo_seg.py   ← 현재 파일 (학습 데이터 변환)       │
#   │ (2) YOLO 모델 학습 (ultralytics CLI 또는 별도 학습 스크립트)         │
#   │ (3) predict_foot_mask_filtered.py  (추론 결과 시각화/검증)          │
#   │ (4) save_foot_cutout.py / save_foot_cutout_cropped.py              │
#   │     / save_all_feet_cutout.py      (발 영역 배경 제거 및 저장)      │
#   │ (5) save_foot_contours.py          (발 외곽선 추출)                │
#   │ (6) extract_forefoot_regions.py    (전족부 영역 추출 - 미구현)      │
#   └─────────────────────────────────────────────────────────────────────┘
#
# [사전 준비]
#   - CVAT에서 "CVAT for images 1.1" 형식으로 어노테이션을 XML로 내보내기
#   - 원본 이미지 폴더 준비
#   - 필요 패키지: pip install opencv-python numpy
#
# [실행 방법]
#   python src/foot_preprocessing/cvat_xml_to_yolo_seg.py \
#       --xml_path data/annotations.xml \
#       --images_dir data/raw/foot_photos/images \
#       --output_dir data/foot_yolo_seg \
#       --label_name foot \
#       --val_ratio 0.2
#
# [주요 옵션]
#   --xml_path        : CVAT에서 내보낸 annotations.xml 파일 경로
#   --images_dir      : 원본 이미지가 들어있는 폴더 경로
#   --output_dir      : 변환된 YOLO 데이터셋 저장 경로 (기본: data/foot_yolo_seg)
#   --label_name      : CVAT에서 사용한 라벨 이름 (기본: "foot")
#   --val_ratio       : 검증 세트 비율 (기본: 0.2 = 20%)
#   --keep_all_contours : 모든 컨투어 유지 (기본: 가장 큰 것만)
#   --save_debug_masks  : 디버깅용 바이너리 마스크 이미지 저장
#
# [출력 구조]
#   data/foot_yolo_seg/
#   ├── images/train/   (학습 이미지)
#   ├── images/val/     (검증 이미지)
#   ├── labels/train/   (학습 라벨 .txt)
#   ├── labels/val/     (검증 라벨 .txt)
#   ├── foot_seg.yaml   (YOLO 학습 설정 파일)
#   └── conversion_log.txt (변환 로그)
# =============================================================================

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert CVAT XML foot masks to YOLO Segmentation dataset."
    )

    parser.add_argument(
        "--xml_path",
        type=str,
        required=True,
        help="Path to CVAT annotations.xml"
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing original images"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/foot_yolo_seg",
        help="Output YOLO segmentation dataset directory"
    )

    parser.add_argument(
        "--label_name",
        type=str,
        default="foot",
        help="CVAT mask label name to use"
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation split ratio"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--min_area",
        type=float,
        default=500.0,
        help="Minimum contour area to keep"
    )

    parser.add_argument(
        "--approx_eps_ratio",
        type=float,
        default=0.0015,
        help="Contour approximation ratio. Smaller = more detailed polygon."
    )

    parser.add_argument(
        "--skip_multi_foot",
        action="store_true",
        help="Skip images that have more than one foot mask"
    )

    parser.add_argument(
        "--keep_all_contours",
        action="store_true",
        help="Keep all valid contours instead of only the largest one"
    )

    parser.add_argument(
        "--save_debug_masks",
        action="store_true",
        help="Save restored binary masks for debugging"
    )

    return parser.parse_args()


def ensure_dirs(output_dir: Path):
    dirs = [
        output_dir / "images" / "train",
        output_dir / "images" / "val",
        output_dir / "labels" / "train",
        output_dir / "labels" / "val",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def find_image_path(images_dir: Path, image_name: str) -> Optional[Path]:
    """
    CVAT XML의 image name이 IMG000001.jpg 형태일 수도 있고,
    subfolder/IMG000001.jpg 형태일 수도 있으므로 둘 다 대응.
    """
    direct_path = images_dir / image_name
    if direct_path.exists():
        return direct_path

    basename = Path(image_name).name

    for ext in IMAGE_EXTENSIONS:
        candidates = list(images_dir.rglob(Path(basename).stem + ext))
        if candidates:
            return candidates[0]

    candidate = list(images_dir.rglob(basename))
    if candidate:
        return candidate[0]

    return None


def decode_cvat_rle_mask(
    rle_text: str,
    mask_width: int,
    mask_height: int
) -> np.ndarray:
    """
    CVAT mask RLE를 binary mask로 복원한다.

    CVAT XML mask는 bbox 영역 기준으로 RLE가 저장된다.
    RLE는 background count부터 시작하고,
    background / foreground가 번갈아 나온다.
    """
    counts = [int(x.strip()) for x in rle_text.split(",") if x.strip()]

    total = mask_width * mask_height
    if sum(counts) != total:
        raise ValueError(
            f"Invalid RLE length. sum(counts)={sum(counts)}, expected={total}"
        )

    flat = np.zeros(total, dtype=np.uint8)

    index = 0
    value = 0

    for count in counts:
        if count > 0:
            if value == 1:
                flat[index:index + count] = 255
            index += count
        value = 1 - value

    mask = flat.reshape((mask_height, mask_width))
    return mask


def restore_full_mask(
    image_width: int,
    image_height: int,
    mask_tag: ET.Element
) -> np.ndarray:
    """
    CVAT mask는 left, top, width, height bbox 내부에만 저장되어 있으므로
    원본 이미지 크기의 full mask로 복원한다.
    """
    left = int(float(mask_tag.attrib["left"]))
    top = int(float(mask_tag.attrib["top"]))
    mask_width = int(float(mask_tag.attrib["width"]))
    mask_height = int(float(mask_tag.attrib["height"]))
    rle_text = mask_tag.attrib["rle"]

    small_mask = decode_cvat_rle_mask(
        rle_text=rle_text,
        mask_width=mask_width,
        mask_height=mask_height
    )

    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    x1 = max(0, left)
    y1 = max(0, top)
    x2 = min(image_width, left + mask_width)
    y2 = min(image_height, top + mask_height)

    small_x1 = x1 - left
    small_y1 = y1 - top
    small_x2 = small_x1 + (x2 - x1)
    small_y2 = small_y1 + (y2 - y1)

    full_mask[y1:y2, x1:x2] = small_mask[small_y1:small_y2, small_x1:small_x2]

    return full_mask


def mask_to_yolo_segments(
    mask: np.ndarray,
    image_width: int,
    image_height: int,
    min_area: float,
    approx_eps_ratio: float,
    keep_all_contours: bool
) -> List[List[float]]:
    """
    Binary mask에서 contour를 뽑고 YOLO Segmentation polygon 형식으로 변환한다.

    YOLO segmentation label format:
    class_id x1 y1 x2 y2 x3 y3 ...
    좌표는 0~1로 normalize.
    """
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        valid_contours.append(contour)

    if not valid_contours:
        return []

    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

    if not keep_all_contours:
        valid_contours = valid_contours[:1]

    segments = []

    for contour in valid_contours:
        perimeter = cv2.arcLength(contour, True)
        epsilon = approx_eps_ratio * perimeter

        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2)

        if len(points) < 3:
            continue

        segment = []

        for x, y in points:
            nx = float(x) / float(image_width)
            ny = float(y) / float(image_height)

            nx = min(max(nx, 0.0), 1.0)
            ny = min(max(ny, 0.0), 1.0)

            segment.extend([nx, ny])

        if len(segment) >= 6:
            segments.append(segment)

    return segments


def write_label_file(label_path: Path, segments: List[List[float]], class_id: int = 0):
    with open(label_path, "w", encoding="utf-8") as f:
        for segment in segments:
            coords = " ".join(f"{v:.6f}" for v in segment)
            f.write(f"{class_id} {coords}\n")


def make_unique_stem(stem: str, used_stems: Dict[str, int]) -> str:
    """
    이미지명이 중복될 경우 라벨 파일명이 덮어써지는 것을 방지.
    """
    if stem not in used_stems:
        used_stems[stem] = 0
        return stem

    used_stems[stem] += 1
    return f"{stem}_{used_stems[stem]}"


def write_yaml(output_dir: Path):
    yaml_text = f"""path: {output_dir.as_posix()}
train: images/train
val: images/val

names:
  0: foot
"""

    yaml_path = output_dir / "foot_seg.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")


def convert():
    args = parse_args()

    xml_path = Path(args.xml_path)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    ensure_dirs(output_dir)

    debug_mask_dir = output_dir / "debug_masks"
    if args.save_debug_masks:
        debug_mask_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_tags = root.findall("image")

    random.seed(args.seed)
    random.shuffle(image_tags)

    val_count = int(len(image_tags) * args.val_ratio)
    val_ids = set(id(tag) for tag in image_tags[:val_count])

    stats = {
        "total_images_in_xml": len(image_tags),
        "converted_images": 0,
        "train_images": 0,
        "val_images": 0,
        "skipped_image_not_found": 0,
        "skipped_no_foot_mask": 0,
        "skipped_multi_foot": 0,
        "skipped_no_valid_contour": 0,
        "created_label_files": 0,
        "created_instances": 0,
    }

    used_stems = {}

    for image_tag in image_tags:
        image_name = image_tag.attrib["name"]
        image_width = int(float(image_tag.attrib["width"]))
        image_height = int(float(image_tag.attrib["height"]))

        image_path = find_image_path(images_dir, image_name)

        if image_path is None:
            stats["skipped_image_not_found"] += 1
            print(f"[SKIP] image not found: {image_name}")
            continue

        foot_masks = [
            tag for tag in image_tag.findall("mask")
            if tag.attrib.get("label") == args.label_name
        ]

        if len(foot_masks) == 0:
            stats["skipped_no_foot_mask"] += 1
            print(f"[SKIP] no foot mask: {image_name}")
            continue

        if args.skip_multi_foot and len(foot_masks) > 1:
            stats["skipped_multi_foot"] += 1
            print(f"[SKIP] multi foot masks: {image_name}")
            continue

        all_segments = []

        merged_debug_mask = np.zeros((image_height, image_width), dtype=np.uint8)

        for mask_tag in foot_masks:
            full_mask = restore_full_mask(
                image_width=image_width,
                image_height=image_height,
                mask_tag=mask_tag
            )

            merged_debug_mask = cv2.bitwise_or(merged_debug_mask, full_mask)

            segments = mask_to_yolo_segments(
                mask=full_mask,
                image_width=image_width,
                image_height=image_height,
                min_area=args.min_area,
                approx_eps_ratio=args.approx_eps_ratio,
                keep_all_contours=args.keep_all_contours
            )

            all_segments.extend(segments)

        if len(all_segments) == 0:
            stats["skipped_no_valid_contour"] += 1
            print(f"[SKIP] no valid contour: {image_name}")
            continue

        split = "val" if id(image_tag) in val_ids else "train"

        original_stem = Path(image_name).stem
        safe_stem = make_unique_stem(original_stem, used_stems)

        target_image_path = output_dir / "images" / split / f"{safe_stem}{image_path.suffix.lower()}"
        target_label_path = output_dir / "labels" / split / f"{safe_stem}.txt"

        shutil.copy2(image_path, target_image_path)

        write_label_file(
            label_path=target_label_path,
            segments=all_segments,
            class_id=0
        )

        if args.save_debug_masks:
            cv2.imwrite(
                str(debug_mask_dir / f"{safe_stem}_mask.png"),
                merged_debug_mask
            )

        stats["converted_images"] += 1
        stats[f"{split}_images"] += 1
        stats["created_label_files"] += 1
        stats["created_instances"] += len(all_segments)

        print(f"[OK] {image_name} -> {split}, instances={len(all_segments)}")

    write_yaml(output_dir)

    log_lines = []
    log_lines.append("CVAT XML to YOLO Segmentation Conversion Log")
    log_lines.append("=" * 60)

    for key, value in stats.items():
        log_lines.append(f"{key}: {value}")

    log_lines.append("")
    log_lines.append(f"output_dir: {output_dir}")
    log_lines.append(f"data_yaml: {output_dir / 'foot_seg.yaml'}")

    log_text = "\n".join(log_lines)

    print()
    print(log_text)

    log_path = output_dir / "conversion_log.txt"
    log_path.write_text(log_text, encoding="utf-8")


if __name__ == "__main__":
    convert()