from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from src.utils.geometry import Line


COLORS = {
    "great_toe": (255, 0, 255),
    "first_metatarsal": (0, 0, 255),
    "second_metatarsal": (0, 200, 255),
}


def norm_to_pixel(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    px = int(round(float(x) * width))
    py = int(round(float(y) * height))

    px = max(0, min(width - 1, px))
    py = max(0, min(height - 1, py))

    return px, py


def draw_line_norm(img: np.ndarray, line: Line, color: Tuple[int, int, int], label: str, thickness: int = 3) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = line

    p1 = norm_to_pixel(x1, y1, w, h)
    p2 = norm_to_pixel(x2, y2, w, h)

    cv2.line(img, p1, p2, color, thickness)
    cv2.circle(img, p1, 5, color, -1)
    cv2.circle(img, p2, 5, color, -1)
    cv2.putText(img, label, (p1[0] + 8, p1[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def save_axis_overlay(
    image_bgr: np.ndarray,
    great_toe: Line,
    first_metatarsal: Line,
    second_metatarsal: Line,
    hva: float,
    ima: float,
    out_path: Path,
    title: str = "",
) -> None:
    overlay = image_bgr.copy()

    draw_line_norm(overlay, great_toe, COLORS["great_toe"], "great_toe")
    draw_line_norm(overlay, first_metatarsal, COLORS["first_metatarsal"], "first_met")
    draw_line_norm(overlay, second_metatarsal, COLORS["second_metatarsal"], "second_met")

    text = f"{title} | HVA={hva:.2f} | IMA={ima:.2f}"
    cv2.rectangle(overlay, (0, 0), (min(overlay.shape[1], 1000), 42), (0, 0, 0), -1)
    cv2.putText(overlay, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)