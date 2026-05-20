import math
from typing import Tuple

import numpy as np


Point = Tuple[float, float]
Line = Tuple[float, float, float, float]


def parse_four_floats(value: object) -> Line:
    if value is None:
        raise ValueError("value is None")

    text = str(value).strip()
    parts = [p.strip() for p in text.split(",")]

    if len(parts) != 4:
        raise ValueError(f"4개 좌표가 아닙니다: {value}")

    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def line_to_points(line: Line) -> Tuple[Point, Point]:
    x1, y1, x2, y2 = line
    return (x1, y1), (x2, y2)


def points_to_line(p1: Point, p2: Point) -> Line:
    return (p1[0], p1[1], p2[0], p2[1])


def transform_point_to_crop(point: Point, box: Line) -> Point:
    x, y = point
    bx1, by1, bx2, by2 = box

    bw = bx2 - bx1
    bh = by2 - by1

    if bw <= 0 or bh <= 0:
        raise ValueError(f"잘못된 box입니다: {box}")

    nx = (x - bx1) / bw
    ny = (y - by1) / bh

    return clip01(nx), clip01(ny)


def transform_line_to_crop(line: Line, box: Line) -> Line:
    p1, p2 = line_to_points(line)
    return points_to_line(transform_point_to_crop(p1, box), transform_point_to_crop(p2, box))


def flip_point_horizontal(point: Point) -> Point:
    x, y = point
    return 1.0 - x, y


def flip_line_horizontal(line: Line) -> Line:
    p1, p2 = line_to_points(line)
    return points_to_line(flip_point_horizontal(p1), flip_point_horizontal(p2))


def rotate_point_180(point: Point) -> Point:
    x, y = point
    return 1.0 - x, 1.0 - y


def rotate_line_180(line: Line) -> Line:
    p1, p2 = line_to_points(line)
    return points_to_line(rotate_point_180(p1), rotate_point_180(p2))


def line_mean_y(line: Line) -> float:
    _, y1, _, y2 = line
    return (y1 + y2) / 2.0


def should_rotate_180(great_toe: Line, first_metatarsal: Line) -> bool:
    return line_mean_y(great_toe) > line_mean_y(first_metatarsal)


def angle_between_lines_deg(line_a: Line, line_b: Line) -> float:
    ax1, ay1, ax2, ay2 = line_a
    bx1, by1, bx2, by2 = line_b

    va = np.array([ax2 - ax1, ay2 - ay1], dtype=np.float64)
    vb = np.array([bx2 - bx1, by2 - by1], dtype=np.float64)

    denom = np.linalg.norm(va) * np.linalg.norm(vb)

    if denom < 1e-12:
        return float("nan")

    cos = float(np.dot(va, vb) / denom)
    cos = max(-1.0, min(1.0, cos))

    angle = math.degrees(math.acos(cos))

    if angle > 90.0:
        angle = 180.0 - angle

    return float(angle)


def line_points_in_unit_box(line: Line) -> bool:
    return all(0.0 <= float(v) <= 1.0 for v in line)