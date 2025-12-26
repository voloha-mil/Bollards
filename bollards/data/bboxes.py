from __future__ import annotations

import math
from typing import Tuple

import pandas as pd

from bollards.constants import BBOX_COLS


def normalize_bbox_xyxy_px(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    w: int,
    h: int,
) -> Tuple[float, float, float, float]:
    x1n = min(max(x1 / w, 0.0), 1.0)
    x2n = min(max(x2 / w, 0.0), 1.0)
    y1n = min(max(y1 / h, 0.0), 1.0)
    y2n = min(max(y2 / h, 0.0), 1.0)
    x1n, x2n = sorted([x1n, x2n])
    y1n, y2n = sorted([y1n, y2n])
    eps = 1e-6
    if x2n - x1n < eps:
        x2n = min(1.0, x1n + eps)
    if y2n - y1n < eps:
        y2n = min(1.0, y1n + eps)
    return x1n, y1n, x2n, y2n


def bbox_xyxy_norm_to_center(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Tuple[float, float, float, float]:
    xc = 0.5 * (x1 + x2)
    yc = 0.5 * (y1 + y2)
    bw = x2 - x1
    bh = y2 - y1
    return xc, yc, bw, bh


def expand_bbox_xyxy_norm(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    expand: float,
) -> Tuple[float, float, float, float]:
    xc, yc, bw, bh = bbox_xyxy_norm_to_center(x1, y1, x2, y2)
    bw *= expand
    bh *= expand
    ex1 = max(0.0, xc - 0.5 * bw)
    ey1 = max(0.0, yc - 0.5 * bh)
    ex2 = min(1.0, xc + 0.5 * bw)
    ey2 = min(1.0, yc + 0.5 * bh)
    return ex1, ey1, ex2, ey2


def bbox_xyxy_norm_to_pixels(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    w: int,
    h: int,
) -> Tuple[int, int, int, int]:
    px1 = int(round(x1 * w))
    py1 = int(round(y1 * h))
    px2 = int(round(x2 * w))
    py2 = int(round(y2 * h))
    if px2 <= px1:
        px2 = min(w, px1 + 1)
    if py2 <= py1:
        py2 = min(h, py1 + 1)
    return px1, py1, px2, py2


def crop_image_from_norm_bbox(img, x1: float, y1: float, x2: float, y2: float, expand: float) -> object:
    ex1, ey1, ex2, ey2 = expand_bbox_xyxy_norm(x1, y1, x2, y2, expand)
    px1, py1, px2, py2 = bbox_xyxy_norm_to_pixels(ex1, ey1, ex2, ey2, *img.size)
    return img.crop((px1, py1, px2, py2))


def compute_avg_bbox_wh(df: pd.DataFrame, *, label: str) -> tuple[float, float]:
    missing = [c for c in BBOX_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{label} CSV missing required bbox columns: {missing}")

    widths = (df["x2"].astype(float) - df["x1"].astype(float)).clip(lower=0.0)
    heights = (df["y2"].astype(float) - df["y1"].astype(float)).clip(lower=0.0)
    widths = widths.dropna()
    heights = heights.dropna()
    if widths.empty or heights.empty:
        raise ValueError(f"{label} CSV has no valid bbox widths/heights to average.")

    avg_w = float(widths.mean())
    avg_h = float(heights.mean())
    if not math.isfinite(avg_w) or not math.isfinite(avg_h):
        raise ValueError(f"{label} CSV bbox average width/height is not finite.")

    return avg_w, avg_h
