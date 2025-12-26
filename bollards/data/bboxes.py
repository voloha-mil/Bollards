from __future__ import annotations

import math

import pandas as pd

from bollards.constants import BBOX_COLS


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
