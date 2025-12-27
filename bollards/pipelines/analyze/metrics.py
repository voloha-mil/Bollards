from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from bollards.constants import BBOX_COLS
from bollards.pipelines.analyze.mappings import ensure_image_id


def dataset_summary(df: pd.DataFrame, country_col: str, region_col: Optional[str]) -> dict[str, Any]:
    image_id = ensure_image_id(df)
    n_images = image_id.nunique()
    n_objects = len(df)
    objects_per_image = float(n_objects) / max(n_images, 1)
    out = {
        "n_images": int(n_images),
        "n_objects": int(n_objects),
        "objects_per_image": objects_per_image,
        "n_countries": int(df[country_col].nunique()) if country_col in df.columns else 0,
    }
    if region_col and region_col in df.columns:
        out["n_regions"] = int(df[region_col].nunique())
    return out


def value_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns or df.empty:
        return pd.DataFrame({col: [], "count": []})
    counts = df[col].value_counts(dropna=True)
    if counts.empty:
        return pd.DataFrame({col: [], "count": []})
    return counts.reset_index().rename(columns={"index": col, col: "count"})


def top_bottom(df: pd.DataFrame, col: str, n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    counts = value_counts(df, col)
    if counts.empty:
        return counts, counts
    top = counts.head(n).copy()
    bottom = counts.tail(n).copy()
    return top, bottom


def calc_bbox_area_aspect(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if not all(c in df.columns for c in BBOX_COLS):
        return pd.DataFrame()
    w = (df["x2"] - df["x1"]).clip(lower=0.0)
    h = (df["y2"] - df["y1"]).clip(lower=0.0)
    area = (w * h).clip(lower=0.0)
    aspect = (w / h.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame({
        f"{prefix}_area": area,
        f"{prefix}_aspect": aspect,
    })
    return out


def calc_crop_area_aspect(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if not all(c in df.columns for c in ["crop_w", "crop_h", "orig_w", "orig_h"]):
        return pd.DataFrame()
    area = (df["crop_w"] * df["crop_h"]) / (df["orig_w"] * df["orig_h"]).replace(0, np.nan)
    aspect = df["crop_w"] / df["crop_h"].replace(0, np.nan)
    out = pd.DataFrame({
        f"{prefix}_area": area,
        f"{prefix}_aspect": aspect,
    })
    return out


def compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    total = len(df)
    top1 = float(df["correct_top1"].mean()) if "correct_top1" in df else 0.0
    top5 = float(df["correct_top5"].mean()) if "correct_top5" in df else 0.0
    metrics = {
        "samples": int(total),
        "top1_country": top1,
        "top5_country": top5,
    }
    if "correct_region_top1" in df:
        region_df = df[df["region"].notna()].copy() if "region" in df else df
        if not region_df.empty:
            metrics["top1_region"] = float(region_df["correct_region_top1"].mean())
            metrics["top5_region"] = float(region_df["correct_region_top5"].mean())
    return metrics


def group_accuracy(df: pd.DataFrame, group_col: str, min_support: int) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    grouped = df[df[group_col].notna()].groupby(group_col)["correct_top1"].agg(["mean", "count"]).reset_index()
    grouped = grouped.rename(columns={"mean": "top1", "count": "support"})
    grouped = grouped[grouped["support"] >= min_support].copy()
    grouped = grouped.sort_values("top1", ascending=True)
    return grouped


def confusion_pairs(df: pd.DataFrame, true_col: str, pred_col: str, top_k: int) -> pd.DataFrame:
    if true_col not in df.columns or pred_col not in df.columns:
        return pd.DataFrame()
    mis = df[df[true_col].notna() & df[pred_col].notna() & (df[true_col] != df[pred_col])].copy()
    if mis.empty:
        return pd.DataFrame()
    pairs = mis.groupby([true_col, pred_col]).size().reset_index(name="count")
    pairs = pairs.sort_values("count", ascending=False).head(top_k)
    return pairs
