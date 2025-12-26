from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from bollards.constants import BBOX_COLS, LABEL_COL, META_COLS, PATH_COL
from bollards.data.country_names import golden_country_to_code


def class_name(cls_id: int, names: Optional[list[str]]) -> str:
    if names and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return f"class_{cls_id}"


def build_country_mappings(
    id_to_country: Optional[list[str]],
    main_df: pd.DataFrame,
) -> Tuple[Optional[list[str]], Optional[Dict[str, int]]]:
    if id_to_country:
        country_to_id = {name: idx for idx, name in enumerate(id_to_country) if name}
        return id_to_country, country_to_id

    if "country" not in main_df.columns or LABEL_COL not in main_df.columns:
        return None, None

    pairs = main_df[[LABEL_COL, "country"]].dropna().drop_duplicates()
    if pairs.empty:
        return None, None

    dup_country = pairs.groupby("country")[LABEL_COL].nunique()
    if (dup_country > 1).any():
        bad = dup_country[dup_country > 1].index.tolist()
        raise ValueError(f"Multiple ids for country names in main data: {bad}")

    dup_id = pairs.groupby(LABEL_COL)["country"].nunique()
    if (dup_id > 1).any():
        bad = dup_id[dup_id > 1].index.tolist()
        raise ValueError(f"Multiple country names for ids in main data: {bad}")

    country_to_id = {str(row["country"]): int(row[LABEL_COL]) for _, row in pairs.iterrows()}
    max_id = max(country_to_id.values())
    id_to_country = [""] * (max_id + 1)
    for name, idx in country_to_id.items():
        id_to_country[idx] = name

    return id_to_country, country_to_id


def prepare_golden_df_for_classifier(
    golden_df: pd.DataFrame,
    country_to_id: Dict[str, int],
    *,
    avg_bbox_w: float,
    avg_bbox_h: float,
) -> pd.DataFrame:
    if avg_bbox_w is None or avg_bbox_h is None:
        raise ValueError("Golden dataset requires explicit avg_bbox_w/avg_bbox_h values.")

    required = [PATH_COL, "country"]
    missing = [c for c in required if c not in golden_df.columns]
    if missing:
        raise ValueError(f"Golden CSV missing required columns: {missing}")

    df = golden_df.copy()
    df["country_code"] = df["country"].apply(golden_country_to_code)
    if df["country_code"].isna().any():
        unknown = sorted(df.loc[df["country_code"].isna(), "country"].dropna().unique().tolist())
        print(f"[warn] golden CSV has unmapped countries; dropping: {unknown}")
        df = df.loc[df["country_code"].notna()].copy()

    df[LABEL_COL] = df["country_code"].map(country_to_id)
    if df[LABEL_COL].isna().any():
        unknown_codes = sorted(df.loc[df[LABEL_COL].isna(), "country_code"].dropna().unique().tolist())
        print(f"[warn] golden CSV has countries not in label map; dropping codes: {unknown_codes}")
        df = df.loc[df[LABEL_COL].notna()].copy()

    if df.empty:
        raise ValueError("Golden CSV has no rows after mapping countries to labels.")

    df[LABEL_COL] = df[LABEL_COL].astype(int)

    bbox_defaults = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
    meta_defaults = {
        "x_center": 0.5,
        "y_center": 0.5,
        "w": avg_bbox_w,
        "h": avg_bbox_h,
        "conf": 1.0,
    }
    for col in BBOX_COLS:
        df[col] = bbox_defaults[col]
    for col in META_COLS:
        df[col] = meta_defaults[col]

    df = df.drop(columns=["country_code"])
    return df


def maybe_add_region_by_country(
    df: pd.DataFrame,
    country_col: str,
    region_map: Optional[Dict[str, str]],
    out_col: str = "region",
) -> pd.DataFrame:
    if not region_map:
        return df
    df = df.copy()
    df[out_col] = df[country_col].map(region_map)
    return df


def build_region_map(
    *,
    region_map_json: Optional[str],
    golden_df: Optional[pd.DataFrame],
) -> Optional[Dict[str, str]]:
    if region_map_json:
        with open(region_map_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): str(v) for k, v in data.items()}

    if golden_df is None:
        return None

    if "continent" not in golden_df.columns or "country" not in golden_df.columns:
        return None

    region_map: Dict[str, str] = {}
    for _, row in golden_df[["continent", "country"]].dropna().iterrows():
        code = golden_country_to_code(str(row["country"]))
        if not code:
            continue
        region = str(row["continent"]).strip()
        if not region:
            continue
        if code in region_map and region_map[code] != region:
            continue
        region_map[code] = region
    return region_map or None


def ensure_image_id(df: pd.DataFrame) -> pd.Series:
    if "image_id" in df.columns:
        return df["image_id"].astype(str)
    if "orig_sha1" in df.columns:
        return df["orig_sha1"].astype(str)
    if PATH_COL in df.columns:
        return df[PATH_COL].apply(lambda p: Path(str(p)).stem)
    return pd.Series(["unknown"] * len(df))
