from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import pandas as pd

from bollards.constants import BBOX_COLS, LABEL_COL, META_COLS, PATH_COL
from bollards.data.country_names import golden_country_to_code


def _build_country_mappings(
    id_to_country: Optional[list[str]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Tuple[Optional[list[str]], Optional[Dict[str, int]]]:
    if id_to_country:
        country_to_id = {name: idx for idx, name in enumerate(id_to_country) if name}
        return id_to_country, country_to_id

    combined = pd.concat([train_df, val_df], ignore_index=True)
    if "country" not in combined.columns or LABEL_COL not in combined.columns:
        return None, None

    pairs = combined[[LABEL_COL, "country"]].dropna().drop_duplicates()
    if pairs.empty:
        return None, None

    dup_country = pairs.groupby("country")[LABEL_COL].nunique()
    if (dup_country > 1).any():
        bad = dup_country[dup_country > 1].index.tolist()
        raise ValueError(f"Multiple ids for country names in training data: {bad}")

    dup_id = pairs.groupby(LABEL_COL)["country"].nunique()
    if (dup_id > 1).any():
        bad = dup_id[dup_id > 1].index.tolist()
        raise ValueError(f"Multiple country names for ids in training data: {bad}")

    country_to_id = {str(row["country"]): int(row[LABEL_COL]) for _, row in pairs.iterrows()}
    max_id = max(country_to_id.values())
    id_to_country = [""] * (max_id + 1)
    for name, idx in country_to_id.items():
        id_to_country[idx] = name

    return id_to_country, country_to_id


def _compute_sqrt_inv_class_weights(labels: pd.Series, num_classes: int) -> list[float]:
    counts = labels.value_counts().reindex(range(num_classes), fill_value=0).sort_index()
    weights = []
    for c in counts.tolist():
        if c > 0:
            weights.append(1.0 / math.sqrt(float(c)))
        else:
            weights.append(0.0)
    nonzero = [w for w in weights if w > 0]
    if nonzero:
        mean = sum(nonzero) / len(nonzero)
        weights = [w / mean if w > 0 else 0.0 for w in weights]
    return weights


def _prepare_golden_df(
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


def _sample_diverse_rows(df: pd.DataFrame, n: int, label_col: str, seed: int = 42) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.head(0).copy()
    if n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if label_col not in df.columns:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    labels = df[label_col].dropna().unique().tolist()
    if not labels:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    if n <= len(labels):
        chosen = pd.Series(labels).sample(n=n, random_state=seed).tolist()
        sampled = (
            df[df[label_col].isin(chosen)]
            .groupby(label_col, group_keys=False)
            .sample(n=1, random_state=seed)
        )
        return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    base = df.groupby(label_col, group_keys=False).sample(n=1, random_state=seed)
    remaining = n - len(base)
    if remaining > 0:
        rest = df.drop(index=base.index)
        if not rest.empty:
            extra = rest.sample(n=min(remaining, len(rest)), random_state=seed)
            base = pd.concat([base, extra], ignore_index=True)

    return base.sample(frac=1.0, random_state=seed).reset_index(drop=True)
