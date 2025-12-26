from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_hist(values: Iterable[float], path: Path, title: str, xlabel: str, bins: int = 30) -> None:
    vals = [v for v in values if np.isfinite(v)]
    if not vals:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=bins, color="#4c78a8", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def plot_bar(df: pd.DataFrame, path: Path, title: str, x_col: str, y_col: str, max_items: int = 20) -> None:
    if df.empty:
        return
    data = df.head(max_items)
    plt.figure(figsize=(7, 4))
    plt.barh(data[x_col].astype(str), data[y_col].astype(float), color="#54a24b")
    plt.title(title)
    plt.xlabel(y_col)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()
