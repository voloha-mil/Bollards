from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_STYLE_SET = False


def _apply_plot_style() -> None:
    global _STYLE_SET
    if _STYLE_SET:
        return
    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.0)
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#2b2b2b",
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "font.size": 11,
    })
    _STYLE_SET = True


def plot_hist(values: Iterable[float], path: Path, title: str, xlabel: str, bins: int = 30) -> None:
    vals = [v for v in values if np.isfinite(v)]
    if not vals:
        return
    _apply_plot_style()
    plt.figure(figsize=(6, 4))
    sns.histplot(vals, bins=bins, color="#4c78a8", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def plot_count_hist(
    values: Iterable[float],
    path: Path,
    title: str,
    xlabel: str,
    max_ticks: int = 12,
) -> None:
    vals = [int(v) for v in values if np.isfinite(v)]
    if not vals:
        return
    _apply_plot_style()
    min_v = min(vals)
    max_v = max(vals)
    plt.figure(figsize=(6, 4))
    if max_v - min_v <= 40:
        bins = np.arange(min_v - 0.5, max_v + 1.5, 1)
        sns.histplot(vals, bins=bins, color="#4c78a8", edgecolor="white")
        plt.xticks(np.arange(min_v, max_v + 1, 1))
    else:
        sns.histplot(vals, bins=30, color="#4c78a8", edgecolor="white")
        if max_v >= min_v:
            raw_step = max(1, int(np.ceil((max_v - min_v + 1) / max_ticks)))
            nice_steps = [1, 2, 5, 10, 20, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000]
            tick_step = next((s for s in nice_steps if s >= raw_step), nice_steps[-1])
            tick_start = (min_v // tick_step) * tick_step
            tick_end = int(np.ceil(max_v / tick_step) * tick_step)
            plt.xticks(np.arange(tick_start, tick_end + 1, tick_step))
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
    _apply_plot_style()
    plt.figure(figsize=(7, 4))
    sns.barplot(
        x=data[y_col].astype(float),
        y=data[x_col].astype(str),
        color="#54a24b",
        orient="h",
    )
    plt.title(title)
    plt.xlabel(y_col)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def plot_scalar_series(
    steps: Iterable[int],
    values: Iterable[float],
    path: Path,
    title: str,
    xlabel: str = "step",
    ylabel: str = "value",
) -> None:
    step_list = list(steps)
    value_list = list(values)
    if not step_list or not value_list:
        return
    _apply_plot_style()
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=step_list, y=value_list, color="#2c6f5f", linewidth=2, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()
