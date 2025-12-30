from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def render_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No data.</p>"
    return "<div class='table-wrap'>" + df.to_html(index=False, escape=True) + "</div>"


def build_report_section(title: str, body: str) -> str:
    return f"<section class='section-card'><h2>{title}</h2>{body}</section>"


def render_table_grid(tables: list[tuple[str, pd.DataFrame]]) -> str:
    if not tables:
        return ""
    cards = []
    for title, df in tables:
        heading = f"<h4>{title}</h4>" if title else ""
        cards.append(f"<div class='table-card'>{heading}{render_table(df)}</div>")
    return "<div class='table-grid'>" + "".join(cards) + "</div>"


def relative_paths(paths: list[str], base_dir: Path) -> list[str]:
    rel = []
    base_dir = Path(base_dir)
    for p in paths:
        path = Path(p)
        if path.is_relative_to(base_dir):
            rel.append(str(path.relative_to(base_dir)))
        else:
            rel.append(str(p))
    return rel
