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
    return df.to_html(index=False, escape=True)


def build_report_section(title: str, body: str) -> str:
    return f"<section><h2>{title}</h2>{body}</section>"


def relative_paths(paths: list[str], base_dir: Path) -> list[str]:
    rel = []
    for p in paths:
        try:
            rel.append(str(Path(p).relative_to(base_dir)))
        except Exception:
            rel.append(p)
    return rel
