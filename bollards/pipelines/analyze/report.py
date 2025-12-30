from __future__ import annotations

from html import escape as escape_html

import pandas as pd

REPORT_CSS = """
    <style>
    :root {
      --bg: #f4f0e6;
      --bg-accent: #e7f0eb;
      --ink: #1f2b24;
      --muted: #5c6b5f;
      --card: #ffffff;
      --accent: #2c6f5f;
      --accent-2: #c89b3c;
      --border: #e2ddd2;
      --shadow: 0 14px 28px rgba(20, 26, 21, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Source Sans 3", "Noto Sans", sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at top left, #faf6ef 0%, var(--bg) 45%, var(--bg-accent) 100%);
    }
    .page { max-width: 1200px; margin: 0 auto; padding: 28px; }
    .page-header {
      padding: 24px 26px;
      border-radius: 18px;
      border: 1px solid var(--border);
      background: linear-gradient(135deg, #fef9f0 0%, #f0f7f4 100%);
      box-shadow: var(--shadow);
      margin-bottom: 26px;
    }
    .eyebrow { text-transform: uppercase; letter-spacing: 0.14em; font-size: 11px; color: var(--muted); margin: 0 0 8px; }
    h1 {
      font-family: "Fraunces", "Iowan Old Style", "Georgia", serif;
      font-size: 34px;
      margin: 0 0 8px;
      letter-spacing: -0.02em;
      color: var(--ink);
    }
    h2 {
      font-size: 22px;
      margin: 0 0 14px;
      color: var(--accent);
    }
    h3 { font-size: 18px; margin: 20px 0 10px; }
    h4 {
      margin: 4px 0 8px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }
    p { margin: 8px 0; color: var(--ink); }
    .meta { color: var(--muted); margin: 0; }
    .badge {
      display: inline-block;
      padding: 2px 10px;
      border-radius: 999px;
      background: rgba(44, 111, 95, 0.12);
      color: var(--accent);
      font-weight: 600;
      font-size: 12px;
    }
    section.section-card {
      background: var(--card);
      border-radius: 16px;
      padding: 20px 22px;
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }
    .table-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px;
      margin-bottom: 10px;
    }
    .table-card {
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      box-shadow: 0 8px 18px rgba(15, 20, 17, 0.06);
    }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 8px 10px; border-bottom: 1px solid var(--border); }
    th {
      text-align: left;
      background: #f7f4ea;
      font-weight: 600;
      color: #2d3b32;
    }
    tr:nth-child(even) { background: #fbfaf7; }
    details.expand {
      margin: 10px 0 16px;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px dashed var(--border);
      background: #fbf7ee;
    }
    details.expand summary {
      cursor: pointer;
      font-weight: 600;
      color: var(--accent);
    }
    details.expand pre {
      white-space: pre-wrap;
      margin: 8px 0 0;
      font-size: 12px;
      color: var(--muted);
    }
    img {
      margin: 8px 8px 8px 0;
      border-radius: 10px;
      border: 1px solid var(--border);
      box-shadow: 0 10px 20px rgba(18, 24, 20, 0.08);
      max-width: 100%;
      height: auto;
      background: #fff;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 10px;
    }
    .grid img { width: 100%; height: auto; display: block; margin: 0; }
    @media (max-width: 720px) {
      .page { padding: 18px; }
      .page-header { padding: 18px; }
      h1 { font-size: 28px; }
    }
    </style>
    """


def format_gallery_labels(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    cols = [c for c in ["image_path", "image_id"] if c in df.columns]
    if not cols:
        return []
    data = df[cols].dropna(how="all").copy()
    if data.empty:
        return []
    data = data.drop_duplicates()
    labels: list[str] = []
    if "image_path" in data.columns and "image_id" in data.columns:
        for _, row in data.iterrows():
            path = str(row.get("image_path", "")).strip()
            img_id = str(row.get("image_id", "")).strip()
            if path and img_id:
                labels.append(f"{path} ({img_id})")
            elif path:
                labels.append(path)
            elif img_id:
                labels.append(img_id)
    elif "image_path" in data.columns:
        labels = [str(p).strip() for p in data["image_path"].tolist() if str(p).strip()]
    else:
        labels = [str(p).strip() for p in data["image_id"].tolist() if str(p).strip()]
    return labels


def render_expand_list(title: str, items: list[str]) -> str:
    if not items:
        return ""
    escaped = escape_html("\n".join(items))
    return (
        "<details class='expand'>"
        f"<summary>{escape_html(title)}</summary>"
        f"<pre>{escaped}</pre>"
        "</details>"
    )


def build_report_html(run_name: str, region_note: str, report_sections: list[str]) -> str:
    header = (
        "<header class='page-header'>"
        "<p class='eyebrow'>Bollards report</p>"
        "<h1>Single-run analysis</h1>"
        f"<p class='meta'>Run <span class='badge'>{escape_html(run_name)}</span></p>"
        f"<p class='meta'>{escape_html(region_note)}</p>"
        "</header>"
    )
    body = "<div class='page'>" + header + "\n".join(report_sections) + "</div>"
    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"{REPORT_CSS}</head><body>{body}</body></html>"
    )
    return html
