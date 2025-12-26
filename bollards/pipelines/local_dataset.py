from __future__ import annotations

import ast
import concurrent.futures as cf
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import boto3
import pandas as pd
from PIL import Image

from bollards.config import PrepareLocalDatasetConfig
from bollards.constants import LOCAL_DATASET_OUT_COLS, LOCAL_DATASET_REQUIRED_COLS
from bollards.io.fs import ensure_dir


def list_filtered_csv_keys(bucket: str, root_prefix: str, split_name: str) -> List[str]:
    """
    Finds all:
      s3://bucket/{root_prefix}/wXX/{split_name}/state/filtered.csv
    Example root_prefix: runs/osv5m_cpu
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    suffix = f"/{split_name}/state/filtered.csv"
    keys: List[str] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=root_prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith(suffix) and "/w" in k:
                keys.append(k)

    keys.sort()
    return keys


def s3_download_file(bucket: str, key: str, local_path: str) -> None:
    ensure_dir(Path(local_path).parent)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)


def parse_list_cell(cell) -> list:
    """
    Parse list-like strings from CSV reliably:
    - boxes_xyxy: [[x1,y1,x2,y2], ...]
    - boxes_conf: [0.7, ...]
    - boxes_cls:  [0.0, ...]
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            import json as _json
            return _json.loads(s)
        except Exception:
            return []


def read_image_size(path: str) -> Tuple[int, int]:
    with Image.open(path) as im:
        w, h = im.size
    return w, h


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


def build_country_map(countries: List[str]) -> Dict[str, int]:
    uniq = sorted(set([str(c) for c in countries if isinstance(c, str) and c.strip()]))
    return {c: i for i, c in enumerate(uniq)}


def stratified_split_by_label(
    df: pd.DataFrame,
    label_col: str,
    val_ratio: float,
    seed: int,
    group_col: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    if group_col:
        group_label_counts = df.groupby(group_col)[label_col].nunique()
        multi_label_groups = group_label_counts[group_label_counts > 1]
        if not multi_label_groups.empty:
            print(
                f"[warn] {len(multi_label_groups)} {group_col} values map to multiple labels; "
                "keeping groups intact may skew stratification."
            )

        val_groups = set()
        for _, g in df.groupby(label_col):
            groups = g[group_col].dropna().unique().tolist()
            if len(groups) <= 1:
                continue
            rng.shuffle(groups)
            n_val = max(1, int(round(len(groups) * val_ratio)))
            if n_val >= len(groups):
                n_val = len(groups) - 1
            val_groups.update(groups[:n_val])

        val_df = df[df[group_col].isin(val_groups)].copy()
        train_df = df[~df[group_col].isin(val_groups)].copy()
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    val_indices: List[int] = []
    for _, g in df.groupby(label_col):
        idxs = g.index.tolist()
        if len(idxs) <= 1:
            continue
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_ratio)))
        if n_val >= len(idxs):
            n_val = len(idxs) - 1
        val_indices.extend(idxs[:n_val])

    val_df = df.loc[val_indices].copy()
    train_df = df.drop(index=val_indices).copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def run_prepare_local_dataset(cfg: PrepareLocalDatasetConfig) -> None:
    random.seed(cfg.seed)

    images_dir = Path(cfg.out_dir) / "images"
    annotated_dir = Path(cfg.out_dir) / "annotated"
    meta_dir = Path(cfg.out_dir) / "meta"
    ensure_dir(images_dir)
    ensure_dir(annotated_dir)
    ensure_dir(meta_dir)

    keys = list_filtered_csv_keys(cfg.bucket, cfg.root_prefix, cfg.split)
    if not keys:
        raise SystemExit(
            f"No filtered.csv found under s3://{cfg.bucket}/{cfg.root_prefix}/w*/{cfg.split}/state/filtered.csv"
        )

    local_csv_paths = []
    for k in keys:
        parts = k.split("/")
        worker = parts[2] if len(parts) > 2 else "wxx"
        fn = f"{worker}_{cfg.split}_filtered.csv"
        lp = meta_dir / fn
        s3_download_file(cfg.bucket, k, str(lp))
        local_csv_paths.append(str(lp))

    dfs = []
    for lp in local_csv_paths:
        df = pd.read_csv(lp)
        for c in LOCAL_DATASET_REQUIRED_COLS:
            if c not in df.columns:
                raise ValueError(f"{lp} missing required column: {c}")
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    all_df = all_df[all_df["split"].astype(str) == str(cfg.split)].copy()

    all_df["n_boxes"] = pd.to_numeric(all_df["n_boxes"], errors="coerce").fillna(0).astype(int)
    all_df = all_df[all_df["n_boxes"] > 0].copy()
    if len(all_df) == 0:
        raise SystemExit("No rows with n_boxes > 0 after filtering.")

    allow_set = set(cfg.cls_allow) if cfg.cls_allow is not None else None

    candidates = []
    country_box_counts: Dict[str, int] = {}
    for _, row in all_df.iterrows():
        image_id = str(row["id"])
        country = str(row["country"])

        s3_bucket = str(row["s3_bucket"])
        s3_image_key = str(row["s3_image_key"])
        s3_annotated_key = (
            str(row["s3_annotated_key"]) if "s3_annotated_key" in row and pd.notna(row["s3_annotated_key"]) else None
        )

        boxes = parse_list_cell(row["boxes_xyxy"])
        confs = parse_list_cell(row["boxes_conf"])
        clss = parse_list_cell(row["boxes_cls"])
        if not isinstance(boxes, list) or len(boxes) == 0:
            continue

        dets = []
        for i, b in enumerate(boxes):
            if not (isinstance(b, (list, tuple)) and len(b) == 4):
                continue
            try:
                conf = float(confs[i]) if i < len(confs) else 0.0
                cls = float(clss[i]) if i < len(clss) else -1.0
                x1, y1, x2, y2 = [float(v) for v in b]
            except Exception:
                continue

            if conf < cfg.min_conf:
                continue
            if allow_set is not None and cls not in allow_set:
                continue

            bw_px = abs(x2 - x1)
            bh_px = abs(y2 - y1)
            if bw_px < cfg.min_box_w_px or bh_px < cfg.min_box_h_px:
                continue

            dets.append((i, conf, cls, x1, y1, x2, y2))

        if not dets:
            continue

        dets.sort(key=lambda t: t[1], reverse=True)
        dets = dets[: max(1, cfg.max_boxes_per_image)]

        for (box_idx, conf, cls, x1, y1, x2, y2) in dets:
            candidates.append({
                "image_id": image_id,
                "country": country,
                "s3_bucket": s3_bucket,
                "s3_image_key": s3_image_key,
                "s3_annotated_key": s3_annotated_key,
                "box_idx": int(box_idx),
                "conf": float(conf),
                "cls": float(cls),
                "x1_px": float(x1),
                "y1_px": float(y1),
                "x2_px": float(x2),
                "y2_px": float(y2),
            })
            country_box_counts[country] = country_box_counts.get(country, 0) + 1

    if cfg.min_country_count > 0:
        keep_countries = {c for c, n in country_box_counts.items() if n >= cfg.min_country_count}
        dropped = sorted(set(country_box_counts.keys()) - keep_countries)
        if dropped:
            print(f"[info] dropping {len(dropped)} countries with < {cfg.min_country_count} boxes.")
        candidates = [c for c in candidates if c["country"] in keep_countries]
        country_box_counts = {c: n for c, n in country_box_counts.items() if c in keep_countries}

    if not candidates:
        raise SystemExit("No candidate boxes after filtering. Relax filters or min_country_count.")

    country_map = build_country_map([c["country"] for c in candidates])
    with open(meta_dir / "country_map.json", "w", encoding="utf-8") as f:
        json.dump(country_map, f, indent=2, ensure_ascii=False)
    country_list = [name for name, _ in sorted(country_map.items(), key=lambda kv: kv[1])]
    with open(meta_dir / "country_list.json", "w", encoding="utf-8") as f:
        json.dump(country_list, f, indent=2, ensure_ascii=False)
    counts_df = pd.DataFrame(
        [{"country": c, "count": n} for c, n in sorted(country_box_counts.items(), key=lambda kv: kv[1], reverse=True)]
    )
    counts_df.to_csv(meta_dir / "country_counts.csv", index=False)

    cand_df = pd.DataFrame(candidates)

    if cfg.sample_strategy == "top_conf":
        cand_df = cand_df.sort_values("conf", ascending=False).reset_index(drop=True)
        selected_df = cand_df.head(cfg.num_boxes).copy()
    else:
        idxs = list(range(len(cand_df)))
        random.shuffle(idxs)
        take = idxs[: min(cfg.num_boxes, len(idxs))]
        selected_df = cand_df.iloc[take].copy().reset_index(drop=True)

    if len(selected_df) < cfg.num_boxes:
        print(f"[warn] Only {len(selected_df)} boxes available after filtering (requested {cfg.num_boxes}).")

    selected_df["country_id"] = selected_df["country"].map(country_map)

    selected_df["local_image_rel"] = selected_df["image_id"].apply(lambda x: os.path.join("images", f"{x}.jpg"))
    if cfg.download_annotated:
        selected_df["local_annot_rel"] = selected_df["image_id"].apply(lambda x: os.path.join("annotated", f"{x}.jpg"))

    img_jobs = {}
    for _, r in selected_df.iterrows():
        img_jobs[r["image_id"]] = (
            r["s3_bucket"],
            r["s3_image_key"],
            os.path.join(cfg.out_dir, "images", f"{r['image_id']}.jpg"),
        )

    def dl_one(bucket_key_path):
        b, k, lp = bucket_key_path
        s3_download_file(str(b), str(k), lp)

    print(f"[info] downloading {len(img_jobs)} original images...")
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(dl_one, img_jobs.values()))

    if cfg.download_annotated:
        ann_jobs = {}
        for _, r in selected_df.iterrows():
            if r["s3_annotated_key"] and isinstance(r["s3_annotated_key"], str):
                ann_jobs[r["image_id"]] = (
                    r["s3_bucket"],
                    r["s3_annotated_key"],
                    os.path.join(cfg.out_dir, "annotated", f"{r['image_id']}.jpg"),
                )

        print(f"[info] downloading {len(ann_jobs)} annotated images...")
        with cf.ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(dl_one, ann_jobs.values()))

    size_map: Dict[str, Tuple[int, int]] = {}
    bad_ids = set()
    for image_id in selected_df["image_id"].unique().tolist():
        lp = os.path.join(cfg.out_dir, "images", f"{image_id}.jpg")
        try:
            size_map[image_id] = read_image_size(lp)
        except Exception:
            bad_ids.add(image_id)

    if bad_ids:
        print(f"[warn] {len(bad_ids)} images unreadable; dropping their boxes.")
        selected_df = selected_df[~selected_df["image_id"].isin(bad_ids)].copy().reset_index(drop=True)

    rows = []
    for _, r in selected_df.iterrows():
        image_id = str(r["image_id"])
        w, h = size_map[image_id]
        x1n, y1n, x2n, y2n = normalize_bbox_xyxy_px(r["x1_px"], r["y1_px"], r["x2_px"], r["y2_px"], w, h)
        xc = 0.5 * (x1n + x2n)
        yc = 0.5 * (y1n + y2n)
        bw = x2n - x1n
        bh = y2n - y1n

        sample_id = f"{image_id}_{int(r['box_idx'])}"
        rows.append({
            "sample_id": sample_id,
            "image_id": image_id,
            "image_path": os.path.join("images", f"{image_id}.jpg"),
            "country": str(r["country"]),
            "country_id": int(r["country_id"]),
            "x1": x1n,
            "y1": y1n,
            "x2": x2n,
            "y2": y2n,
            "x_center": xc,
            "y_center": yc,
            "w": bw,
            "h": bh,
            "conf": float(max(0.0, min(1.0, r["conf"]))),
            "cls": float(r["cls"]),
        })

    out_df = pd.DataFrame(rows)[LOCAL_DATASET_OUT_COLS]
    merged_path = meta_dir / "merged_boxes.csv"
    out_df.to_csv(merged_path, index=False)
    print(f"[info] wrote {merged_path} rows={len(out_df)} images={out_df['image_path'].nunique()}")

    train_df, val_df = stratified_split_by_label(
        out_df,
        "country_id",
        cfg.val_ratio,
        cfg.seed,
        group_col="image_id",
    )
    train_path = meta_dir / "train.csv"
    val_path = meta_dir / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print(f"[info] wrote {train_path} rows={len(train_df)}")
    print(f"[info] wrote {val_path} rows={len(val_df)}")
