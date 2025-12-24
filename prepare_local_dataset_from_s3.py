#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import json
import os
import random
from typing import Dict, List, Tuple, Optional

import boto3
import pandas as pd
from PIL import Image


# Fixed schema (no guessing)
CSV_COLS_REQUIRED = [
    "id", "split", "country", "n_boxes",
    "boxes_xyxy", "boxes_conf", "boxes_cls",
    "s3_bucket", "s3_image_key",
]

# Optional but present in your schema (for visualization download)
OPTIONAL_COLS = ["s3_annotated_key"]

# Output schema consumed by train.py (crop on the fly using bbox columns)
OUT_COLS = [
    "sample_id",
    "image_path",      # local relative path to original image (under out_dir)
    "country_id",      # int label
    "country",         # string label (ISO2 in your data)
    "x1", "y1", "x2", "y2",               # bbox normalized [0,1] in original image
    "x_center", "y_center", "w", "h",     # derived normalized geometry
    "conf", "cls",                        # detector score and class id
    "image_id",                           # original id
]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


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
    ensure_dir(os.path.dirname(local_path))
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


def normalize_bbox_xyxy_px(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[float, float, float, float]:
    # convert px -> normalized and clamp
    x1n = min(max(x1 / W, 0.0), 1.0)
    x2n = min(max(x2 / W, 0.0), 1.0)
    y1n = min(max(y1 / H, 0.0), 1.0)
    y2n = min(max(y2 / H, 0.0), 1.0)
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


def stratified_split_by_label(df: pd.DataFrame, label_col: str, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--bucket", default="geo-bollard-ml")
    ap.add_argument("--root_prefix", default="runs/osv5m_cpu")
    ap.add_argument("--split", default="test", help="split name in your pipeline (e.g. test)")
    ap.add_argument("--out_dir", default="./local_data")

    # NEW: target number of boxes (final dataset size controller)
    ap.add_argument("--num_boxes", type=int, required=True, help="final number of bbox samples to extract after filtering")

    ap.add_argument("--max_boxes_per_image", type=int, default=1, help="keep top-K boxes by conf per image after filtering")
    ap.add_argument("--min_conf", type=float, default=0.0)
    ap.add_argument("--cls_allow", type=float, nargs="*", default=None, help="allowed cls ids (e.g. 0.0). If omitted, allow all.")
    ap.add_argument("--min_box_w_px", type=float, default=0.0, help="drop boxes with width < this (pixels)")
    ap.add_argument("--min_box_h_px", type=float, default=0.0, help="drop boxes with height < this (pixels)")

    ap.add_argument("--sample_strategy", choices=["random", "top_conf"], default="random",
                    help="how to select num_boxes from the filtered pool")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--download_annotated", action="store_true")

    args = ap.parse_args()
    random.seed(args.seed)

    images_dir = os.path.join(args.out_dir, "images")
    annotated_dir = os.path.join(args.out_dir, "annotated")
    meta_dir = os.path.join(args.out_dir, "meta")
    ensure_dir(images_dir)
    ensure_dir(annotated_dir)
    ensure_dir(meta_dir)

    # 1) discover + download all filtered.csv
    keys = list_filtered_csv_keys(args.bucket, args.root_prefix, args.split)
    if not keys:
        raise SystemExit(
            f"No filtered.csv found under s3://{args.bucket}/{args.root_prefix}/w*/{args.split}/state/filtered.csv"
        )

    local_csv_paths = []
    for k in keys:
        parts = k.split("/")
        worker = parts[2] if len(parts) > 2 else "wxx"
        fn = f"{worker}_{args.split}_filtered.csv"
        lp = os.path.join(meta_dir, fn)
        s3_download_file(args.bucket, k, lp)
        local_csv_paths.append(lp)

    # 2) merge + basic row filtering
    dfs = []
    for lp in local_csv_paths:
        df = pd.read_csv(lp)
        for c in CSV_COLS_REQUIRED:
            if c not in df.columns:
                raise ValueError(f"{lp} missing required column: {c}")
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    # Keep only rows for requested split name (extra safety)
    all_df = all_df[all_df["split"].astype(str) == str(args.split)].copy()

    all_df["n_boxes"] = pd.to_numeric(all_df["n_boxes"], errors="coerce").fillna(0).astype(int)
    all_df = all_df[all_df["n_boxes"] > 0].copy()
    if len(all_df) == 0:
        raise SystemExit("No rows with n_boxes > 0 after filtering.")

    # 3) country map
    country_map = build_country_map(all_df["country"].astype(str).tolist())
    with open(os.path.join(meta_dir, "country_map.json"), "w", encoding="utf-8") as f:
        json.dump(country_map, f, indent=2, ensure_ascii=False)

    allow_set = set(args.cls_allow) if args.cls_allow is not None else None

    # 4) Expand all rows into candidate box samples (NO image downloads yet)
    # Apply filters that do not require image sizes.
    candidates = []
    for _, row in all_df.iterrows():
        image_id = str(row["id"])
        country = str(row["country"])
        if country not in country_map:
            continue
        country_id = int(country_map[country])

        s3_bucket = str(row["s3_bucket"])
        s3_image_key = str(row["s3_image_key"])
        s3_annotated_key = str(row["s3_annotated_key"]) if "s3_annotated_key" in row and pd.notna(row["s3_annotated_key"]) else None

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

            if conf < args.min_conf:
                continue
            if allow_set is not None and cls not in allow_set:
                continue

            bw_px = abs(x2 - x1)
            bh_px = abs(y2 - y1)
            if bw_px < args.min_box_w_px or bh_px < args.min_box_h_px:
                continue

            dets.append((i, conf, cls, x1, y1, x2, y2))

        if not dets:
            continue

        # sort by confidence and keep top-K per image
        dets.sort(key=lambda t: t[1], reverse=True)
        dets = dets[: max(1, args.max_boxes_per_image)]

        for (box_idx, conf, cls, x1, y1, x2, y2) in dets:
            candidates.append({
                "image_id": image_id,
                "country": country,
                "country_id": country_id,
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

    if not candidates:
        raise SystemExit("No candidate boxes after filtering. Relax filters.")

    cand_df = pd.DataFrame(candidates)

    # 5) Select exactly num_boxes from filtered pool
    if args.sample_strategy == "top_conf":
        cand_df = cand_df.sort_values("conf", ascending=False).reset_index(drop=True)
        selected_df = cand_df.head(args.num_boxes).copy()
    else:
        # random
        idxs = list(range(len(cand_df)))
        random.shuffle(idxs)
        take = idxs[: min(args.num_boxes, len(idxs))]
        selected_df = cand_df.iloc[take].copy().reset_index(drop=True)

    if len(selected_df) < args.num_boxes:
        print(f"[warn] Only {len(selected_df)} boxes available after filtering (requested {args.num_boxes}).")

    # 6) Download only required original images (+ annotated if requested)
    # Use local filenames based on image_id to avoid collisions.
    # (assumes each id maps to one image key; holds for your dataset)
    selected_df["local_image_rel"] = selected_df["image_id"].apply(lambda x: os.path.join("images", f"{x}.jpg"))
    if args.download_annotated:
        selected_df["local_annot_rel"] = selected_df["image_id"].apply(lambda x: os.path.join("annotated", f"{x}.jpg"))

    # Build unique download list
    img_jobs = {}
    for _, r in selected_df.iterrows():
        img_jobs[r["image_id"]] = (r["s3_bucket"], r["s3_image_key"], os.path.join(args.out_dir, "images", f"{r['image_id']}.jpg"))

    def dl_one(bucket_key_path):
        b, k, lp = bucket_key_path
        s3_download_file(str(b), str(k), lp)

    print(f"[info] downloading {len(img_jobs)} original images...")
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(dl_one, img_jobs.values()))

    if args.download_annotated:
        ann_jobs = {}
        for _, r in selected_df.iterrows():
            if r["s3_annotated_key"] and isinstance(r["s3_annotated_key"], str):
                ann_jobs[r["image_id"]] = (r["s3_bucket"], r["s3_annotated_key"], os.path.join(args.out_dir, "annotated", f"{r['image_id']}.jpg"))

        print(f"[info] downloading {len(ann_jobs)} annotated images...")
        with cf.ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(dl_one, ann_jobs.values()))

    # 7) Read image sizes (drop boxes whose images are unreadable)
    size_map: Dict[str, Tuple[int, int]] = {}
    bad_ids = set()
    for image_id in selected_df["image_id"].unique().tolist():
        lp = os.path.join(args.out_dir, "images", f"{image_id}.jpg")
        try:
            size_map[image_id] = read_image_size(lp)
        except Exception:
            bad_ids.add(image_id)

    if bad_ids:
        print(f"[warn] {len(bad_ids)} images unreadable; dropping their boxes.")
        selected_df = selected_df[~selected_df["image_id"].isin(bad_ids)].copy().reset_index(drop=True)

    # 8) Build final training table (normalized bbox + meta)
    rows = []
    for _, r in selected_df.iterrows():
        image_id = str(r["image_id"])
        W, H = size_map[image_id]
        x1n, y1n, x2n, y2n = normalize_bbox_xyxy_px(r["x1_px"], r["y1_px"], r["x2_px"], r["y2_px"], W, H)
        xc = 0.5 * (x1n + x2n)
        yc = 0.5 * (y1n + y2n)
        bw = (x2n - x1n)
        bh = (y2n - y1n)

        sample_id = f"{image_id}_{int(r['box_idx'])}"
        rows.append({
            "sample_id": sample_id,
            "image_id": image_id,
            "image_path": os.path.join("images", f"{image_id}.jpg"),
            "country": str(r["country"]),
            "country_id": int(r["country_id"]),
            "x1": x1n, "y1": y1n, "x2": x2n, "y2": y2n,
            "x_center": xc, "y_center": yc, "w": bw, "h": bh,
            "conf": float(max(0.0, min(1.0, r["conf"]))),
            "cls": float(r["cls"]),
        })

    out_df = pd.DataFrame(rows)[OUT_COLS]
    merged_path = os.path.join(meta_dir, "merged_boxes.csv")
    out_df.to_csv(merged_path, index=False)
    print(f"[info] wrote {merged_path} rows={len(out_df)} images={out_df['image_path'].nunique()}")

    # 9) train/val split
    train_df, val_df = stratified_split_by_label(out_df, "country_id", args.val_ratio, args.seed)
    train_path = os.path.join(meta_dir, "train.csv")
    val_path = os.path.join(meta_dir, "val.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print(f"[info] wrote {train_path} rows={len(train_df)}")
    print(f"[info] wrote {val_path} rows={len(val_df)}")


if __name__ == "__main__":
    main()
