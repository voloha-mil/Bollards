from __future__ import annotations

import ast
import concurrent.futures as cf
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm

from bollards.config import PrepareLocalDatasetConfig
from bollards.constants import LOCAL_DATASET_OUT_COLS, LOCAL_DATASET_REQUIRED_COLS
from bollards.data.bboxes import normalize_bbox_xyxy_px
from bollards.io.fs import ensure_dir
from bollards.io.s3 import s3_download_file as s3_download_file_raw, s3_list_keys
from bollards.utils.seeding import make_python_rng


def list_filtered_csv_keys(bucket: str, root_prefix: str, split_name: str) -> List[str]:
    """
    Finds all:
      s3://bucket/{root_prefix}/wXX/{split_name}/state/filtered.csv
    Example root_prefix: runs/osv5m_cpu
    """
    suffix = f"/{split_name}/state/filtered.csv"
    prefix = root_prefix.rstrip("/") + "/"
    keys = [
        k for k in s3_list_keys(bucket=bucket, prefix=prefix, suffixes=[suffix])
        if k.endswith(suffix) and "/w" in k
    ]
    return sorted(keys)


def download_s3_file(bucket: str, key: str, local_path: str) -> None:
    ensure_dir(Path(local_path).parent)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return
    s3_download_file_raw(bucket=bucket, key=key, local_path=Path(local_path))


def load_filtered_split_df(
    bucket: str,
    root_prefix: str,
    split_name: str,
    meta_dir: Path,
) -> pd.DataFrame:
    keys = list_filtered_csv_keys(bucket, root_prefix, split_name)
    if not keys:
        raise SystemExit(
            f"No filtered.csv found under s3://{bucket}/{root_prefix}/w*/{split_name}/state/filtered.csv"
        )

    local_csv_paths = []
    for k in keys:
        parts = k.split("/")
        worker = parts[2] if len(parts) > 2 else "wxx"
        fn = f"{worker}_{split_name}_filtered.csv"
        lp = meta_dir / fn
        download_s3_file(bucket, k, str(lp))
        local_csv_paths.append(str(lp))

    dfs = []
    for lp in local_csv_paths:
        df = pd.read_csv(lp)
        for c in LOCAL_DATASET_REQUIRED_COLS:
            if c not in df.columns:
                raise ValueError(f"{lp} missing required column: {c}")
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    all_df = all_df[all_df["split"].astype(str) == str(split_name)].copy()
    all_df["n_boxes"] = pd.to_numeric(all_df["n_boxes"], errors="coerce").fillna(0).astype(int)
    all_df = all_df[all_df["n_boxes"] > 0].copy()
    return all_df


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


def collect_candidates(
    df: pd.DataFrame,
    cfg: PrepareLocalDatasetConfig,
    allow_set: set[float] | None,
    allow_countries: set[str] | None = None,
) -> Tuple[List[dict], Dict[str, int]]:
    candidates = []
    country_box_counts: Dict[str, int] = {}
    for _, row in df.iterrows():
        image_id = str(row["id"])
        country = str(row["country"])
        if allow_countries is not None and country not in allow_countries:
            continue

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

    return candidates, country_box_counts


def select_candidates(
    candidates: List[dict],
    cfg: PrepareLocalDatasetConfig,
    split_label: str,
    rng: random.Random,
) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame()
    cand_df = pd.DataFrame(candidates)

    if cfg.sample_strategy == "top_conf":
        cand_df = cand_df.sort_values("conf", ascending=False).reset_index(drop=True)
        selected_df = cand_df.head(cfg.num_boxes).copy()
    else:
        idxs = list(range(len(cand_df)))
        rng.shuffle(idxs)
        take = idxs[: min(cfg.num_boxes, len(idxs))]
        selected_df = cand_df.iloc[take].copy().reset_index(drop=True)

    if len(selected_df) < cfg.num_boxes:
        print(
            f"[warn] Only {len(selected_df)} boxes available after filtering for {split_label} "
            f"(requested {cfg.num_boxes})."
        )
    return selected_df


def build_output_df(selected_df: pd.DataFrame, size_map: Dict[str, Tuple[int, int]]) -> pd.DataFrame:
    rows = []
    for _, r in selected_df.iterrows():
        image_id = str(r["image_id"])
        if image_id not in size_map:
            continue
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

    return pd.DataFrame(rows)[LOCAL_DATASET_OUT_COLS]


def run_prepare_local_dataset(cfg: PrepareLocalDatasetConfig) -> None:
    train_rng = make_python_rng(cfg.seed, "local_dataset_train")
    val_rng = make_python_rng(cfg.seed, "local_dataset_val")

    images_dir = Path(cfg.out_dir) / "images"
    annotated_dir = Path(cfg.out_dir) / "annotated"
    meta_dir = Path(cfg.out_dir) / "meta"
    ensure_dir(images_dir)
    ensure_dir(annotated_dir)
    ensure_dir(meta_dir)
    train_df = load_filtered_split_df(cfg.bucket, cfg.root_prefix, cfg.train_split, meta_dir)
    if len(train_df) == 0:
        raise SystemExit(f"No rows with n_boxes > 0 for train split '{cfg.train_split}'.")

    val_df = load_filtered_split_df(cfg.bucket, cfg.root_prefix, cfg.val_split, meta_dir)
    if len(val_df) == 0:
        raise SystemExit(f"No rows with n_boxes > 0 for val split '{cfg.val_split}'.")

    allow_set = set(cfg.cls_allow) if cfg.cls_allow is not None else None

    train_candidates, country_box_counts = collect_candidates(train_df, cfg, allow_set)
    if not train_candidates:
        raise SystemExit("No train candidate boxes after filtering. Relax filters or min_country_count.")

    keep_countries = set(country_box_counts.keys())
    if cfg.min_country_count > 0:
        keep_countries = {c for c, n in country_box_counts.items() if n >= cfg.min_country_count}
        dropped = sorted(set(country_box_counts.keys()) - keep_countries)
        if dropped:
            print(f"[info] dropping {len(dropped)} countries with < {cfg.min_country_count} boxes.")
        train_candidates = [c for c in train_candidates if c["country"] in keep_countries]
        country_box_counts = {c: n for c, n in country_box_counts.items() if c in keep_countries}

    if not train_candidates:
        raise SystemExit("No train candidate boxes after filtering. Relax filters or min_country_count.")

    if not keep_countries:
        raise SystemExit("No countries remain after min_country_count filtering.")

    val_candidates, _ = collect_candidates(val_df, cfg, allow_set, allow_countries=keep_countries)
    if not val_candidates:
        raise SystemExit("No validation candidate boxes after filtering. Check filters or min_country_count.")

    country_map = build_country_map([c["country"] for c in train_candidates])
    with open(meta_dir / "country_map.json", "w", encoding="utf-8") as f:
        json.dump(country_map, f, indent=2, ensure_ascii=False)
    country_list = [name for name, _ in sorted(country_map.items(), key=lambda kv: kv[1])]
    with open(meta_dir / "country_list.json", "w", encoding="utf-8") as f:
        json.dump(country_list, f, indent=2, ensure_ascii=False)
    counts_df = pd.DataFrame(
        [{"country": c, "count": n} for c, n in sorted(country_box_counts.items(), key=lambda kv: kv[1], reverse=True)]
    )
    counts_df.to_csv(meta_dir / "country_counts.csv", index=False)

    train_selected_df = select_candidates(train_candidates, cfg, f"train ({cfg.train_split})", rng=train_rng)
    if train_selected_df.empty:
        raise SystemExit("No train boxes after sampling.")
    val_selected_df = select_candidates(val_candidates, cfg, f"val ({cfg.val_split})", rng=val_rng)
    if val_selected_df.empty:
        raise SystemExit("No validation boxes after sampling.")

    train_selected_df["country_id"] = train_selected_df["country"].map(country_map)
    val_selected_df["country_id"] = val_selected_df["country"].map(country_map)
    if train_selected_df["country_id"].isna().any() or val_selected_df["country_id"].isna().any():
        raise ValueError("Found countries in data that are missing from train country_map.")

    train_selected_df["local_image_rel"] = train_selected_df["image_id"].apply(
        lambda x: os.path.join("images", f"{x}.jpg")
    )
    val_selected_df["local_image_rel"] = val_selected_df["image_id"].apply(
        lambda x: os.path.join("images", f"{x}.jpg")
    )
    if cfg.download_annotated:
        train_selected_df["local_annot_rel"] = train_selected_df["image_id"].apply(
            lambda x: os.path.join("annotated", f"{x}.jpg")
        )
        val_selected_df["local_annot_rel"] = val_selected_df["image_id"].apply(
            lambda x: os.path.join("annotated", f"{x}.jpg")
        )

    selected_df = pd.concat([train_selected_df, val_selected_df], ignore_index=True)

    img_jobs = {}
    for _, r in selected_df.iterrows():
        img_jobs[r["image_id"]] = (
            r["s3_bucket"],
            r["s3_image_key"],
            os.path.join(cfg.out_dir, "images", f"{r['image_id']}.jpg"),
        )

    def dl_one(bucket_key_path):
        b, k, lp = bucket_key_path
        download_s3_file(str(b), str(k), lp)

    print(f"[info] downloading {len(img_jobs)} original images...")
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(dl_one, job) for job in img_jobs.values()]
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="download images", dynamic_ncols=True):
            fut.result()

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
        train_selected_df = train_selected_df[~train_selected_df["image_id"].isin(bad_ids)].copy().reset_index(drop=True)
        val_selected_df = val_selected_df[~val_selected_df["image_id"].isin(bad_ids)].copy().reset_index(drop=True)

    if train_selected_df.empty:
        raise SystemExit("No train boxes remain after dropping unreadable images.")
    if val_selected_df.empty:
        raise SystemExit("No validation boxes remain after dropping unreadable images.")

    train_out_df = build_output_df(train_selected_df, size_map)
    val_out_df = build_output_df(val_selected_df, size_map)

    out_df = pd.concat([train_out_df, val_out_df], ignore_index=True)
    merged_path = meta_dir / "merged_boxes.csv"
    out_df.to_csv(merged_path, index=False)
    print(f"[info] wrote {merged_path} rows={len(out_df)} images={out_df['image_path'].nunique()}")

    train_path = meta_dir / "train.csv"
    val_path = meta_dir / "val.csv"
    train_out_df.to_csv(train_path, index=False)
    val_out_df.to_csv(val_path, index=False)
    print(f"[info] wrote {train_path} rows={len(train_out_df)}")
    print(f"[info] wrote {val_path} rows={len(val_out_df)}")
