from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from bollards.config import AnalyzeRunConfig
from bollards.constants import BBOX_COLS, LABEL_COL, META_COLS, PATH_COL
from bollards.data.country_names import golden_country_to_code
from bollards.data.bboxes import compute_avg_bbox_wh
from bollards.data.datasets import BollardCropsDataset
from bollards.data.labels import load_id_to_country
from bollards.data.transforms import build_transforms
from bollards.detect.yolo import load_yolo, run_inference_batch
from bollards.io.fs import ensure_dir
from bollards.io.hf import hf_download_model_file
from bollards.models.bollard_net import BollardNet, ModelConfig
from bollards.pipelines.local_dataset import normalize_bbox_xyxy_px


class BollardCropsDatasetWithIndex(BollardCropsDataset):
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        item["index"] = torch.tensor(idx, dtype=torch.long)
        return item


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("analyze_run")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def resolve_device(device_str: str) -> torch.device:
    if device_str and device_str != "auto":
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_detector(cfg: AnalyzeRunConfig, logger: logging.Logger):
    weights_path = cfg.detector.weights_path
    if weights_path:
        weights = Path(weights_path)
    else:
        hf_cache = Path(cfg.detector.hf_cache)
        weights = hf_download_model_file(
            repo_id=cfg.detector.hf_repo,
            filename=cfg.detector.hf_filename,
            cache_dir=hf_cache,
        )
    logger.info("Using detector weights: %s", weights)
    return load_yolo(weights)


def _load_classifier(cfg: AnalyzeRunConfig, device: torch.device, logger: logging.Logger) -> tuple[BollardNet, ModelConfig]:
    ckpt_path = cfg.classifier.checkpoint_path
    if not ckpt_path:
        raise SystemExit("classifier.checkpoint_path is required")

    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg_dict = ckpt.get("cfg")
    if not model_cfg_dict:
        raise SystemExit("Classifier checkpoint missing cfg metadata")

    model_cfg = ModelConfig(**model_cfg_dict)
    model_cfg.pretrained = False
    if model_cfg.meta_dim != len(META_COLS):
        raise SystemExit(f"model.meta_dim must match META_COLS ({len(META_COLS)})")

    model = BollardNet(model_cfg).to(device)
    state = ckpt.get("model") or ckpt.get("state_dict")
    if not state:
        raise SystemExit("Classifier checkpoint missing model weights")
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded classifier: %s", ckpt_path)
    return model, model_cfg


def _load_font(font_size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for p in candidates:
        try:
            if os.path.exists(p):
                return ImageFont.truetype(p, size=font_size)
        except Exception:
            pass
    return ImageFont.load_default()


def _class_name(cls_id: int, names: Optional[list[str]]) -> str:
    if names and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return f"class_{cls_id}"


def _build_country_mappings(
    id_to_country: Optional[list[str]],
    main_df: pd.DataFrame,
) -> tuple[Optional[list[str]], Optional[Dict[str, int]]]:
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


def _prepare_golden_df_for_classifier(
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


def _maybe_add_region_by_country(
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


def _build_region_map(cfg: AnalyzeRunConfig, golden_df: Optional[pd.DataFrame]) -> Optional[Dict[str, str]]:
    if cfg.data.region_map_json:
        with open(cfg.data.region_map_json, "r", encoding="utf-8") as f:
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


def _ensure_image_id(df: pd.DataFrame) -> pd.Series:
    if "image_id" in df.columns:
        return df["image_id"].astype(str)
    if "orig_sha1" in df.columns:
        return df["orig_sha1"].astype(str)
    if PATH_COL in df.columns:
        return df[PATH_COL].apply(lambda p: Path(str(p)).stem)
    return pd.Series(["unknown"] * len(df))


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _plot_hist(values: Iterable[float], path: Path, title: str, xlabel: str, bins: int = 30) -> None:
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


def _plot_bar(df: pd.DataFrame, path: Path, title: str, x_col: str, y_col: str, max_items: int = 20) -> None:
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


def _crop_with_bbox(img: Image.Image, x1: float, y1: float, x2: float, y2: float, expand: float) -> Image.Image:
    w, h = img.size
    x1n, y1n, x2n, y2n = x1, y1, x2, y2
    x1n, x2n = sorted([max(0.0, min(1.0, x1n)), max(0.0, min(1.0, x2n))])
    y1n, y2n = sorted([max(0.0, min(1.0, y1n)), max(0.0, min(1.0, y2n))])

    xc = 0.5 * (x1n + x2n)
    yc = 0.5 * (y1n + y2n)
    bw = (x2n - x1n) * expand
    bh = (y2n - y1n) * expand

    ex1 = max(0.0, xc - 0.5 * bw)
    ey1 = max(0.0, yc - 0.5 * bh)
    ex2 = min(1.0, xc + 0.5 * bw)
    ey2 = min(1.0, yc + 0.5 * bh)

    px1 = int(round(ex1 * w))
    py1 = int(round(ey1 * h))
    px2 = int(round(ex2 * w))
    py2 = int(round(ey2 * h))
    if px2 <= px1:
        px2 = min(w, px1 + 1)
    if py2 <= py1:
        py2 = min(h, py1 + 1)

    return img.crop((px1, py1, px2, py2))


def _draw_boxes(img: Image.Image, boxes: list[dict[str, Any]], color_map: dict[int, tuple[int, int, int]]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for det in boxes:
        x1 = int(round(det["x1"] * w))
        y1 = int(round(det["y1"] * h))
        x2 = int(round(det["x2"] * w))
        y2 = int(round(det["y2"] * h))
        cls_id = int(det["cls"]) if "cls" in det else -1
        color = color_map.get(cls_id, (255, 0, 0))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
    return img


def _make_color_map(ids: Iterable[int]) -> dict[int, tuple[int, int, int]]:
    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 212),
    ]
    ids = list(sorted(set(ids)))
    return {cid: palette[i % len(palette)] for i, cid in enumerate(ids)}


def _dataset_summary(df: pd.DataFrame, country_col: str, region_col: Optional[str]) -> dict[str, Any]:
    image_id = _ensure_image_id(df)
    n_images = image_id.nunique()
    n_objects = len(df)
    objects_per_image = float(n_objects) / max(n_images, 1)
    out = {
        "n_images": int(n_images),
        "n_objects": int(n_objects),
        "objects_per_image": objects_per_image,
        "n_countries": int(df[country_col].nunique()) if country_col in df.columns else 0,
    }
    if region_col and region_col in df.columns:
        out["n_regions"] = int(df[region_col].nunique())
    return out


def _value_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns or df.empty:
        return pd.DataFrame({col: [], "count": []})
    counts = df[col].value_counts(dropna=True)
    if counts.empty:
        return pd.DataFrame({col: [], "count": []})
    return counts.reset_index().rename(columns={"index": col, col: "count"})


def _top_bottom(df: pd.DataFrame, col: str, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = _value_counts(df, col)
    if counts.empty:
        return counts, counts
    top = counts.head(n).copy()
    bottom = counts.tail(n).copy()
    return top, bottom


def _calc_bbox_area_aspect(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if not all(c in df.columns for c in BBOX_COLS):
        return pd.DataFrame()
    w = (df["x2"] - df["x1"]).clip(lower=0.0)
    h = (df["y2"] - df["y1"]).clip(lower=0.0)
    area = (w * h).clip(lower=0.0)
    aspect = (w / h.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame({
        f"{prefix}_area": area,
        f"{prefix}_aspect": aspect,
    })
    return out


def _calc_crop_area_aspect(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if not all(c in df.columns for c in ["crop_w", "crop_h", "orig_w", "orig_h"]):
        return pd.DataFrame()
    area = (df["crop_w"] * df["crop_h"]) / (df["orig_w"] * df["orig_h"]).replace(0, np.nan)
    aspect = df["crop_w"] / df["crop_h"].replace(0, np.nan)
    out = pd.DataFrame({
        f"{prefix}_area": area,
        f"{prefix}_aspect": aspect,
    })
    return out


def _filter_detections(
    cfg: AnalyzeRunConfig,
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    clss: np.ndarray,
) -> list[dict[str, float]]:
    allow_set = set(cfg.detector.cls_allow) if cfg.detector.cls_allow is not None else None
    filtered = []
    for i in range(len(confs)):
        conf = float(confs[i])
        cls = float(clss[i])
        x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[i]]

        if conf < cfg.detector.conf:
            continue
        if allow_set is not None and cls not in allow_set:
            continue
        if abs(x2 - x1) < cfg.detector.min_box_w_px:
            continue
        if abs(y2 - y1) < cfg.detector.min_box_h_px:
            continue

        filtered.append({
            "conf": conf,
            "cls": cls,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })

    filtered.sort(key=lambda d: d["conf"], reverse=True)
    max_keep = max(1, int(cfg.detector.max_boxes_per_image))
    return filtered[:max_keep]


def _run_detector(
    cfg: AnalyzeRunConfig,
    logger: logging.Logger,
    main_df: pd.DataFrame,
    img_root: Path,
    run_dir: Path,
) -> pd.DataFrame:
    detector = _load_detector(cfg, logger)
    device = resolve_device(cfg.device)

    image_paths = main_df[PATH_COL].dropna().unique().tolist()
    if cfg.detector.max_images > 0:
        image_paths = image_paths[: cfg.detector.max_images]

    records = []
    det_counts = []

    batches = [image_paths[i : i + cfg.detector.batch] for i in range(0, len(image_paths), cfg.detector.batch)]
    for batch_paths in tqdm(batches, desc="detect", leave=False, dynamic_ncols=True):
        abs_paths = [img_root / p for p in batch_paths]
        results = run_inference_batch(
            model=detector,
            image_paths=abs_paths,
            imgsz=cfg.detector.imgsz,
            conf=cfg.detector.conf,
            device=str(device),
            batch=cfg.detector.batch,
        )
        for rel_path, result in zip(batch_paths, results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                det_counts.append(0)
                continue
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            filtered = _filter_detections(cfg, boxes_xyxy, confs, clss)
            det_counts.append(len(filtered))
            if not filtered:
                continue

            h, w = result.orig_shape
            for det in filtered:
                x1n, y1n, x2n, y2n = normalize_bbox_xyxy_px(det["x1"], det["y1"], det["x2"], det["y2"], w, h)
                xc = 0.5 * (x1n + x2n)
                yc = 0.5 * (y1n + y2n)
                bw = x2n - x1n
                bh = y2n - y1n
                records.append({
                    "image_path": rel_path,
                    "x1": x1n,
                    "y1": y1n,
                    "x2": x2n,
                    "y2": y2n,
                    "x_center": xc,
                    "y_center": yc,
                    "w": bw,
                    "h": bh,
                    "conf": float(det["conf"]),
                    "cls": float(det["cls"]),
                })

    det_df = pd.DataFrame(records)
    det_csv = run_dir / "artifacts" / "detector" / "detections.csv"
    _save_csv(det_df, det_csv)

    counts_df = pd.DataFrame({"detections_per_image": det_counts})
    _save_csv(counts_df, run_dir / "artifacts" / "detector" / "detections_per_image.csv")
    return det_df


def _run_classifier(
    cfg: AnalyzeRunConfig,
    df: pd.DataFrame,
    img_root: Path,
    id_to_country: Optional[list[str]],
    region_map: Optional[Dict[str, str]],
    model: BollardNet,
    device: torch.device,
) -> pd.DataFrame:
    tfm = build_transforms(train=False, img_size=cfg.classifier.img_size)
    ds = BollardCropsDatasetWithIndex(df, str(img_root), tfm, expand=cfg.classifier.expand)

    loader = DataLoader(
        ds,
        batch_size=cfg.classifier.batch_size,
        shuffle=False,
        num_workers=cfg.classifier.num_workers,
        pin_memory=device.type == "cuda",
    )

    records = []
    for batch in tqdm(loader, desc="classify", leave=False, dynamic_ncols=True):
        images = batch["image"].to(device, non_blocking=True)
        meta = batch["meta"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        indices = batch["index"].cpu().numpy().tolist()

        with torch.no_grad():
            logits = model(images, meta)
            probs = torch.softmax(logits, dim=1)

        top5 = torch.topk(probs, k=min(5, probs.size(1)), dim=1).indices.cpu().numpy()
        top1 = probs.argmax(dim=1).cpu().numpy()
        top1_conf = probs.max(dim=1).values.cpu().numpy()

        for i, idx in enumerate(indices):
            row = df.iloc[int(idx)]
            true_id = int(labels[i].item())
            pred_id = int(top1[i])
            true_name = id_to_country[true_id] if id_to_country and true_id < len(id_to_country) else str(true_id)
            pred_name = id_to_country[pred_id] if id_to_country and pred_id < len(id_to_country) else str(pred_id)

            top5_ids = [int(v) for v in top5[i]]
            top5_names = [
                id_to_country[v] if id_to_country and v < len(id_to_country) else str(v)
                for v in top5_ids
            ]

            true_region = region_map.get(true_name) if region_map else None
            pred_region = region_map.get(pred_name) if region_map else None
            top5_regions = [region_map.get(name) for name in top5_names] if region_map else []

            correct_top1 = pred_id == true_id
            correct_top5 = true_id in top5_ids
            correct_region_top1 = bool(true_region and pred_region and true_region == pred_region)
            correct_region_top5 = bool(true_region and top5_regions and true_region in top5_regions)

            records.append({
                "image_path": str(row[PATH_COL]) if PATH_COL in row else "",
                "x1": float(row.get("x1", 0.0)),
                "y1": float(row.get("y1", 0.0)),
                "x2": float(row.get("x2", 1.0)),
                "y2": float(row.get("y2", 1.0)),
                "country_id": true_id,
                "country": true_name,
                "pred_id": pred_id,
                "pred_country": pred_name,
                "top1_conf": float(top1_conf[i]),
                "top5_ids": json.dumps(top5_ids),
                "top5_countries": json.dumps(top5_names),
                "correct_top1": bool(correct_top1),
                "correct_top5": bool(correct_top5),
                "region": true_region,
                "pred_region": pred_region,
                "correct_region_top1": bool(correct_region_top1),
                "correct_region_top5": bool(correct_region_top5),
            })

    return pd.DataFrame(records)


def _compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    total = len(df)
    top1 = float(df["correct_top1"].mean()) if "correct_top1" in df else 0.0
    top5 = float(df["correct_top5"].mean()) if "correct_top5" in df else 0.0
    metrics = {
        "samples": int(total),
        "top1_country": top1,
        "top5_country": top5,
    }
    if "correct_region_top1" in df:
        region_df = df[df["region"].notna()].copy() if "region" in df else df
        if not region_df.empty:
            metrics["top1_region"] = float(region_df["correct_region_top1"].mean())
            metrics["top5_region"] = float(region_df["correct_region_top5"].mean())
    return metrics


def _group_accuracy(df: pd.DataFrame, group_col: str, min_support: int) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    grouped = df[df[group_col].notna()].groupby(group_col)["correct_top1"].agg(["mean", "count"]).reset_index()
    grouped = grouped.rename(columns={"mean": "top1", "count": "support"})
    grouped = grouped[grouped["support"] >= min_support].copy()
    grouped = grouped.sort_values("top1", ascending=True)
    return grouped


def _confusion_pairs(df: pd.DataFrame, true_col: str, pred_col: str, top_k: int) -> pd.DataFrame:
    if true_col not in df.columns or pred_col not in df.columns:
        return pd.DataFrame()
    mis = df[df[true_col].notna() & df[pred_col].notna() & (df[true_col] != df[pred_col])].copy()
    if mis.empty:
        return pd.DataFrame()
    pairs = mis.groupby([true_col, pred_col]).size().reset_index(name="count")
    pairs = pairs.sort_values("count", ascending=False).head(top_k)
    return pairs


def _save_gallery(
    df: pd.DataFrame,
    img_root: Path,
    out_dir: Path,
    title: str,
    expand: float,
    max_items: int,
) -> list[str]:
    if df.empty:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    font = _load_font(16)
    rel_paths = []

    for i, row in enumerate(df.head(max_items).itertuples(index=False)):
        img_path = img_root / getattr(row, "image_path")
        if not img_path.exists():
            continue
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                crop = _crop_with_bbox(
                    im,
                    float(getattr(row, "x1")),
                    float(getattr(row, "y1")),
                    float(getattr(row, "x2")),
                    float(getattr(row, "y2")),
                    expand=expand,
                )
                lines = [
                    f"T: {getattr(row, 'country', '')}",
                    f"P: {getattr(row, 'pred_country', '')} ({getattr(row, 'top1_conf', 0.0):.2f})",
                ]
                draw = ImageDraw.Draw(crop)
                text = "\n".join(lines)
                bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2)
                text_h = bbox[3] - bbox[1]
                pad = 4
                draw.rectangle((0, 0, crop.size[0], text_h + pad * 2), fill=(0, 0, 0))
                draw.multiline_text((pad, pad), text, fill=(255, 255, 255), font=font, spacing=2)

                out_path = out_dir / f"{title}_{i:03d}.jpg"
                crop.save(out_path)
                rel_paths.append(str(out_path))
        except Exception:
            continue

    return rel_paths


def _render_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No data.</p>"
    return df.to_html(index=False, escape=True)


def _build_report_section(title: str, body: str) -> str:
    return f"<section><h2>{title}</h2>{body}</section>"


def _relative_paths(paths: list[str], base_dir: Path) -> list[str]:
    rel = []
    for p in paths:
        try:
            rel.append(str(Path(p).relative_to(base_dir)))
        except Exception:
            rel.append(p)
    return rel


def run_analyze_run(cfg: AnalyzeRunConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    out_dir = Path(cfg.output.out_dir)
    ensure_dir(out_dir)

    run_name = cfg.output.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = out_dir / run_name
    if run_dir.exists():
        suffix = 1
        while (out_dir / f"{run_name}_{suffix:02d}").exists():
            suffix += 1
        run_dir = out_dir / f"{run_name}_{suffix:02d}"
    ensure_dir(run_dir)

    log_path = run_dir / "analysis.log"
    logger = setup_logger(log_path)

    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = resolve_device(cfg.device)
    logger.info("Using device: %s", device)

    main_df = pd.read_csv(cfg.data.main_csv)
    golden_df = pd.read_csv(cfg.data.golden_csv) if cfg.data.golden_csv else None

    id_to_country = load_id_to_country(cfg.data.country_map_json)
    id_to_country, country_to_id = _build_country_mappings(id_to_country, main_df)

    region_map = _build_region_map(cfg, golden_df)
    if cfg.data.region_map_json:
        region_note = f"Region mapping: {cfg.data.region_map_json}"
    elif region_map is not None:
        region_note = "Region mapping: derived from golden dataset continent"
    else:
        region_note = "Region mapping: unavailable (region metrics skipped)"

    report_sections: list[str] = []

    # Dataset analyzer: main
    main_stats = {}
    if not main_df.empty:
        main_df = main_df.copy()
        if "country" not in main_df.columns and id_to_country is not None:
            main_df["country"] = main_df[LABEL_COL].apply(
                lambda idx: id_to_country[int(idx)] if int(idx) < len(id_to_country) else str(idx)
            )

        if region_map and "country" in main_df.columns:
            main_df = _maybe_add_region_by_country(main_df, "country", region_map)

        main_stats = _dataset_summary(main_df, "country", "region" if "region" in main_df.columns else None)
        _save_json(main_stats, run_dir / "artifacts" / "main" / "summary.json")
        main_counts = _value_counts(main_df, "country") if "country" in main_df.columns else pd.DataFrame()
        main_region_counts = _value_counts(main_df, "region") if "region" in main_df.columns else pd.DataFrame()

        _save_csv(main_counts, run_dir / "artifacts" / "main" / "country_counts.csv")
        _save_csv(main_region_counts, run_dir / "artifacts" / "main" / "region_counts.csv")

        if "cls" in main_df.columns:
            main_df["class_name"] = main_df["cls"].astype(int).apply(
                lambda x: _class_name(x, cfg.detector.class_names)
            )
        else:
            main_df["class_name"] = cfg.data.golden_default_category
        class_counts = _value_counts(main_df, "class_name")
        if "class_name" not in class_counts.columns:
            class_counts = pd.DataFrame({"class_name": [], "count": []})
        _save_csv(class_counts, run_dir / "artifacts" / "main" / "class_counts.csv")

        area_aspect = _calc_bbox_area_aspect(main_df, "bbox")
        if not area_aspect.empty:
            _plot_hist(area_aspect["bbox_area"], run_dir / "artifacts" / "main" / "bbox_area_hist.png", "BBox area", "area fraction")
            _plot_hist(area_aspect["bbox_aspect"], run_dir / "artifacts" / "main" / "bbox_aspect_hist.png", "BBox aspect", "aspect ratio")

        for cls_name in class_counts["class_name"].head(cfg.output.max_categories_plots).tolist():
            subset = main_df[main_df["class_name"] == cls_name]
            sub_area = _calc_bbox_area_aspect(subset, "bbox")
            if sub_area.empty:
                continue
            safe_name = cls_name.replace("/", "_")
            _plot_hist(sub_area["bbox_area"], run_dir / "artifacts" / "main" / f"bbox_area_{safe_name}.png", f"BBox area ({cls_name})", "area fraction")
            _plot_hist(sub_area["bbox_aspect"], run_dir / "artifacts" / "main" / f"bbox_aspect_{safe_name}.png", f"BBox aspect ({cls_name})", "aspect ratio")

        top_country, bottom_country = _top_bottom(main_df, "country") if "country" in main_df.columns else (pd.DataFrame(), pd.DataFrame())
        top_region, bottom_region = _top_bottom(main_df, "region") if "region" in main_df.columns else (pd.DataFrame(), pd.DataFrame())

        main_section = ""
        main_section += f"<p>Images: {main_stats.get('n_images', 0)} | Objects: {main_stats.get('n_objects', 0)} | Objects/image: {main_stats.get('objects_per_image', 0):.2f}</p>"
        main_section += f"<p>Countries: {main_stats.get('n_countries', 0)}"
        if "n_regions" in main_stats:
            main_section += f" | Regions: {main_stats['n_regions']}"
        main_section += "</p>"
        main_section += "<h3>Top countries</h3>" + _render_table(top_country)
        main_section += "<h3>Bottom countries</h3>" + _render_table(bottom_country)
        if not top_region.empty:
            main_section += "<h3>Top regions</h3>" + _render_table(top_region)
            main_section += "<h3>Bottom regions</h3>" + _render_table(bottom_region)
        main_section += "<h3>Class distribution</h3>" + _render_table(class_counts.head(20))
        main_section += "<h3>Geometry</h3>"
        if (run_dir / "artifacts" / "main" / "bbox_area_hist.png").exists():
            main_section += f"<img src='artifacts/main/bbox_area_hist.png' width='420'>"
        if (run_dir / "artifacts" / "main" / "bbox_aspect_hist.png").exists():
            main_section += f"<img src='artifacts/main/bbox_aspect_hist.png' width='420'>"
        if not class_counts.empty:
            main_section += "<h3>Geometry by class</h3>"
            for cls_name in class_counts["class_name"].head(cfg.output.max_categories_plots).tolist():
                safe_name = cls_name.replace("/", "_")
                area_path = run_dir / "artifacts" / "main" / f"bbox_area_{safe_name}.png"
                aspect_path = run_dir / "artifacts" / "main" / f"bbox_aspect_{safe_name}.png"
                if area_path.exists():
                    main_section += f"<img src='artifacts/main/bbox_area_{safe_name}.png' width='360'>"
                if aspect_path.exists():
                    main_section += f"<img src='artifacts/main/bbox_aspect_{safe_name}.png' width='360'>"
        report_sections.append(_build_report_section("Main dataset", main_section))

    # Dataset analyzer: golden
    if golden_df is not None and not golden_df.empty:
        golden_stats = {}
        golden_df = golden_df.copy()
        if "country" in golden_df.columns:
            golden_df["country_code"] = golden_df["country"].apply(golden_country_to_code)
        if "country_code" in golden_df.columns:
            golden_df["country_code"] = golden_df["country_code"].fillna("")
        golden_df["class_name"] = cfg.data.golden_default_category
        golden_df = _maybe_add_region_by_country(golden_df, "country_code", region_map, out_col="region")
        if "region" not in golden_df.columns and "continent" in golden_df.columns:
            golden_df["region"] = golden_df["continent"]

        golden_stats = _dataset_summary(golden_df, "country", "region" if "region" in golden_df.columns else None)
        _save_json(golden_stats, run_dir / "artifacts" / "golden" / "summary.json")
        golden_country_counts = _value_counts(golden_df, "country")
        golden_region_counts = _value_counts(golden_df, "region") if "region" in golden_df.columns else pd.DataFrame()
        _save_csv(golden_country_counts, run_dir / "artifacts" / "golden" / "country_counts.csv")
        _save_csv(golden_region_counts, run_dir / "artifacts" / "golden" / "region_counts.csv")
        class_counts = _value_counts(golden_df, "class_name")
        if "class_name" not in class_counts.columns:
            class_counts = pd.DataFrame({"class_name": [], "count": []})
        _save_csv(class_counts, run_dir / "artifacts" / "golden" / "class_counts.csv")

        golden_geom = _calc_crop_area_aspect(golden_df, "crop")
        if not golden_geom.empty:
            _plot_hist(golden_geom["crop_area"], run_dir / "artifacts" / "golden" / "bbox_area_hist.png", "BBox area", "area fraction")
            _plot_hist(golden_geom["crop_aspect"], run_dir / "artifacts" / "golden" / "bbox_aspect_hist.png", "BBox aspect", "aspect ratio")
        if not class_counts.empty:
            for cls_name in class_counts["class_name"].head(cfg.output.max_categories_plots).tolist():
                subset = golden_df[golden_df["class_name"] == cls_name]
                sub_geom = _calc_crop_area_aspect(subset, "crop")
                if sub_geom.empty:
                    continue
                safe_name = cls_name.replace("/", "_")
                _plot_hist(sub_geom["crop_area"], run_dir / "artifacts" / "golden" / f"bbox_area_{safe_name}.png", f"BBox area ({cls_name})", "area fraction")
                _plot_hist(sub_geom["crop_aspect"], run_dir / "artifacts" / "golden" / f"bbox_aspect_{safe_name}.png", f"BBox aspect ({cls_name})", "aspect ratio")

        top_country, bottom_country = _top_bottom(golden_df, "country")
        top_region, bottom_region = _top_bottom(golden_df, "region") if "region" in golden_df.columns else (pd.DataFrame(), pd.DataFrame())

        golden_section = ""
        golden_section += f"<p>Images: {golden_stats.get('n_images', 0)} | Objects: {golden_stats.get('n_objects', 0)} | Objects/image: {golden_stats.get('objects_per_image', 0):.2f}</p>"
        golden_section += f"<p>Countries: {golden_stats.get('n_countries', 0)}"
        if "n_regions" in golden_stats:
            golden_section += f" | Regions: {golden_stats['n_regions']}"
        golden_section += "</p>"
        golden_section += "<h3>Top countries</h3>" + _render_table(top_country)
        golden_section += "<h3>Bottom countries</h3>" + _render_table(bottom_country)
        if not top_region.empty:
            golden_section += "<h3>Top regions</h3>" + _render_table(top_region)
            golden_section += "<h3>Bottom regions</h3>" + _render_table(bottom_region)
        golden_section += "<h3>Class distribution</h3>" + _render_table(class_counts.head(20))
        golden_section += "<h3>Geometry</h3>"
        if (run_dir / "artifacts" / "golden" / "bbox_area_hist.png").exists():
            golden_section += f"<img src='artifacts/golden/bbox_area_hist.png' width='420'>"
        if (run_dir / "artifacts" / "golden" / "bbox_aspect_hist.png").exists():
            golden_section += f"<img src='artifacts/golden/bbox_aspect_hist.png' width='420'>"
        if not class_counts.empty:
            golden_section += "<h3>Geometry by class</h3>"
            for cls_name in class_counts["class_name"].head(cfg.output.max_categories_plots).tolist():
                safe_name = cls_name.replace("/", "_")
                area_path = run_dir / "artifacts" / "golden" / f"bbox_area_{safe_name}.png"
                aspect_path = run_dir / "artifacts" / "golden" / f"bbox_aspect_{safe_name}.png"
                if area_path.exists():
                    golden_section += f"<img src='artifacts/golden/bbox_area_{safe_name}.png' width='360'>"
                if aspect_path.exists():
                    golden_section += f"<img src='artifacts/golden/bbox_aspect_{safe_name}.png' width='360'>"
        report_sections.append(_build_report_section("Golden dataset", golden_section))

    # Detector prediction analyzer
    det_df = pd.DataFrame()
    if cfg.detector.enabled:
        logger.info("Running detector on main dataset")
        det_df = _run_detector(cfg, logger, main_df, Path(cfg.data.main_img_root), run_dir)
        if not det_df.empty:
            det_df["class_name"] = det_df["cls"].astype(int).apply(
                lambda x: _class_name(x, cfg.detector.class_names)
            )

        det_section = ""
        det_counts_path = run_dir / "artifacts" / "detector" / "detections_per_image.csv"
        det_conf_hist = run_dir / "artifacts" / "detector" / "conf_hist.png"
        det_class_counts = _value_counts(det_df, "class_name") if not det_df.empty else pd.DataFrame({"class_name": [], "count": []})
        if "class_name" not in det_class_counts.columns:
            det_class_counts = pd.DataFrame({"class_name": [], "count": []})
        _save_csv(det_class_counts, run_dir / "artifacts" / "detector" / "class_counts.csv")

        if det_counts_path.exists():
            det_counts = pd.read_csv(det_counts_path)["detections_per_image"].tolist()
            _plot_hist(det_counts, run_dir / "artifacts" / "detector" / "detections_hist.png", "Detections per image", "detections")

        if not det_df.empty:
            _plot_hist(det_df["conf"], det_conf_hist, "Detection confidence", "confidence")
            for cls_name in det_class_counts["class_name"].head(cfg.output.max_categories_plots).tolist():
                subset = det_df[det_df["class_name"] == cls_name]
                safe_name = cls_name.replace("/", "_")
                _plot_hist(
                    subset["conf"],
                    run_dir / "artifacts" / "detector" / f"conf_hist_{safe_name}.png",
                    f"Confidence ({cls_name})",
                    "confidence",
                )

        det_section += "<h3>Detections per image</h3>"
        if (run_dir / "artifacts" / "detector" / "detections_hist.png").exists():
            det_section += "<img src='artifacts/detector/detections_hist.png' width='420'>"
        det_section += "<h3>Confidence distribution</h3>"
        if det_conf_hist.exists():
            det_section += "<img src='artifacts/detector/conf_hist.png' width='420'>"
        if not det_class_counts.empty:
            det_section += "<h3>Confidence by class</h3>"
            for cls_name in det_class_counts["class_name"].head(cfg.output.max_categories_plots).tolist():
                safe_name = cls_name.replace("/", "_")
                conf_path = run_dir / "artifacts" / "detector" / f"conf_hist_{safe_name}.png"
                if conf_path.exists():
                    det_section += f"<img src='artifacts/detector/conf_hist_{safe_name}.png' width='420'>"

        if not det_class_counts.empty:
            det_section += "<h3>Class counts</h3>" + _render_table(det_class_counts.head(20))

        # Per-category galleries
        if not det_df.empty:
            img_root = Path(cfg.data.main_img_root)
            color_map = _make_color_map(det_df["cls"].astype(int).tolist())
            gallery_by_class: dict[str, list[str]] = {}
            for cls_name in det_class_counts["class_name"].head(cfg.output.max_categories_plots).tolist():
                cls_subset = det_df[det_df["class_name"] == cls_name]
                img_ids = cls_subset["image_path"].dropna().unique().tolist()
                random.shuffle(img_ids)
                img_ids = img_ids[: cfg.output.gallery_per_category]
                gallery_dir = run_dir / "artifacts" / "detector" / "gallery" / cls_name.replace("/", "_")
                gallery_dir.mkdir(parents=True, exist_ok=True)
                gallery_by_class[cls_name] = []
                for i, rel_path in enumerate(img_ids):
                    img_path = img_root / rel_path
                    if not img_path.exists():
                        continue
                    try:
                        with Image.open(img_path) as im:
                            im = im.convert("RGB")
                            boxes = cls_subset[cls_subset["image_path"] == rel_path][
                                ["x1", "y1", "x2", "y2", "cls"]
                            ].to_dict("records")
                            annotated = _draw_boxes(im, boxes, color_map)
                            out_path = gallery_dir / f"det_{i:03d}.jpg"
                            annotated.save(out_path)
                            gallery_by_class[cls_name].append(str(out_path))
                    except Exception:
                        continue

            if gallery_by_class:
                det_section += "<h3>Sample detections by class</h3>"
                for cls_name, paths in gallery_by_class.items():
                    if not paths:
                        continue
                    rel = _relative_paths(paths, run_dir)
                    det_section += f"<h4>{cls_name}</h4><div class='grid'>"
                    det_section += "".join([f"<img src='{p}' loading='lazy'>" for p in rel])
                    det_section += "</div>"

        report_sections.append(_build_report_section("Detector predictions (main)", det_section))

    # Classifier evaluation
    if id_to_country is None or country_to_id is None:
        logger.info("Skipping classifier eval (missing country map)")
    else:
        class_sections = []
        classifier_model, _ = _load_classifier(cfg, device, logger)
        # Golden classifier eval
        if golden_df is not None and not golden_df.empty:
            avg_bbox_w, avg_bbox_h = compute_avg_bbox_wh(main_df, label="Main")
            golden_eval_df = _prepare_golden_df_for_classifier(
                golden_df,
                country_to_id,
                avg_bbox_w=avg_bbox_w,
                avg_bbox_h=avg_bbox_h,
            )
            golden_preds = _run_classifier(
                cfg,
                golden_eval_df,
                Path(cfg.data.golden_img_root or Path(cfg.data.golden_csv or ".").parent),
                id_to_country,
                region_map,
                classifier_model,
                device,
            )
            golden_preds_path = run_dir / "artifacts" / "classifier" / "golden" / "predictions.csv"
            _save_csv(golden_preds, golden_preds_path)
            metrics = _compute_metrics(golden_preds)
            _save_json(metrics, run_dir / "artifacts" / "classifier" / "golden" / "metrics.json")

            country_groups = _group_accuracy(golden_preds, "country", cfg.output.min_support)
            region_groups = _group_accuracy(golden_preds, "region", cfg.output.min_support)
            _save_csv(country_groups, run_dir / "artifacts" / "classifier" / "golden" / "country_groups.csv")
            _save_csv(region_groups, run_dir / "artifacts" / "classifier" / "golden" / "region_groups.csv")

            country_conf = _confusion_pairs(golden_preds, "country", "pred_country", cfg.output.top_k)
            region_conf = _confusion_pairs(golden_preds, "region", "pred_region", cfg.output.top_k)
            _save_csv(country_conf, run_dir / "artifacts" / "classifier" / "golden" / "confusion_country.csv")
            _save_csv(region_conf, run_dir / "artifacts" / "classifier" / "golden" / "confusion_region.csv")

            correct = golden_preds[golden_preds["correct_top1"]].sample(
                n=min(cfg.output.gallery_size, len(golden_preds[golden_preds["correct_top1"]])),
                random_state=cfg.seed,
            ) if not golden_preds.empty else pd.DataFrame()
            incorrect = golden_preds[~golden_preds["correct_top1"]]
            incorrect_sample = incorrect.sample(
                n=min(cfg.output.gallery_size, len(incorrect)),
                random_state=cfg.seed,
            ) if not incorrect.empty else pd.DataFrame()
            high_conf_wrong = incorrect.sort_values("top1_conf", ascending=False).head(cfg.output.gallery_size)

            galleries = {}
            img_root = Path(cfg.data.golden_img_root or Path(cfg.data.golden_csv or ".").parent)
            galleries["correct"] = _save_gallery(correct, img_root, run_dir / "artifacts" / "classifier" / "golden" / "gallery_correct", "correct", cfg.classifier.expand, cfg.output.gallery_size)
            galleries["incorrect"] = _save_gallery(incorrect_sample, img_root, run_dir / "artifacts" / "classifier" / "golden" / "gallery_incorrect", "incorrect", cfg.classifier.expand, cfg.output.gallery_size)
            galleries["high_conf_wrong"] = _save_gallery(high_conf_wrong, img_root, run_dir / "artifacts" / "classifier" / "golden" / "gallery_high_conf_wrong", "highconf_wrong", cfg.classifier.expand, cfg.output.gallery_size)

            section = ""
            section += "<p>Top-1 country: {:.3f} | Top-5 country: {:.3f}</p>".format(metrics.get("top1_country", 0.0), metrics.get("top5_country", 0.0))
            if "top1_region" in metrics:
                section += "<p>Top-1 region: {:.3f} | Top-5 region: {:.3f}</p>".format(metrics.get("top1_region", 0.0), metrics.get("top5_region", 0.0))
            section += "<h3>Best/worst countries</h3>"
            section += "<h4>Worst</h4>" + _render_table(country_groups.head(cfg.output.top_k))
            section += "<h4>Best</h4>" + _render_table(country_groups.tail(cfg.output.top_k))
            if not region_groups.empty:
                section += "<h3>Best/worst regions</h3>"
                section += "<h4>Worst</h4>" + _render_table(region_groups.head(cfg.output.top_k))
                section += "<h4>Best</h4>" + _render_table(region_groups.tail(cfg.output.top_k))
            section += "<h3>Top confusion pairs (country)</h3>" + _render_table(country_conf)
            if not region_conf.empty:
                section += "<h3>Top confusion pairs (region)</h3>" + _render_table(region_conf)

            if galleries["correct"]:
                rel = _relative_paths(galleries["correct"], run_dir)
                section += "<h3>Correct samples</h3><div class='grid'>" + "".join([f"<img src='{p}' loading='lazy'>" for p in rel]) + "</div>"
            if galleries["incorrect"]:
                rel = _relative_paths(galleries["incorrect"], run_dir)
                section += "<h3>Incorrect samples</h3><div class='grid'>" + "".join([f"<img src='{p}' loading='lazy'>" for p in rel]) + "</div>"
            if galleries["high_conf_wrong"]:
                rel = _relative_paths(galleries["high_conf_wrong"], run_dir)
                section += "<h3>High-confidence wrong</h3><div class='grid'>" + "".join([f"<img src='{p}' loading='lazy'>" for p in rel]) + "</div>"

            class_sections.append(_build_report_section("Classifier (golden)", section))

        # Main classifier eval
        if not main_df.empty:
            eval_df = main_df.copy()
            if cfg.detector.enabled and not det_df.empty:
                if "country" in eval_df.columns:
                    image_country = eval_df.groupby(PATH_COL)["country"].first().to_dict()
                    image_country_id = eval_df.groupby(PATH_COL)[LABEL_COL].first().to_dict() if LABEL_COL in eval_df.columns else {}
                else:
                    image_country = {}
                    image_country_id = {}

                det_df = det_df.copy()
                det_df["country"] = det_df["image_path"].map(image_country)
                det_df[LABEL_COL] = det_df["image_path"].map(image_country_id)
                det_df = det_df.dropna(subset=[LABEL_COL])
                det_df[LABEL_COL] = det_df[LABEL_COL].astype(int)
                eval_df = det_df

            if all(c in eval_df.columns for c in [PATH_COL, LABEL_COL, *BBOX_COLS, *META_COLS]):
                main_preds = _run_classifier(
                    cfg,
                    eval_df,
                    Path(cfg.data.main_img_root),
                    id_to_country,
                    region_map,
                    classifier_model,
                    device,
                )
                main_preds_path = run_dir / "artifacts" / "classifier" / "main" / "predictions.csv"
                _save_csv(main_preds, main_preds_path)
                metrics = _compute_metrics(main_preds)
                _save_json(metrics, run_dir / "artifacts" / "classifier" / "main" / "metrics.json")

                country_groups = _group_accuracy(main_preds, "country", cfg.output.min_support)
                region_groups = _group_accuracy(main_preds, "region", cfg.output.min_support)
                _save_csv(country_groups, run_dir / "artifacts" / "classifier" / "main" / "country_groups.csv")
                _save_csv(region_groups, run_dir / "artifacts" / "classifier" / "main" / "region_groups.csv")

                country_conf = _confusion_pairs(main_preds, "country", "pred_country", cfg.output.top_k)
                region_conf = _confusion_pairs(main_preds, "region", "pred_region", cfg.output.top_k)
                _save_csv(country_conf, run_dir / "artifacts" / "classifier" / "main" / "confusion_country.csv")
                _save_csv(region_conf, run_dir / "artifacts" / "classifier" / "main" / "confusion_region.csv")

                correct = main_preds[main_preds["correct_top1"]].sample(
                    n=min(cfg.output.gallery_size, len(main_preds[main_preds["correct_top1"]])),
                    random_state=cfg.seed,
                ) if not main_preds.empty else pd.DataFrame()
                incorrect = main_preds[~main_preds["correct_top1"]]
                incorrect_sample = incorrect.sample(
                    n=min(cfg.output.gallery_size, len(incorrect)),
                    random_state=cfg.seed,
                ) if not incorrect.empty else pd.DataFrame()
                high_conf_wrong = incorrect.sort_values("top1_conf", ascending=False).head(cfg.output.gallery_size)

                galleries = {}
                img_root = Path(cfg.data.main_img_root)
                galleries["correct"] = _save_gallery(correct, img_root, run_dir / "artifacts" / "classifier" / "main" / "gallery_correct", "correct", cfg.classifier.expand, cfg.output.gallery_size)
                galleries["incorrect"] = _save_gallery(incorrect_sample, img_root, run_dir / "artifacts" / "classifier" / "main" / "gallery_incorrect", "incorrect", cfg.classifier.expand, cfg.output.gallery_size)
                galleries["high_conf_wrong"] = _save_gallery(high_conf_wrong, img_root, run_dir / "artifacts" / "classifier" / "main" / "gallery_high_conf_wrong", "highconf_wrong", cfg.classifier.expand, cfg.output.gallery_size)

                section = ""
                section += "<p>Top-1 country: {:.3f} | Top-5 country: {:.3f}</p>".format(metrics.get("top1_country", 0.0), metrics.get("top5_country", 0.0))
                if "top1_region" in metrics:
                    section += "<p>Top-1 region: {:.3f} | Top-5 region: {:.3f}</p>".format(metrics.get("top1_region", 0.0), metrics.get("top5_region", 0.0))
                section += "<h3>Best/worst countries</h3>"
                section += "<h4>Worst</h4>" + _render_table(country_groups.head(cfg.output.top_k))
                section += "<h4>Best</h4>" + _render_table(country_groups.tail(cfg.output.top_k))
                if not region_groups.empty:
                    section += "<h3>Best/worst regions</h3>"
                    section += "<h4>Worst</h4>" + _render_table(region_groups.head(cfg.output.top_k))
                    section += "<h4>Best</h4>" + _render_table(region_groups.tail(cfg.output.top_k))
                section += "<h3>Top confusion pairs (country)</h3>" + _render_table(country_conf)
                if not region_conf.empty:
                    section += "<h3>Top confusion pairs (region)</h3>" + _render_table(region_conf)

                if galleries["correct"]:
                    rel = _relative_paths(galleries["correct"], run_dir)
                    section += "<h3>Correct samples</h3><div class='grid'>" + "".join([f"<img src='{p}' loading='lazy'>" for p in rel]) + "</div>"
                if galleries["incorrect"]:
                    rel = _relative_paths(galleries["incorrect"], run_dir)
                    section += "<h3>Incorrect samples</h3><div class='grid'>" + "".join([f"<img src='{p}' loading='lazy'>" for p in rel]) + "</div>"
                if galleries["high_conf_wrong"]:
                    rel = _relative_paths(galleries["high_conf_wrong"], run_dir)
                    section += "<h3>High-confidence wrong</h3><div class='grid'>" + "".join([f"<img src='{p}' loading='lazy'>" for p in rel]) + "</div>"

                class_sections.append(_build_report_section("Classifier (main)", section))

        report_sections.extend(class_sections)

    css = """
    <style>
    body { font-family: Arial, sans-serif; margin: 24px; color: #1b1b1b; }
    h1 { margin-bottom: 0; }
    h2 { margin-top: 32px; }
    h3 { margin-top: 20px; }
    table { border-collapse: collapse; margin: 12px 0; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }
    th { background: #f5f5f5; }
    img { margin: 6px; border: 1px solid #ddd; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 8px; }
    section { margin-bottom: 32px; }
    </style>
    """

    header = f"<h1>Single-run analysis</h1><p>Run: {run_dir.name}</p><p>{region_note}</p>"
    body = header + "\n".join(report_sections)
    html = f"<!doctype html><html><head><meta charset='utf-8'>{css}</head><body>{body}</body></html>"
    report_path = run_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")

    logger.info("Report saved: %s", report_path)
