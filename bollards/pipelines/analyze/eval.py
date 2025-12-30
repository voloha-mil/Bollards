from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bollards.pipelines.analyze.config import AnalyzeRunConfig
from bollards.constants import PATH_COL
from bollards.data.bboxes import bbox_xyxy_norm_to_center, normalize_bbox_xyxy_px
from bollards.data.datasets import BollardCropsDataset
from bollards.data.transforms import build_transforms
from bollards.models.classifier import BollardNet
from bollards.models.detector_yolo import run_inference_batch
from bollards.models.loaders import load_detector
from bollards.pipelines.analyze.reporting import save_csv
from bollards.utils.runtime import resolve_device
from bollards.utils.seeding import make_torch_generator, seed_worker


class BollardCropsDatasetWithIndex(BollardCropsDataset):
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        item["index"] = torch.tensor(idx, dtype=torch.long)
        return item


def run_classifier(
    cfg: AnalyzeRunConfig,
    df: pd.DataFrame,
    img_root,
    id_to_country: Optional[list[str]],
    region_map: Optional[Dict[str, str]],
    model: BollardNet,
    device: torch.device,
) -> pd.DataFrame:
    tfm = build_transforms(train=False, img_size=cfg.classifier.img_size)
    ds = BollardCropsDatasetWithIndex(df, str(img_root), tfm, expand=cfg.classifier.expand)

    loader_kwargs = {
        "num_workers": cfg.classifier.num_workers,
        "pin_memory": device.type == "cuda",
        "generator": make_torch_generator(cfg.seed, "analyze_loader"),
    }
    if cfg.classifier.num_workers > 0:
        loader_kwargs["worker_init_fn"] = seed_worker

    loader = DataLoader(
        ds,
        batch_size=cfg.classifier.batch_size,
        shuffle=False,
        **loader_kwargs,
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
                "image_id": str(row.get("image_id", "")) if "image_id" in row else "",
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


def filter_detections(
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

        if conf < cfg.detector.min_conf:
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


def run_detector(
    cfg: AnalyzeRunConfig,
    logger: logging.Logger,
    main_df: pd.DataFrame,
    img_root: Path,
    run_dir: Path,
) -> pd.DataFrame:
    detector = load_detector(
        weights_path=cfg.detector.weights_path,
        hf_repo=cfg.detector.hf_repo,
        hf_filename=cfg.detector.hf_filename,
        hf_cache=cfg.detector.hf_cache,
        logger=logger,
    )
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
            conf=cfg.detector.min_conf,
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
            filtered = filter_detections(cfg, boxes_xyxy, confs, clss)
            det_counts.append(len(filtered))
            if not filtered:
                continue

            h, w = result.orig_shape
            for det in filtered:
                x1n, y1n, x2n, y2n = normalize_bbox_xyxy_px(det["x1"], det["y1"], det["x2"], det["y2"], w, h)
                xc, yc, bw, bh = bbox_xyxy_norm_to_center(x1n, y1n, x2n, y2n)
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
    save_csv(det_df, det_csv)

    counts_df = pd.DataFrame({"detections_per_image": det_counts})
    save_csv(counts_df, run_dir / "artifacts" / "detector" / "detections_per_image.csv")
    return det_df
