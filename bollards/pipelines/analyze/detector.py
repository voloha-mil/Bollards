from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from bollards.config import AnalyzeRunConfig
from bollards.constants import PATH_COL
from bollards.data.bboxes import bbox_xyxy_norm_to_center, normalize_bbox_xyxy_px
from bollards.detect.yolo import run_inference_batch
from bollards.pipelines.analyze.io import save_csv
from bollards.pipelines.common import load_detector, resolve_device


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
