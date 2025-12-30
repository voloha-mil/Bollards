from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Optional

import pandas as pd
from PIL import Image

from bollards.data.bboxes import crop_image_from_norm_bbox
from bollards.utils.visuals.boxes import draw_boxes
from bollards.utils.visuals.annotate import annotate_pil_images


def save_gallery(
    df: pd.DataFrame,
    img_root: Path,
    out_dir: Path,
    title: str,
    expand: float,
    max_items: int,
    target_size: Optional[int] = None,
) -> list[str]:
    if df.empty:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    crops = []
    text_blocks = []
    out_paths = []
    rel_paths = []

    for i, row in enumerate(df.head(max_items).itertuples(index=False)):
        img_path = img_root / getattr(row, "image_path")
        if not img_path.exists():
            continue
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                crop = crop_image_from_norm_bbox(
                    im,
                    float(getattr(row, "x1")),
                    float(getattr(row, "y1")),
                    float(getattr(row, "x2")),
                    float(getattr(row, "y2")),
                    expand=expand,
                )
                if target_size and target_size > 0:
                    crop = crop.resize((target_size, target_size), resample=Image.BILINEAR)
                text_blocks.append([
                    "T:{true}  P:{pred}  p={conf:.2f}".format(
                        true=getattr(row, "country", ""),
                        pred=getattr(row, "pred_country", ""),
                        conf=float(getattr(row, "top1_conf", 0.0)),
                    )
                ])
                crops.append(crop)
                out_paths.append(out_dir / f"{title}_{i:03d}.jpg")
        except (OSError, ValueError):
            continue

    if not crops:
        return []

    annotated = annotate_pil_images(crops, text_blocks)
    for img, out_path in zip(annotated, out_paths):
        img.save(out_path)
        rel_paths.append(str(out_path))

    return rel_paths


def save_detection_galleries(
    df: pd.DataFrame,
    img_root: Path,
    out_root: Path,
    class_names: list[str],
    rng: Random,
    max_items: int,
    color_map: dict[int, tuple[int, int, int]],
    prefix: str = "det",
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    if df.empty or not class_names:
        return {}, {}

    required_cols = {"class_name", "image_path", "x1", "y1", "x2", "y2", "cls"}
    if not required_cols.issubset(df.columns):
        return {}, {}

    gallery_by_class: dict[str, list[str]] = {}
    labels_by_class: dict[str, list[str]] = {}

    for cls_name in class_names:
        cls_subset = df[df["class_name"] == cls_name]
        if cls_subset.empty:
            continue
        img_ids = cls_subset["image_path"].dropna().unique().tolist()
        if not img_ids:
            continue
        rng.shuffle(img_ids)
        img_ids = img_ids[:max_items]
        gallery_dir = out_root / cls_name.replace("/", "_")
        gallery_dir.mkdir(parents=True, exist_ok=True)
        gallery_by_class[cls_name] = []
        labels_by_class[cls_name] = []
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
                    annotated = draw_boxes(im, boxes, color_map)
                    out_path = gallery_dir / f"{prefix}_{i:03d}.jpg"
                    annotated.save(out_path)
                    gallery_by_class[cls_name].append(str(out_path))
                    labels_by_class[cls_name].append(str(rel_path))
            except (OSError, ValueError):
                continue

    return gallery_by_class, labels_by_class
