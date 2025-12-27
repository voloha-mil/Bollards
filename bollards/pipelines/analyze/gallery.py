from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image

from bollards.data.bboxes import crop_image_from_norm_bbox
from bollards.train.visuals import annotate_pil_images


def save_gallery(
    df,
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
        except Exception:
            continue

    if not crops:
        return []

    annotated = annotate_pil_images(crops, text_blocks)
    for img, out_path in zip(annotated, out_paths):
        img.save(out_path)
        rel_paths.append(str(out_path))

    return rel_paths
