from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from bollards.data.bboxes import crop_image_from_norm_bbox


def load_font(font_size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for p in candidates:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, size=font_size)
        except Exception:
            pass
    return ImageFont.load_default()


def save_gallery(
    df,
    img_root: Path,
    out_dir: Path,
    title: str,
    expand: float,
    max_items: int,
) -> list[str]:
    if df.empty:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(16)
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
