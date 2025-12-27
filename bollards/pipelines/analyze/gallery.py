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


def _split_long_token(token: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    parts: list[str] = []
    current = ""
    for ch in token:
        trial = current + ch
        if current and draw.textlength(trial, font=font) > max_width:
            parts.append(current)
            current = ch
        else:
            current = trial
    if current:
        parts.append(current)
    return parts or [token]


def _wrap_line(line: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if not line:
        return [""]
    if draw.textlength(line, font=font) <= max_width:
        return [line]
    tokens = line.split(" ")
    if len(tokens) == 1:
        return _split_long_token(line, draw, font, max_width)
    lines: list[str] = []
    current = ""
    for token in tokens:
        parts = [token]
        if draw.textlength(token, font=font) > max_width:
            parts = _split_long_token(token, draw, font, max_width)
        for part in parts:
            if not current:
                current = part
                continue
            trial = f"{current} {part}"
            if draw.textlength(trial, font=font) <= max_width:
                current = trial
            else:
                lines.append(current)
                current = part
    if current:
        lines.append(current)
    return lines


def _layout_text(
    lines: list[str],
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    max_width: int,
    spacing: int,
) -> tuple[list[str], int]:
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(_wrap_line(line, draw, font, max_width))
    text = "\n".join(wrapped)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    text_h = bbox[3] - bbox[1]
    return wrapped, text_h


def _render_label(crop: Image.Image, lines: list[str]) -> Image.Image:
    base_size = int(round(min(crop.size) * 0.06))
    max_size = 40
    font_size = max(12, min(max_size, base_size))
    max_label_h = max(36, int(round(crop.height * 0.35)))
    dummy = Image.new("RGB", (max(1, crop.width), 1))
    draw = ImageDraw.Draw(dummy)
    for size in range(font_size, 11, -2):
        font = load_font(size)
        pad = max(4, int(round(size * 0.3)))
        spacing = max(2, int(round(size * 0.2)))
        max_width = max(10, crop.width - 2 * pad)
        wrapped, text_h = _layout_text(lines, draw, font, max_width, spacing)
        label_h = text_h + pad * 2
        if label_h <= max_label_h or size == 12:
            text = "\n".join(wrapped)
            label = Image.new("RGB", (crop.width, label_h), (0, 0, 0))
            label_draw = ImageDraw.Draw(label)
            label_draw.multiline_text((pad, pad), text, fill=(255, 255, 255), font=font, spacing=spacing)
            return label
    return Image.new("RGB", (crop.width, max_label_h), (0, 0, 0))


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
                label = _render_label(crop, lines)
                combined = Image.new("RGB", (crop.width, crop.height + label.height), (0, 0, 0))
                combined.paste(label, (0, 0))
                combined.paste(crop, (0, label.height))

                out_path = out_dir / f"{title}_{i:03d}.jpg"
                combined.save(out_path)
                rel_paths.append(str(out_path))
        except Exception:
            continue

    return rel_paths
