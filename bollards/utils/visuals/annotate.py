import os
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from torchvision.utils import make_grid

from bollards.data.transforms import denormalize


def _load_font(font_size: int) -> ImageFont.ImageFont:
    """
    Load a readable TrueType font if available; fall back to PIL default.
    Works well on Ubuntu (DejaVu) and macOS (Arial).
    """
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


def _text_bbox(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont
) -> Tuple[int, int, int, int]:
    if hasattr(draw, "textbbox"):
        return draw.textbbox((0, 0), text, font=font)
    width, height = font.getsize(text)
    return (0, 0, width, height)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    if hasattr(draw, "textlength"):
        return float(draw.textlength(text, font=font))
    if hasattr(font, "getlength"):
        return float(font.getlength(text))
    bbox = _text_bbox(draw, text, font)
    return float(bbox[2] - bbox[0])


def _line_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
    bbox = _text_bbox(draw, "Ag", font)
    return max(1, int(bbox[3] - bbox[1]))


def _split_long_word(
    draw: ImageDraw.ImageDraw, word: str, font: ImageFont.ImageFont, max_width: int
) -> List[str]:
    parts = []
    current = ""
    for ch in word:
        trial = f"{current}{ch}"
        if not current or _text_width(draw, trial, font) <= max_width:
            current = trial
        else:
            parts.append(current)
            current = ch
    if current:
        parts.append(current)
    return parts


def _wrap_text(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int
) -> List[str]:
    if not text:
        return [""]
    words = text.split()
    if not words:
        return [text]

    lines: List[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if not current or _text_width(draw, trial, font) <= max_width:
            current = trial
            continue

        lines.append(current)
        current = ""

        if _text_width(draw, word, font) <= max_width:
            current = word
        else:
            parts = _split_long_word(draw, word, font, max_width)
            lines.extend(parts[:-1])
            current = parts[-1] if parts else ""

    if current:
        lines.append(current)

    return lines


def annotate_pil_images(
    pil_images: List[Image.Image],
    text_blocks: List[List[str]],
    *,
    font_size: int = 18,
    pad_x: int = 6,
    pad_y: int = 4,
    line_gap: int = 2,
) -> List[Image.Image]:
    if not pil_images:
        return []
    if len(pil_images) != len(text_blocks):
        raise ValueError("annotate_pil_images expects matching image/text lengths.")

    font = _load_font(font_size)
    measure_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    line_h = _line_height(measure_draw, font)
    max_lines = 1
    wrapped_blocks: List[List[str]] = []

    for pil, lines in zip(pil_images, text_blocks):
        draw = ImageDraw.Draw(pil)
        max_width = max(1, pil.size[0] - pad_x * 2)
        wrapped_lines: List[str] = []
        for base in lines:
            wrapped_lines.extend(_wrap_text(draw, base, font, max_width))
        if not wrapped_lines:
            wrapped_lines = [""]
        wrapped_blocks.append(wrapped_lines)
        max_lines = max(max_lines, len(wrapped_lines))

    text_h = pad_y * 2 + line_h * max_lines + line_gap * (max_lines - 1)
    annotated: List[Image.Image] = []
    for pil, wrapped_lines in zip(pil_images, wrapped_blocks):
        annotated_pil = Image.new("RGB", (pil.size[0], pil.size[1] + text_h), color=(0, 0, 0))
        annotated_pil.paste(pil, (0, text_h))
        draw = ImageDraw.Draw(annotated_pil)
        y = pad_y
        for line in wrapped_lines:
            draw.text((pad_x, y), line, fill=(255, 255, 255), font=font)
            y += line_h + line_gap

        annotated.append(annotated_pil)

    return annotated


@torch.no_grad()
def annotate_grid_images(
    images_norm: torch.Tensor,
    y_true: torch.Tensor,
    logits: torch.Tensor,
    id_to_country: Optional[List[str]],
    max_items: int = 16,
    topk: int = 3,
    font_size: int = 18,
) -> Tuple[torch.Tensor, str]:
    """
    Creates an annotated image grid and a text table summary.

    Adds:
      - T: true label name
      - P: predicted label name
      - p(T): probability assigned to the true class
      - top-k predictions with probabilities
    """
    bsz = images_norm.size(0)
    n = min(bsz, max_items)

    probs = torch.softmax(logits[:n], dim=1)
    topv, topi = torch.topk(probs, k=min(topk, probs.size(1)), dim=1)

    annotated = []
    lines = ["idx\ttrue\tpred\tp(true)\t(topk)"]
    pil_images = []
    text_blocks: List[List[str]] = []

    for i in range(n):
        img = denormalize(images_norm[i]).cpu()
        pil = transforms.ToPILImage()(img)

        yt = int(y_true[i].item())
        yp = int(topi[i, 0].item())

        true_name = id_to_country[yt] if id_to_country and yt < len(id_to_country) else str(yt)
        pred_name = id_to_country[yp] if id_to_country and yp < len(id_to_country) else str(yp)
        p_true = float(probs[i, yt].item()) if yt < probs.size(1) else 0.0

        topk_str = []
        for k in range(topi.size(1)):
            cid = int(topi[i, k].item())
            name = id_to_country[cid] if id_to_country and cid < len(id_to_country) else str(cid)
            topk_str.append(f"{name}:{float(topv[i, k].item()):.2f}")

        base_lines = [f"T:{true_name}  P:{pred_name}  p(T)={p_true:.2f}"]
        if topk_str:
            base_lines.append("  ".join(topk_str))

        pil_images.append(pil)
        text_blocks.append(base_lines)
        lines.append(f"{i}\t{true_name}\t{pred_name}\t{p_true:.3f}\t{' | '.join(topk_str)}")

    annotated_pil = annotate_pil_images(pil_images, text_blocks, font_size=font_size)
    for pil in annotated_pil:
        annotated.append(transforms.ToTensor()(pil))

    grid = make_grid(torch.stack(annotated, dim=0), nrow=4, padding=2)
    table = "\n".join(lines)
    return grid, table
