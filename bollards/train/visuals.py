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

    font = _load_font(font_size)
    annotated = []
    lines = ["idx\ttrue\tpred\tp(true)\t(topk)"]

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

        draw = ImageDraw.Draw(pil)
        text = f"T:{true_name}  P:{pred_name}  p(T)={p_true:.2f}\n" + "  ".join(topk_str)

        bg_h = int(font_size * 2.8)
        draw.rectangle((0, 0, pil.size[0], bg_h), fill=(0, 0, 0))
        draw.multiline_text((6, 4), text, fill=(255, 255, 255), font=font, spacing=2)

        annotated.append(transforms.ToTensor()(pil))
        lines.append(f"{i}\t{true_name}\t{pred_name}\t{p_true:.3f}\t{' | '.join(topk_str)}")

    grid = make_grid(torch.stack(annotated, dim=0), nrow=4, padding=2)
    table = "\n".join(lines)
    return grid, table
