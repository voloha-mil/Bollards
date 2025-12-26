from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ImageDraw, ImageFont, ImageOps


@dataclass
class GridItem:
    path: Path
    label: str
    confidence: float


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


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 3)] + "..."


def render_grid(
    items: Iterable[GridItem],
    *,
    cols: int,
    thumb_size: int,
    max_items: int,
    font_size: int = 16,
    pad: int = 6,
) -> Optional[Image.Image]:
    items_list = list(items)
    if not items_list:
        return None

    max_items = max(1, max_items)
    items_list = items_list[-max_items:]
    cols = max(1, cols)
    rows = int(math.ceil(len(items_list) / cols))

    tile_w = thumb_size
    tile_h = thumb_size
    grid_w = cols * tile_w + (cols - 1) * pad
    grid_h = rows * tile_h + (rows - 1) * pad

    grid = Image.new("RGB", (grid_w, grid_h), color=(20, 20, 20))
    font = _load_font(font_size)

    for idx, item in enumerate(items_list):
        row = idx // cols
        col = idx % cols
        x = col * (tile_w + pad)
        y = row * (tile_h + pad)

        try:
            img = Image.open(item.path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (tile_w, tile_h), color=(50, 50, 50))

        img = ImageOps.fit(img, (tile_w, tile_h), method=Image.BICUBIC)

        label = _truncate(item.label, 18)
        text = f"{label} {item.confidence:.2f}"
        draw = ImageDraw.Draw(img)
        bg_h = int(font_size * 1.4)
        draw.rectangle((0, 0, tile_w, bg_h), fill=(0, 0, 0))
        draw.text((6, 2), text, fill=(255, 255, 255), font=font)

        grid.paste(img, (x, y))

    return grid


class GridViewer:
    def __init__(self, title: str = "Live Bollard Session") -> None:
        import tkinter as tk
        from PIL import ImageTk

        self._tk = tk
        self._ImageTk = ImageTk
        self.root = tk.Tk()
        self.root.title(title)
        self.label = tk.Label(self.root)
        self.label.pack()
        self._photo = None

    def update(self, img: Image.Image) -> None:
        photo = self._ImageTk.PhotoImage(img)
        self.label.configure(image=photo)
        self.label.image = photo
        self._photo = photo
        self.root.update_idletasks()
        self.root.update()

    def close(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass
