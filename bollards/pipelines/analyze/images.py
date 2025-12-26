from __future__ import annotations

from typing import Any, Iterable

from PIL import Image, ImageDraw


def draw_boxes(img: Image.Image, boxes: list[dict[str, Any]], color_map: dict[int, tuple[int, int, int]]) -> Image.Image:
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


def make_color_map(ids: Iterable[int]) -> dict[int, tuple[int, int, int]]:
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
