from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from bollards.constants import BBOX_COLS, LABEL_COL, META_COLS, PATH_COL


class BollardCropsDataset(Dataset):
    """
    Loads original image, crops expanded bbox, applies transforms.
    """

    def __init__(self, df: pd.DataFrame, img_root: str, tfm: transforms.Compose, expand: float = 2.0):
        self.img_root = img_root
        self.tfm = tfm
        self.expand = expand

        required = [PATH_COL, LABEL_COL, *BBOX_COLS, *META_COLS]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")
        self.df = df[required].copy().reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        rel_path = str(row[PATH_COL])
        img_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.img_root, rel_path)

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w, h = img.size

            x1, y1, x2, y2 = [float(row[c]) for c in BBOX_COLS]

            # Expand bbox around center in normalized coords
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            bw = (x2 - x1) * self.expand
            bh = (y2 - y1) * self.expand

            ex1 = max(0.0, cx - 0.5 * bw)
            ey1 = max(0.0, cy - 0.5 * bh)
            ex2 = min(1.0, cx + 0.5 * bw)
            ey2 = min(1.0, cy + 0.5 * bh)

            px1 = int(round(ex1 * w))
            py1 = int(round(ey1 * h))
            px2 = int(round(ex2 * w))
            py2 = int(round(ey2 * h))
            if px2 <= px1:
                px2 = min(w, px1 + 1)
            if py2 <= py1:
                py2 = min(h, py1 + 1)

            crop = img.crop((px1, py1, px2, py2))

        crop = self.tfm(crop)

        label = int(row[LABEL_COL])
        meta = row[META_COLS].astype("float32").to_numpy()
        meta = np.clip(meta, 0.0, 1.0)

        return {
            "image": crop,
            "meta": torch.from_numpy(meta),
            "label": torch.tensor(label, dtype=torch.long),
        }
