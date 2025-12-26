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
from bollards.data.bboxes import bbox_xyxy_norm_to_pixels, expand_bbox_xyxy_norm


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

            ex1, ey1, ex2, ey2 = expand_bbox_xyxy_norm(x1, y1, x2, y2, self.expand)
            px1, py1, px2, py2 = bbox_xyxy_norm_to_pixels(ex1, ey1, ex2, ey2, w, h)
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
