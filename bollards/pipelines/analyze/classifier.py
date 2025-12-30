from __future__ import annotations

import json
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bollards.config import AnalyzeRunConfig
from bollards.constants import PATH_COL
from bollards.data.datasets import BollardCropsDataset
from bollards.data.transforms import build_transforms
from bollards.models.bollard_net import BollardNet
from bollards.utils.seeding import make_torch_generator, seed_worker


class BollardCropsDatasetWithIndex(BollardCropsDataset):
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        item["index"] = torch.tensor(idx, dtype=torch.long)
        return item


def run_classifier(
    cfg: AnalyzeRunConfig,
    df: pd.DataFrame,
    img_root,
    id_to_country: Optional[list[str]],
    region_map: Optional[Dict[str, str]],
    model: BollardNet,
    device: torch.device,
) -> pd.DataFrame:
    tfm = build_transforms(train=False, img_size=cfg.classifier.img_size)
    ds = BollardCropsDatasetWithIndex(df, str(img_root), tfm, expand=cfg.classifier.expand)

    loader_kwargs = {
        "num_workers": cfg.classifier.num_workers,
        "pin_memory": device.type == "cuda",
        "generator": make_torch_generator(cfg.seed, "analyze_loader"),
    }
    if cfg.classifier.num_workers > 0:
        loader_kwargs["worker_init_fn"] = seed_worker

    loader = DataLoader(
        ds,
        batch_size=cfg.classifier.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    records = []
    for batch in tqdm(loader, desc="classify", leave=False, dynamic_ncols=True):
        images = batch["image"].to(device, non_blocking=True)
        meta = batch["meta"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        indices = batch["index"].cpu().numpy().tolist()

        with torch.no_grad():
            logits = model(images, meta)
            probs = torch.softmax(logits, dim=1)

        top5 = torch.topk(probs, k=min(5, probs.size(1)), dim=1).indices.cpu().numpy()
        top1 = probs.argmax(dim=1).cpu().numpy()
        top1_conf = probs.max(dim=1).values.cpu().numpy()

        for i, idx in enumerate(indices):
            row = df.iloc[int(idx)]
            true_id = int(labels[i].item())
            pred_id = int(top1[i])
            true_name = id_to_country[true_id] if id_to_country and true_id < len(id_to_country) else str(true_id)
            pred_name = id_to_country[pred_id] if id_to_country and pred_id < len(id_to_country) else str(pred_id)

            top5_ids = [int(v) for v in top5[i]]
            top5_names = [
                id_to_country[v] if id_to_country and v < len(id_to_country) else str(v)
                for v in top5_ids
            ]

            true_region = region_map.get(true_name) if region_map else None
            pred_region = region_map.get(pred_name) if region_map else None
            top5_regions = [region_map.get(name) for name in top5_names] if region_map else []

            correct_top1 = pred_id == true_id
            correct_top5 = true_id in top5_ids
            correct_region_top1 = bool(true_region and pred_region and true_region == pred_region)
            correct_region_top5 = bool(true_region and top5_regions and true_region in top5_regions)

            records.append({
                "image_path": str(row[PATH_COL]) if PATH_COL in row else "",
                "image_id": str(row.get("image_id", "")) if "image_id" in row else "",
                "x1": float(row.get("x1", 0.0)),
                "y1": float(row.get("y1", 0.0)),
                "x2": float(row.get("x2", 1.0)),
                "y2": float(row.get("y2", 1.0)),
                "country_id": true_id,
                "country": true_name,
                "pred_id": pred_id,
                "pred_country": pred_name,
                "top1_conf": float(top1_conf[i]),
                "top5_ids": json.dumps(top5_ids),
                "top5_countries": json.dumps(top5_names),
                "correct_top1": bool(correct_top1),
                "correct_top5": bool(correct_top5),
                "region": true_region,
                "pred_region": pred_region,
                "correct_region_top1": bool(correct_region_top1),
                "correct_region_top5": bool(correct_region_top5),
            })

    return pd.DataFrame(records)
