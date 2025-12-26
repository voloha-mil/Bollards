from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PrepareLocalDatasetConfig:
    bucket: str = "geo-bollard-ml"
    root_prefix: str = "runs/osv5m_cpu"
    train_split: str = "train"
    val_split: str = "test"
    out_dir: str = "./local_data"
    num_boxes: int = 1000
    max_boxes_per_image: int = 1
    min_conf: float = 0.0
    cls_allow: Optional[list[float]] = None
    min_box_w_px: float = 0.0
    min_box_h_px: float = 0.0
    min_country_count: int = 0
    sample_strategy: str = "random"
    seed: int = 42
    download_annotated: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrepareLocalDatasetConfig":
        train_split = data.get("train_split")
        if not train_split:
            train_split = data.get("split", "train")
        return cls(
            bucket=str(data.get("bucket", "geo-bollard-ml")),
            root_prefix=str(data.get("root_prefix", "runs/osv5m_cpu")),
            train_split=str(train_split),
            val_split=str(data.get("val_split", "test")),
            out_dir=str(data.get("out_dir", "./local_data")),
            num_boxes=int(data.get("num_boxes", 1000)),
            max_boxes_per_image=int(data.get("max_boxes_per_image", 1)),
            min_conf=float(data.get("min_conf", 0.0)),
            cls_allow=data.get("cls_allow"),
            min_box_w_px=float(data.get("min_box_w_px", 0.0)),
            min_box_h_px=float(data.get("min_box_h_px", 0.0)),
            min_country_count=int(data.get("min_country_count", 0)),
            sample_strategy=str(data.get("sample_strategy", "random")),
            seed=int(data.get("seed", 42)),
            download_annotated=bool(data.get("download_annotated", False)),
        )
