from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


HF_DATASET_REPO = "osv5m/osv5m"
HF_MODEL_REPO = "maco018/YOLOv12_traffic-delineator"
HF_MODEL_FILENAME = "models/YOLOv12_traffic-delineator.pt"

META_FIELDS = [
    "latitude",
    "longitude",
    "country",
    "region",
    "sub_region",
    "city",
    "captured_at",
    "creator_username",
    "creator_id",
    "thumb_original_url",
]


@dataclass
class Meta:
    id: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    country: Optional[str] = None
    region: Optional[str] = None
    sub_region: Optional[str] = None
    city: Optional[str] = None
    thumb_original_url: Optional[str] = None
    captured_at: Optional[str] = None
    creator_username: Optional[str] = None
    creator_id: Optional[str] = None


@dataclass
class Cursor:
    shard_index: int = 0
    member_index: int = 0


@dataclass
class ShardCache:
    shard: Optional[str] = None
    zf: Optional[object] = None  # zipfile.ZipFile
    members: Optional[List[str]] = None


@dataclass
class LoadedBatch:
    shard: str
    extracted_paths: List[Path]
    cursor_before: Cursor


def shard_ids_for_split(split: str) -> List[str]:
    if split == "test":
        return [f"{i:02d}" for i in range(5)]
    if split == "train":
        return [f"{i:02d}" for i in range(98)]
    raise ValueError("split must be 'test' or 'train'")
