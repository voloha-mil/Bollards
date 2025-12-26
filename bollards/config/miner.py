from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MinerConfig:
    split: str = "test"
    target: int = 1000
    batch: int = 16
    device: str = "auto"
    imgsz: int = 960
    conf: float = 0.25
    hf_cache: str = "./hf_cache"
    workdir: str = "./work_tmp"
    filtered: str = "./filtered_dataset_osv5m"
    meta: str = "test"
    s3_bucket: Optional[str] = None
    s3_prefix: str = ""
    s3_restore: bool = False
    s3_sync_state_every: int = 1
    s3_delete_local: bool = False
    worker_id: int = 0
    num_workers: int = 1
    max_retries: int = 2

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MinerConfig":
        return cls(
            split=str(data.get("split", "test")),
            target=int(data.get("target", 1000)),
            batch=int(data.get("batch", 16)),
            device=str(data.get("device", "auto")),
            imgsz=int(data.get("imgsz", 960)),
            conf=float(data.get("conf", 0.25)),
            hf_cache=str(data.get("hf_cache", "./hf_cache")),
            workdir=str(data.get("workdir", "./work_tmp")),
            filtered=str(data.get("filtered", "./filtered_dataset_osv5m")),
            meta=str(data.get("meta", "test")),
            s3_bucket=data.get("s3_bucket"),
            s3_prefix=str(data.get("s3_prefix", "")),
            s3_restore=bool(data.get("s3_restore", False)),
            s3_sync_state_every=int(data.get("s3_sync_state_every", 1)),
            s3_delete_local=bool(data.get("s3_delete_local", False)),
            worker_id=int(data.get("worker_id", 0)),
            num_workers=int(data.get("num_workers", 1)),
            max_retries=int(data.get("max_retries", 2)),
        )
