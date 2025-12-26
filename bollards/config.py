from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from bollards.models.bollard_net import ModelConfig

T = TypeVar("T")


@dataclass
class DataConfig:
    train_csv: str
    val_csv: str
    img_root: str
    country_map_json: Optional[str] = None
    golden_csv: Optional[str] = None
    golden_img_root: Optional[str] = None
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    balanced_sampler: bool = False
    expand: float = 2.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataConfig":
        return cls(
            train_csv=data["train_csv"],
            val_csv=data["val_csv"],
            img_root=data["img_root"],
            country_map_json=data.get("country_map_json"),
            golden_csv=data.get("golden_csv"),
            golden_img_root=data.get("golden_img_root"),
            img_size=int(data.get("img_size", 224)),
            batch_size=int(data.get("batch_size", 64)),
            num_workers=int(data.get("num_workers", 4)),
            balanced_sampler=bool(data.get("balanced_sampler", False)),
            expand=float(data.get("expand", 2.0)),
        )


@dataclass
class OptimConfig:
    lr: float = 3e-4
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    conf_weight_min: float = 0.2

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimConfig":
        return cls(
            lr=float(data.get("lr", 3e-4)),
            backbone_lr=float(data.get("backbone_lr", 1e-4)),
            weight_decay=float(data.get("weight_decay", 1e-4)),
            label_smoothing=float(data.get("label_smoothing", 0.05)),
            conf_weight_min=float(data.get("conf_weight_min", 0.2)),
        )


@dataclass
class ScheduleConfig:
    epochs: int = 15
    freeze_epochs: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduleConfig":
        return cls(
            epochs=int(data.get("epochs", 15)),
            freeze_epochs=int(data.get("freeze_epochs", 1)),
        )


@dataclass
class LoggingConfig:
    out_dir: str = "runs/bollard_country"
    tb_dir: Optional[str] = None
    run_name: Optional[str] = None
    log_images: int = 16
    log_image_every: int = 1
    tb_font_size: int = 18

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoggingConfig":
        return cls(
            out_dir=str(data.get("out_dir", "runs/bollard_country")),
            tb_dir=data.get("tb_dir"),
            run_name=data.get("run_name"),
            log_images=int(data.get("log_images", 16)),
            log_image_every=int(data.get("log_image_every", 1)),
            tb_font_size=int(data.get("tb_font_size", 18)),
        )


@dataclass
class TrainConfig:
    data: DataConfig
    model: "ModelConfig"
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    device: str = "auto"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        from bollards.models.bollard_net import ModelConfig

        return cls(
            data=DataConfig.from_dict(data["data"]),
            model=ModelConfig(**data["model"]),
            optim=OptimConfig.from_dict(data.get("optim", {})),
            schedule=ScheduleConfig.from_dict(data.get("schedule", {})),
            logging=LoggingConfig.from_dict(data.get("logging", {})),
            device=str(data.get("device", "auto")),
        )


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


@dataclass
class PrepareLocalDatasetConfig:
    bucket: str = "geo-bollard-ml"
    root_prefix: str = "runs/osv5m_cpu"
    split: str = "test"
    out_dir: str = "./local_data"
    num_boxes: int = 1000
    max_boxes_per_image: int = 1
    min_conf: float = 0.0
    cls_allow: Optional[list[float]] = None
    min_box_w_px: float = 0.0
    min_box_h_px: float = 0.0
    sample_strategy: str = "random"
    val_ratio: float = 0.1
    seed: int = 42
    download_annotated: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrepareLocalDatasetConfig":
        return cls(
            bucket=str(data.get("bucket", "geo-bollard-ml")),
            root_prefix=str(data.get("root_prefix", "runs/osv5m_cpu")),
            split=str(data.get("split", "test")),
            out_dir=str(data.get("out_dir", "./local_data")),
            num_boxes=int(data.get("num_boxes", 1000)),
            max_boxes_per_image=int(data.get("max_boxes_per_image", 1)),
            min_conf=float(data.get("min_conf", 0.0)),
            cls_allow=data.get("cls_allow"),
            min_box_w_px=float(data.get("min_box_w_px", 0.0)),
            min_box_h_px=float(data.get("min_box_h_px", 0.0)),
            sample_strategy=str(data.get("sample_strategy", "random")),
            val_ratio=float(data.get("val_ratio", 0.1)),
            seed=int(data.get("seed", 42)),
            download_annotated=bool(data.get("download_annotated", False)),
        )


def load_config(path: Path, cls: Type[T]) -> T:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return cls.from_dict(data)  # type: ignore[attr-defined]


def resolve_config_path(config_path: Optional[str], default_name: str) -> Path:
    if config_path:
        return Path(config_path)
    default_path = Path("configs") / default_name
    if default_path.exists():
        return default_path
    raise SystemExit(f"Missing config. Provide --config or create {default_path}.")


def _parse_override_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        lowered = raw.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        return raw


def _coerce_value(value: Any, current: Any) -> Any:
    if current is None:
        return value
    if isinstance(current, bool):
        return bool(value)
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value.strip():
            return [v.strip() for v in value.split(",") if v.strip()]
        return value
    if isinstance(current, str):
        return str(value)
    return value


def apply_overrides(cfg: T, overrides: list[str]) -> T:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item}. Use key=value")
        key, raw = item.split("=", 1)
        value = _parse_override_value(raw)

        target: Any = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise KeyError(f"Unknown config key: {key}")
            target = getattr(target, part)

        attr = parts[-1]
        if not hasattr(target, attr):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(target, attr)
        setattr(target, attr, _coerce_value(value, current))

    return cfg
