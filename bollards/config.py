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
    val_num_workers: int = 0
    golden_num_workers: int = 0
    prefetch_factor: int = 1
    persistent_workers: bool = False
    balanced_sampler: bool = False
    expand: float = 2.0
    max_train_samples: int = 0
    max_val_samples: int = 0

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
            val_num_workers=int(data.get("val_num_workers", 0)),
            golden_num_workers=int(data.get("golden_num_workers", 0)),
            prefetch_factor=int(data.get("prefetch_factor", 1)),
            persistent_workers=bool(data.get("persistent_workers", False)),
            balanced_sampler=bool(data.get("balanced_sampler", False)),
            expand=float(data.get("expand", 2.0)),
            max_train_samples=int(data.get("max_train_samples", 0)),
            max_val_samples=int(data.get("max_val_samples", 0)),
        )


@dataclass
class OptimConfig:
    lr: float = 3e-4
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    conf_weight_min: float = 0.2
    focal_gamma: float = 0.0
    focal_alpha: Optional[float | list[float]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimConfig":
        focal_alpha = data.get("focal_alpha")
        if isinstance(focal_alpha, list):
            focal_alpha = [float(v) for v in focal_alpha]
        elif focal_alpha is not None:
            focal_alpha = float(focal_alpha)
        return cls(
            lr=float(data.get("lr", 3e-4)),
            backbone_lr=float(data.get("backbone_lr", 1e-4)),
            weight_decay=float(data.get("weight_decay", 1e-4)),
            label_smoothing=float(data.get("label_smoothing", 0.05)),
            conf_weight_min=float(data.get("conf_weight_min", 0.2)),
            focal_gamma=float(data.get("focal_gamma", 0.0)),
            focal_alpha=focal_alpha,
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


@dataclass
class LiveScreenCaptureConfig:
    monitor_index: int = 1
    region: Optional[dict[str, int]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LiveScreenCaptureConfig":
        region = data.get("region")
        if isinstance(region, dict):
            region = {k: int(v) for k, v in region.items()}
        else:
            region = None
        return cls(
            monitor_index=int(data.get("monitor_index", 1)),
            region=region,
        )


@dataclass
class LiveScreenDetectorConfig:
    weights_path: Optional[str] = None
    hf_repo: str = "maco018/YOLOv12_traffic-delineator"
    hf_filename: str = "models/YOLOv12_traffic-delineator.pt"
    hf_cache: str = "./hf_cache"
    imgsz: int = 960
    conf: float = 0.25

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LiveScreenDetectorConfig":
        try:
            from osv5m import HF_MODEL_FILENAME, HF_MODEL_REPO
        except Exception:
            HF_MODEL_REPO = cls.hf_repo
            HF_MODEL_FILENAME = cls.hf_filename

        return cls(
            weights_path=data.get("weights_path"),
            hf_repo=str(data.get("hf_repo", HF_MODEL_REPO)),
            hf_filename=str(data.get("hf_filename", HF_MODEL_FILENAME)),
            hf_cache=str(data.get("hf_cache", "./hf_cache")),
            imgsz=int(data.get("imgsz", 960)),
            conf=float(data.get("conf", 0.25)),
        )


@dataclass
class LiveScreenFiltersConfig:
    min_conf: float = 0.4
    cls_allow: Optional[list[float]] = None
    min_box_w_px: float = 8.0
    min_box_h_px: float = 16.0
    max_boxes_per_image: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LiveScreenFiltersConfig":
        return cls(
            min_conf=float(data.get("min_conf", 0.4)),
            cls_allow=data.get("cls_allow"),
            min_box_w_px=float(data.get("min_box_w_px", 8.0)),
            min_box_h_px=float(data.get("min_box_h_px", 16.0)),
            max_boxes_per_image=int(data.get("max_boxes_per_image", 1)),
        )


@dataclass
class LiveScreenClassifierConfig:
    checkpoint_path: str = ""
    country_map_json: Optional[str] = None
    img_size: int = 224
    expand: float = 2.0
    min_class_conf: float = 0.6
    topk: int = 3

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LiveScreenClassifierConfig":
        return cls(
            checkpoint_path=str(data.get("checkpoint_path", "")),
            country_map_json=data.get("country_map_json"),
            img_size=int(data.get("img_size", 224)),
            expand=float(data.get("expand", 2.0)),
            min_class_conf=float(data.get("min_class_conf", 0.6)),
            topk=int(data.get("topk", 3)),
        )


@dataclass
class LiveScreenTriggerConfig:
    mode: str = "hotkey"
    hotkey: str = "<ctrl>+<shift>+b"
    stdin_prompt: str = "Press Enter to capture, or type quit to exit."

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LiveScreenTriggerConfig":
        return cls(
            mode=str(data.get("mode", "hotkey")),
            hotkey=str(data.get("hotkey", "<ctrl>+<shift>+b")),
            stdin_prompt=str(data.get("stdin_prompt", "Press Enter to capture, or type quit to exit.")),
        )


@dataclass
class LiveScreenOutputConfig:
    out_dir: str = "runs/live_screen"
    run_name: Optional[str] = None
    save_screenshots: bool = True
    save_grid: bool = True
    grid_max_items: int = 24
    grid_cols: int = 6
    grid_thumb_size: int = 160
    viewer_enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LiveScreenOutputConfig":
        return cls(
            out_dir=str(data.get("out_dir", "runs/live_screen")),
            run_name=data.get("run_name"),
            save_screenshots=bool(data.get("save_screenshots", True)),
            save_grid=bool(data.get("save_grid", True)),
            grid_max_items=int(data.get("grid_max_items", 24)),
            grid_cols=int(data.get("grid_cols", 6)),
            grid_thumb_size=int(data.get("grid_thumb_size", 160)),
            viewer_enabled=bool(data.get("viewer_enabled", True)),
        )


@dataclass
class LiveScreenConfig:
    capture: LiveScreenCaptureConfig
    detector: LiveScreenDetectorConfig
    filters: LiveScreenFiltersConfig
    classifier: LiveScreenClassifierConfig
    output: LiveScreenOutputConfig
    trigger: LiveScreenTriggerConfig
    device: str = "auto"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LiveScreenConfig":
        return cls(
            capture=LiveScreenCaptureConfig.from_dict(data.get("capture", {})),
            detector=LiveScreenDetectorConfig.from_dict(data.get("detector", {})),
            filters=LiveScreenFiltersConfig.from_dict(data.get("filters", {})),
            classifier=LiveScreenClassifierConfig.from_dict(data.get("classifier", {})),
            output=LiveScreenOutputConfig.from_dict(data.get("output", {})),
            trigger=LiveScreenTriggerConfig.from_dict(data.get("trigger", {})),
            device=str(data.get("device", "auto")),
        )


@dataclass
class AnalyzeRunDataConfig:
    main_csv: str
    main_img_root: str
    golden_csv: Optional[str] = None
    golden_img_root: Optional[str] = None
    country_map_json: Optional[str] = None
    region_map_json: Optional[str] = None
    golden_default_category: str = "bollard"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalyzeRunDataConfig":
        return cls(
            main_csv=str(data.get("main_csv", "")),
            main_img_root=str(data.get("main_img_root", "")),
            golden_csv=data.get("golden_csv"),
            golden_img_root=data.get("golden_img_root"),
            country_map_json=data.get("country_map_json"),
            region_map_json=data.get("region_map_json"),
            golden_default_category=str(data.get("golden_default_category", "bollard")),
        )


@dataclass
class AnalyzeRunDetectorConfig:
    enabled: bool = False
    weights_path: Optional[str] = None
    hf_repo: str = "maco018/YOLOv12_traffic-delineator"
    hf_filename: str = "models/YOLOv12_traffic-delineator.pt"
    hf_cache: str = "./hf_cache"
    imgsz: int = 960
    conf: float = 0.25
    batch: int = 8
    max_images: int = 0
    cls_allow: Optional[list[float]] = None
    min_box_w_px: float = 0.0
    min_box_h_px: float = 0.0
    max_boxes_per_image: int = 50
    class_names: Optional[list[str]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalyzeRunDetectorConfig":
        return cls(
            enabled=bool(data.get("enabled", False)),
            weights_path=data.get("weights_path"),
            hf_repo=str(data.get("hf_repo", "maco018/YOLOv12_traffic-delineator")),
            hf_filename=str(data.get("hf_filename", "models/YOLOv12_traffic-delineator.pt")),
            hf_cache=str(data.get("hf_cache", "./hf_cache")),
            imgsz=int(data.get("imgsz", 960)),
            conf=float(data.get("conf", 0.25)),
            batch=int(data.get("batch", 8)),
            max_images=int(data.get("max_images", 0)),
            cls_allow=data.get("cls_allow"),
            min_box_w_px=float(data.get("min_box_w_px", 0.0)),
            min_box_h_px=float(data.get("min_box_h_px", 0.0)),
            max_boxes_per_image=int(data.get("max_boxes_per_image", 50)),
            class_names=data.get("class_names"),
        )


@dataclass
class AnalyzeRunClassifierConfig:
    checkpoint_path: str = ""
    img_size: int = 224
    expand: float = 2.0
    batch_size: int = 64
    num_workers: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalyzeRunClassifierConfig":
        return cls(
            checkpoint_path=str(data.get("checkpoint_path", "")),
            img_size=int(data.get("img_size", 224)),
            expand=float(data.get("expand", 2.0)),
            batch_size=int(data.get("batch_size", 64)),
            num_workers=int(data.get("num_workers", 0)),
        )


@dataclass
class AnalyzeRunOutputConfig:
    out_dir: str = "runs/analyze_run"
    run_name: Optional[str] = None
    gallery_size: int = 24
    gallery_per_category: int = 12
    top_k: int = 5
    min_support: int = 5
    max_categories_plots: int = 6

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalyzeRunOutputConfig":
        return cls(
            out_dir=str(data.get("out_dir", "runs/analyze_run")),
            run_name=data.get("run_name"),
            gallery_size=int(data.get("gallery_size", 24)),
            gallery_per_category=int(data.get("gallery_per_category", 12)),
            top_k=int(data.get("top_k", 5)),
            min_support=int(data.get("min_support", 5)),
            max_categories_plots=int(data.get("max_categories_plots", 6)),
        )


@dataclass
class AnalyzeRunConfig:
    data: AnalyzeRunDataConfig
    detector: AnalyzeRunDetectorConfig
    classifier: AnalyzeRunClassifierConfig
    output: AnalyzeRunOutputConfig
    device: str = "auto"
    seed: int = 42

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalyzeRunConfig":
        return cls(
            data=AnalyzeRunDataConfig.from_dict(data.get("data", {})),
            detector=AnalyzeRunDetectorConfig.from_dict(data.get("detector", {})),
            classifier=AnalyzeRunClassifierConfig.from_dict(data.get("classifier", {})),
            output=AnalyzeRunOutputConfig.from_dict(data.get("output", {})),
            device=str(data.get("device", "auto")),
            seed=int(data.get("seed", 42)),
        )


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_includes(data: Any, base_dir: Path) -> Any:
    if isinstance(data, dict):
        merged: dict[str, Any] = {}
        include = data.get("include")
        if include:
            include_list = include if isinstance(include, list) else [include]
            for inc in include_list:
                inc_path = Path(inc)
                if not inc_path.is_absolute():
                    inc_path = base_dir / inc_path
                with inc_path.open("r", encoding="utf-8") as f:
                    inc_data = json.load(f)
                inc_data = _resolve_includes(inc_data, inc_path.parent)
                if isinstance(inc_data, dict):
                    merged = _merge_dicts(merged, inc_data)
        for key, value in data.items():
            if key == "include":
                continue
            merged[key] = _resolve_includes(value, base_dir)
        return merged
    if isinstance(data, list):
        return [_resolve_includes(item, base_dir) for item in data]
    return data


def load_config(path: Path, cls: Type[T]) -> T:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data = _resolve_includes(data, path.parent)
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
