from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from bollards.models.bollard_net import ModelConfig


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
    sampler_alpha: float = 0.5
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
            sampler_alpha=float(data.get("sampler_alpha", 0.5)),
            expand=float(data.get("expand", 2.0)),
            max_train_samples=int(data.get("max_train_samples", 0)),
            max_val_samples=int(data.get("max_val_samples", 0)),
        )


@dataclass
class AugmentConfig:
    enabled: bool = True
    resize_pad: int = 32
    crop_scale_min: float = 0.9
    crop_scale_max: float = 1.0
    hflip_p: float = 0.5
    brightness: float = 0.15
    contrast: float = 0.15
    saturation: float = 0.10
    hue: float = 0.02
    blur_p: float = 0.15
    blur_kernel: int = 3
    affine_p: float = 0.5
    affine_degrees: float = 7.0
    affine_translate: float = 0.02
    affine_scale_min: float = 0.95
    affine_scale_max: float = 1.05

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AugmentConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            resize_pad=int(data.get("resize_pad", 32)),
            crop_scale_min=float(data.get("crop_scale_min", 0.9)),
            crop_scale_max=float(data.get("crop_scale_max", 1.0)),
            hflip_p=float(data.get("hflip_p", 0.5)),
            brightness=float(data.get("brightness", 0.15)),
            contrast=float(data.get("contrast", 0.15)),
            saturation=float(data.get("saturation", 0.10)),
            hue=float(data.get("hue", 0.02)),
            blur_p=float(data.get("blur_p", 0.15)),
            blur_kernel=int(data.get("blur_kernel", 3)),
            affine_p=float(data.get("affine_p", 0.5)),
            affine_degrees=float(data.get("affine_degrees", 7.0)),
            affine_translate=float(data.get("affine_translate", 0.02)),
            affine_scale_min=float(data.get("affine_scale_min", 0.95)),
            affine_scale_max=float(data.get("affine_scale_max", 1.05)),
        )


@dataclass
class OptimConfig:
    lr: float = 3e-4
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    conf_weight_min: float = 0.2
    class_weighting: bool = True
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
            class_weighting=bool(data.get("class_weighting", True)),
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
    best_metric: str = "val_top1"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoggingConfig":
        return cls(
            out_dir=str(data.get("out_dir", "runs/bollard_country")),
            tb_dir=data.get("tb_dir"),
            run_name=data.get("run_name"),
            log_images=int(data.get("log_images", 16)),
            log_image_every=int(data.get("log_image_every", 1)),
            tb_font_size=int(data.get("tb_font_size", 18)),
            best_metric=str(data.get("best_metric", "val_top1")),
        )


_DEFAULT_HUB_UPLOAD_INCLUDE = [
    "best.pt",
    "config.json",
    "country_map.json",
    "country_list.json",
    "country_counts.csv",
    "country_mapping.json",
]


def _default_hub_upload_include() -> list[str]:
    return list(_DEFAULT_HUB_UPLOAD_INCLUDE)


@dataclass
class HubConfig:
    enabled: bool = False
    repo_id: Optional[str] = None
    private: bool = False
    token: Optional[str] = None
    token_env: Optional[str] = "HF_TOKEN"
    path_in_repo: Optional[str] = None
    upload_include: list[str] = field(default_factory=_default_hub_upload_include)
    commit_message: str = "Add best checkpoint"
    fail_on_error: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HubConfig":
        def _clean(value: Any) -> Optional[str]:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        include = data.get("upload_include")
        if include is None:
            include = data.get("include")
        if include is None:
            include_list = _default_hub_upload_include()
        elif isinstance(include, list):
            include_list = [str(item).strip() for item in include if str(item).strip()]
        else:
            include_list = [str(include).strip()] if str(include).strip() else []

        commit_message = data.get("commit_message")
        if commit_message is None:
            commit_message = "Add best checkpoint"

        return cls(
            enabled=bool(data.get("enabled", False)),
            repo_id=_clean(data.get("repo_id")),
            private=bool(data.get("private", False)),
            token=_clean(data.get("token")),
            token_env=_clean(data.get("token_env", "HF_TOKEN")),
            path_in_repo=_clean(data.get("path_in_repo")),
            upload_include=include_list,
            commit_message=str(commit_message),
            fail_on_error=bool(data.get("fail_on_error", False)),
        )


@dataclass
class AnalyzeAfterTrainConfig:
    enabled: bool = False
    config_path: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalyzeAfterTrainConfig":
        return cls(
            enabled=bool(data.get("enabled", False)),
            config_path=data.get("config_path"),
        )


@dataclass
class TrainConfig:
    data: DataConfig
    model: "ModelConfig"
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    analyze: AnalyzeAfterTrainConfig = field(default_factory=AnalyzeAfterTrainConfig)
    device: str = "auto"
    seed: int = 42

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        from bollards.models.bollard_net import ModelConfig

        return cls(
            data=DataConfig.from_dict(data["data"]),
            model=ModelConfig(**data["model"]),
            augment=AugmentConfig.from_dict(data.get("augment", {})),
            optim=OptimConfig.from_dict(data.get("optim", {})),
            schedule=ScheduleConfig.from_dict(data.get("schedule", {})),
            logging=LoggingConfig.from_dict(data.get("logging", {})),
            hub=HubConfig.from_dict(data.get("hub", {})),
            analyze=AnalyzeAfterTrainConfig.from_dict(data.get("analyze", {})),
            device=str(data.get("device", "auto")),
            seed=int(data.get("seed", 42)),
        )
