from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AnalyzeRunDataConfig:
    main_csv: str
    main_img_root: str
    main_val_csv: Optional[str] = None
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
            main_val_csv=data.get("main_val_csv"),
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
