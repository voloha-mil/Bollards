from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


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
            from bollards.osv5m import HF_MODEL_FILENAME, HF_MODEL_REPO
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
