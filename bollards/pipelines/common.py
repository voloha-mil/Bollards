from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch

from bollards.constants import META_COLS
from bollards.detect.yolo import load_yolo
from bollards.io.hf import hf_download_model_file
from bollards.models.bollard_net import BollardNet, ModelConfig


def setup_logger(name: str, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def resolve_device(device_str: str) -> torch.device:
    if device_str and device_str != "auto":
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_detector(
    *,
    weights_path: Optional[str],
    hf_repo: str,
    hf_filename: str,
    hf_cache: str,
    logger: logging.Logger,
) -> object:
    if weights_path:
        weights = Path(weights_path)
    else:
        weights = hf_download_model_file(
            repo_id=hf_repo,
            filename=hf_filename,
            cache_dir=Path(hf_cache),
        )
    logger.info("Using detector weights: %s", weights)
    return load_yolo(weights)


def load_classifier(
    *,
    checkpoint_path: str,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[BollardNet, ModelConfig]:
    if not checkpoint_path:
        raise SystemExit("classifier.checkpoint_path is required")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg_dict = ckpt.get("cfg")
    if not model_cfg_dict:
        raise SystemExit("Classifier checkpoint missing cfg metadata")

    model_cfg = ModelConfig(**model_cfg_dict)
    model_cfg.pretrained = False
    if model_cfg.meta_dim != len(META_COLS):
        raise SystemExit(f"model.meta_dim must match META_COLS ({len(META_COLS)})")

    model = BollardNet(model_cfg).to(device)
    state = ckpt.get("model") or ckpt.get("state_dict")
    if not state:
        raise SystemExit("Classifier checkpoint missing model weights")
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded classifier: %s", checkpoint_path)
    return model, model_cfg
