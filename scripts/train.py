#!/usr/bin/env python3
from __future__ import annotations

import argparse

from bollards.config import TrainConfig, apply_overrides, load_config, resolve_config_path
from bollards.train.runner import run_training


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to train config JSON")
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values, e.g. data.batch_size=128",
    )
    args = ap.parse_args()

    cfg_path = resolve_config_path(args.config, "train.json")
    cfg = load_config(cfg_path, TrainConfig)
    cfg = apply_overrides(cfg, args.overrides)

    run_training(cfg)


if __name__ == "__main__":
    main()
