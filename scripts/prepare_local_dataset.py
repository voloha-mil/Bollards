#!/usr/bin/env python3
from __future__ import annotations

import argparse

from bollards.config import PrepareLocalDatasetConfig, apply_overrides, load_config, resolve_config_path
from bollards.pipelines.local_dataset import run_prepare_local_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to dataset prep config JSON")
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values, e.g. num_boxes=2000",
    )
    args = ap.parse_args()

    cfg_path = resolve_config_path(args.config, "prepare_local_dataset.json")
    cfg = load_config(cfg_path, PrepareLocalDatasetConfig)
    cfg = apply_overrides(cfg, args.overrides)

    run_prepare_local_dataset(cfg)


if __name__ == "__main__":
    main()
