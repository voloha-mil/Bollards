#!/usr/bin/env python3
from __future__ import annotations

import argparse

from bollards.config import LiveScreenConfig, apply_overrides, load_config, resolve_config_path
from bollards.pipelines.live_screen import run_live_screen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to live-screen config JSON")
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values, e.g. classifier.min_class_conf=0.7",
    )
    args = ap.parse_args()

    cfg_path = resolve_config_path(args.config, "live_screen.json")
    cfg = load_config(cfg_path, LiveScreenConfig)
    cfg = apply_overrides(cfg, args.overrides)

    run_live_screen(cfg)


if __name__ == "__main__":
    main()
