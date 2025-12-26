#!/usr/bin/env python3
from __future__ import annotations

import argparse

from bollards.config import AnalyzeRunConfig, apply_overrides, load_config, resolve_config_path
from bollards.pipelines.analyze_run import run_analyze_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to analysis config JSON")
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values, e.g. detector.enabled=true",
    )
    args = ap.parse_args()

    cfg_path = resolve_config_path(args.config, "analyze_run.json")
    cfg = load_config(cfg_path, AnalyzeRunConfig)
    cfg = apply_overrides(cfg, args.overrides)

    run_analyze_run(cfg)


if __name__ == "__main__":
    main()
