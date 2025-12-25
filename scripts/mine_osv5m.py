#!/usr/bin/env python3
from __future__ import annotations

import argparse

from bollards.config import MinerConfig, apply_overrides, load_config, resolve_config_path
from bollards.pipelines.osv5m_mine import run_miner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to miner config JSON")
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values, e.g. target=500",
    )
    args = ap.parse_args()

    cfg_path = resolve_config_path(args.config, "mine_osv5m.json")
    cfg = load_config(cfg_path, MinerConfig)
    cfg = apply_overrides(cfg, args.overrides)

    raise SystemExit(run_miner(cfg))


if __name__ == "__main__":
    main()
