from __future__ import annotations

import argparse
from typing import Callable, Optional, Sequence

from bollards.pipelines.analyze.config import AnalyzeRunConfig
from bollards.pipelines.live_screen.config import LiveScreenConfig
from bollards.pipelines.local_dataset.config import PrepareLocalDatasetConfig
from bollards.pipelines.osv5m.config import MinerConfig
from bollards.pipelines.train.config import TrainConfig
from bollards.utils.config import apply_overrides, load_config, resolve_config_path
from bollards.pipelines.analyze.run import run_analyze_run
from bollards.pipelines.live_screen.run import run_live_screen
from bollards.pipelines.local_dataset.run import run_prepare_local_dataset
from bollards.pipelines.osv5m.run import run_miner
from bollards.pipelines.train.run import run_training
from bollards.utils.env import load_env


def _add_config_args(parser: argparse.ArgumentParser, config_help: str) -> None:
    parser.add_argument("--config", default=None, help=config_help)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values, e.g. data.batch_size=128",
    )


def _run_train(args: argparse.Namespace) -> int:
    cfg_path = resolve_config_path(args.config, "train.json")
    cfg = load_config(cfg_path, TrainConfig)
    cfg = apply_overrides(cfg, args.overrides)
    run_training(cfg)
    return 0


def _run_mine(args: argparse.Namespace) -> int:
    cfg_path = resolve_config_path(args.config, "mine_osv5m.json")
    cfg = load_config(cfg_path, MinerConfig)
    cfg = apply_overrides(cfg, args.overrides)
    return int(run_miner(cfg))


def _run_prepare_local(args: argparse.Namespace) -> int:
    cfg_path = resolve_config_path(args.config, "prepare_local_dataset.json")
    cfg = load_config(cfg_path, PrepareLocalDatasetConfig)
    cfg = apply_overrides(cfg, args.overrides)
    run_prepare_local_dataset(cfg)
    return 0


def _run_live_screen(args: argparse.Namespace) -> int:
    cfg_path = resolve_config_path(args.config, "live_screen.json")
    cfg = load_config(cfg_path, LiveScreenConfig)
    cfg = apply_overrides(cfg, args.overrides)
    run_live_screen(cfg)
    return 0


def _run_analyze(args: argparse.Namespace) -> int:
    cfg_path = resolve_config_path(args.config, "analyze_run.json")
    cfg = load_config(cfg_path, AnalyzeRunConfig)
    cfg = apply_overrides(cfg, args.overrides)
    run_analyze_run(cfg)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_env()
    parser = argparse.ArgumentParser(prog="bollards")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the classifier")
    _add_config_args(train_parser, "Path to train config JSON")
    train_parser.set_defaults(_run=_run_train)

    mine_parser = subparsers.add_parser("mine-osv5m", help="Mine OSV5M detections")
    _add_config_args(mine_parser, "Path to miner config JSON")
    mine_parser.set_defaults(_run=_run_mine)

    prep_parser = subparsers.add_parser("prepare-local", help="Prepare local dataset from S3")
    _add_config_args(prep_parser, "Path to dataset prep config JSON")
    prep_parser.set_defaults(_run=_run_prepare_local)

    live_parser = subparsers.add_parser("live-screen", help="Run live screen classification")
    _add_config_args(live_parser, "Path to live screen config JSON")
    live_parser.set_defaults(_run=_run_live_screen)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a run")
    _add_config_args(analyze_parser, "Path to analysis config JSON")
    analyze_parser.set_defaults(_run=_run_analyze)

    args = parser.parse_args(argv)
    run: Callable[[argparse.Namespace], int] = args._run
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
