from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from bollards.pipelines.analyze.config import AnalyzeRunConfig
from bollards.constants import BBOX_COLS, LABEL_COL, META_COLS, PATH_COL
from bollards.data.bboxes import compute_avg_bbox_wh
from bollards.data.country_names import country_code_to_name, golden_country_to_code
from bollards.data.labels import load_id_to_country
from bollards.utils.io.fs import ensure_dir
from bollards.pipelines.analyze.eval import run_classifier as _run_classifier
from bollards.pipelines.analyze.eval import run_detector as _run_detector
from bollards.pipelines.analyze.mappings import (
    build_country_mappings as _build_country_mappings,
    build_region_map,
    maybe_add_region_by_country as _maybe_add_region_by_country,
    prepare_golden_df_for_classifier as _prepare_golden_df_for_classifier,
)
from bollards.pipelines.analyze.report import build_report_html
from bollards.pipelines.analyze.sections import (
    build_classifier_section,
    build_detector_section,
    build_golden_dataset_section,
    build_main_dataset_section,
    build_tensorboard_section,
)
from bollards.models.loaders import load_classifier
from bollards.utils.runtime import resolve_device, setup_logger as _setup_logger
from bollards.utils.seeding import make_python_rng, seed_everything


def setup_logger(log_path: Path):
    return _setup_logger("analyze_run", log_path)


def _build_region_map(cfg: AnalyzeRunConfig, golden_df: pd.DataFrame | None):
    return build_region_map(region_map_json=cfg.data.region_map_json, golden_df=golden_df)


def _load_classifier(cfg: AnalyzeRunConfig, device, logger):
    return load_classifier(
        checkpoint_path=cfg.classifier.checkpoint_path,
        device=device,
        logger=logger,
    )


def _add_country_and_region(
    df: pd.DataFrame,
    *,
    id_to_country: list[str] | None,
    region_map: dict[str, str] | None,
) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "country" not in df.columns and id_to_country is not None and LABEL_COL in df.columns:
        df["country"] = df[LABEL_COL].apply(
            lambda idx: id_to_country[int(idx)] if int(idx) < len(id_to_country) else str(idx)
        )
    if region_map and "country" in df.columns:
        df = _maybe_add_region_by_country(df, "country", region_map)
    return df


def _normalize_golden_df(
    df: pd.DataFrame,
    *,
    region_map: dict[str, str] | None,
) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "country" in df.columns:
        df["country_code"] = df["country"].apply(golden_country_to_code)
        code_series = df["country_code"].fillna("")
        canonical = code_series.map(country_code_to_name)
        canonical = canonical.where(canonical != code_series, None)
        df["country"] = canonical.fillna(df["country"])
        df["country_code"] = code_series
    df = _maybe_add_region_by_country(df, "country_code", region_map, out_col="region")
    if "region" not in df.columns and "continent" in df.columns:
        df["region"] = df["continent"]
    return df


def run_analyze_run(cfg: AnalyzeRunConfig) -> None:
    seed_everything(cfg.seed)
    rng = make_python_rng(cfg.seed, "analyze_shuffle")

    out_dir = Path(cfg.output.out_dir)
    ensure_dir(out_dir)

    run_name = cfg.output.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = out_dir / run_name
    if run_dir.exists():
        suffix = 1
        while (out_dir / f"{run_name}_{suffix:02d}").exists():
            suffix += 1
        run_dir = out_dir / f"{run_name}_{suffix:02d}"
    ensure_dir(run_dir)

    log_path = run_dir / "analysis.log"
    logger = setup_logger(log_path)

    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = resolve_device(cfg.device)
    logger.info("Using device: %s", device)

    main_df = pd.read_csv(cfg.data.main_csv)
    golden_df = pd.read_csv(cfg.data.golden_csv) if cfg.data.golden_csv else None
    main_val_df = pd.read_csv(cfg.data.main_val_csv) if cfg.data.main_val_csv else pd.DataFrame()

    training_run_dir = Path(cfg.data.training_run_dir) if cfg.data.training_run_dir else None
    if training_run_dir:
        cfg.classifier.checkpoint_path = str(training_run_dir / "best.pt")
        cfg.data.country_map_json = str(training_run_dir / "country_map.json")
        logger.info("Using training run dir: %s", training_run_dir)
        logger.info("Classifier checkpoint: %s", cfg.classifier.checkpoint_path)
        logger.info("Country map: %s", cfg.data.country_map_json)

    id_to_country = load_id_to_country(cfg.data.country_map_json)
    id_to_country, country_to_id = _build_country_mappings(id_to_country, main_df)

    region_map = _build_region_map(cfg, golden_df)
    if cfg.data.region_map_json:
        region_note = f"Region mapping: {cfg.data.region_map_json}"
    elif region_map is not None:
        region_note = "Region mapping: derived from golden dataset continent"
    else:
        region_note = "Region mapping: unavailable (region metrics skipped)"

    if not main_val_df.empty:
        main_val_df = _add_country_and_region(
            main_val_df,
            id_to_country=id_to_country,
            region_map=region_map,
        )

    report_sections: list[str] = []

    # Dataset analyzer: main
    if not main_df.empty:
        main_df = _add_country_and_region(
            main_df,
            id_to_country=id_to_country,
            region_map=region_map,
        )
        main_section = build_main_dataset_section(
            main_df,
            run_dir=run_dir,
            default_class=cfg.data.golden_default_category,
            class_names=cfg.detector.class_names,
        )
        if main_section:
            report_sections.append(main_section)

    # Dataset analyzer: golden
    if golden_df is not None and not golden_df.empty:
        golden_df = _normalize_golden_df(golden_df, region_map=region_map)
        golden_section = build_golden_dataset_section(
            golden_df,
            run_dir=run_dir,
            default_class=cfg.data.golden_default_category,
            class_names=cfg.detector.class_names,
        )
        if golden_section:
            report_sections.append(golden_section)

    # Detector prediction analyzer
    det_df = pd.DataFrame()
    if cfg.detector.enabled:
        logger.info("Running detector on main dataset")
        det_df = _run_detector(cfg, logger, main_df, Path(cfg.data.main_img_root), run_dir)
        det_section = build_detector_section(
            det_df,
            cfg=cfg,
            rng=rng,
            run_dir=run_dir,
            img_root=Path(cfg.data.main_img_root),
        )
        report_sections.append(det_section)

    # Classifier evaluation
    if id_to_country is None or country_to_id is None:
        logger.info("Skipping classifier eval (missing country map)")
    else:
        class_sections: list[str] = []
        classifier_model, _ = _load_classifier(cfg, device, logger)
        # Golden classifier eval
        if golden_df is not None and not golden_df.empty:
            avg_bbox_w, avg_bbox_h = compute_avg_bbox_wh(main_df, label="Main")
            golden_eval_df = _prepare_golden_df_for_classifier(
                golden_df,
                country_to_id,
                avg_bbox_w=avg_bbox_w,
                avg_bbox_h=avg_bbox_h,
            )
            golden_preds = _run_classifier(
                cfg,
                golden_eval_df,
                Path(cfg.data.golden_img_root or Path(cfg.data.golden_csv or ".").parent),
                id_to_country,
                region_map,
                classifier_model,
                device,
            )
            img_root = Path(cfg.data.golden_img_root or Path(cfg.data.golden_csv or ".").parent)
            class_sections.append(build_classifier_section(
                golden_preds,
                cfg=cfg,
                run_dir=run_dir,
                dataset_key="golden",
                title="Classifier (golden)",
                img_root=img_root,
            ))

        # Main classifier eval (val only)
        if main_val_df.empty:
            logger.info("Skipping classifier eval on main dataset (no main_val_csv provided).")
        else:
            eval_df = main_val_df.copy()
            if cfg.detector.enabled and not det_df.empty:
                det_eval_df = det_df.copy()
                val_paths = eval_df[PATH_COL].dropna().astype(str).unique().tolist()
                if val_paths:
                    det_eval_df = det_eval_df[det_eval_df["image_path"].isin(val_paths)]
                if "country" in eval_df.columns:
                    image_country = eval_df.groupby(PATH_COL)["country"].first().to_dict()
                    image_country_id = eval_df.groupby(PATH_COL)[LABEL_COL].first().to_dict() if LABEL_COL in eval_df.columns else {}
                else:
                    image_country = {}
                    image_country_id = {}

                det_eval_df["country"] = det_eval_df["image_path"].map(image_country)
                det_eval_df[LABEL_COL] = det_eval_df["image_path"].map(image_country_id)
                det_eval_df = det_eval_df.dropna(subset=[LABEL_COL])
                det_eval_df[LABEL_COL] = det_eval_df[LABEL_COL].astype(int)
                eval_df = det_eval_df

            if all(c in eval_df.columns for c in [PATH_COL, LABEL_COL, *BBOX_COLS, *META_COLS]):
                main_preds = _run_classifier(
                    cfg,
                    eval_df,
                    Path(cfg.data.main_img_root),
                    id_to_country,
                    region_map,
                    classifier_model,
                    device,
                )
                class_sections.append(build_classifier_section(
                    main_preds,
                    cfg=cfg,
                    run_dir=run_dir,
                    dataset_key="main",
                    title="Classifier (main)",
                    img_root=Path(cfg.data.main_img_root),
                    eval_label=cfg.data.main_val_csv,
                ))

        report_sections.extend(class_sections)

    tb_dir = None
    if training_run_dir:
        tb_dir = training_run_dir / "tb"
    elif cfg.classifier.checkpoint_path:
        tb_dir = Path(cfg.classifier.checkpoint_path).parent / "tb"

    if tb_dir:
        tb_section = build_tensorboard_section(
            tb_dir,
            run_dir=run_dir,
            logger=logger,
        )
        if tb_section:
            report_sections.append(tb_section)

    html = build_report_html(run_dir.name, region_note, report_sections)
    report_path = run_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")

    logger.info("Report saved: %s", report_path)
