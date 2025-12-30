from __future__ import annotations

import json
from html import escape as _escape_html
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

from bollards.config import AnalyzeRunConfig
from bollards.constants import BBOX_COLS, LABEL_COL, META_COLS, PATH_COL
from bollards.data.bboxes import compute_avg_bbox_wh
from bollards.data.country_names import country_code_to_name, golden_country_to_code
from bollards.data.labels import load_id_to_country
from bollards.io.fs import ensure_dir
from bollards.pipelines.analyze.classifier import run_classifier as _run_classifier
from bollards.pipelines.analyze.detector import run_detector as _run_detector
from bollards.pipelines.analyze.gallery import save_gallery as _save_gallery
from bollards.pipelines.analyze.images import draw_boxes as _draw_boxes, make_color_map as _make_color_map
from bollards.pipelines.analyze.io import (
    build_report_section as _build_report_section,
    relative_paths as _relative_paths,
    render_table_grid as _render_table_grid,
    render_table as _render_table,
    save_csv as _save_csv,
    save_json as _save_json,
)
from bollards.pipelines.analyze.mappings import (
    build_country_mappings as _build_country_mappings,
    build_region_map,
    class_name as _class_name,
    maybe_add_region_by_country as _maybe_add_region_by_country,
    prepare_golden_df_for_classifier as _prepare_golden_df_for_classifier,
)
from bollards.pipelines.analyze.metrics import (
    calc_bbox_area_aspect as _calc_bbox_area_aspect,
    calc_crop_area_aspect as _calc_crop_area_aspect,
    compute_metrics as _compute_metrics,
    confusion_pairs as _confusion_pairs,
    dataset_summary as _dataset_summary,
    group_accuracy as _group_accuracy,
    image_counts_by_country as _image_counts_by_country,
    image_count_distribution as _image_count_distribution,
    top_bottom as _top_bottom,
    value_counts as _value_counts,
)
from bollards.pipelines.analyze.plots import plot_count_hist as _plot_count_hist
from bollards.pipelines.analyze.plots import plot_hist as _plot_hist
from bollards.pipelines.analyze.plots import plot_scalar_series as _plot_scalar_series
from bollards.pipelines.common import load_classifier, resolve_device, setup_logger as _setup_logger
from bollards.utils.seeding import make_python_rng, seed_everything


def setup_logger(log_path: Path):
    return _setup_logger("analyze_run", log_path)


def _compute_class_counts(
    df: pd.DataFrame,
    *,
    default_class: str,
    class_names: list[str] | None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"class_name": [], "count": []})

    if "cls" in df.columns:
        cls_series = pd.to_numeric(df["cls"], errors="coerce")
        if isinstance(cls_series, pd.DataFrame):
            cls_series = cls_series.iloc[:, 0]
        cls_series = cls_series.dropna()
        if not cls_series.empty:
            counts = cls_series.astype(int).value_counts().reset_index()
            counts.columns = ["class_id", "count"]
            counts["class_name"] = counts["class_id"].apply(lambda x: _class_name(x, class_names))
            return counts[["class_name", "count"]]

    if "class_name" in df.columns:
        counts = _value_counts(df, "class_name")
        if not counts.empty and "class_name" in counts.columns:
            return counts[["class_name", "count"]]

    if default_class:
        return pd.DataFrame({"class_name": [default_class], "count": [len(df)]})

    return pd.DataFrame({"class_name": [], "count": []})


def _build_region_map(cfg: AnalyzeRunConfig, golden_df: pd.DataFrame | None):
    return build_region_map(region_map_json=cfg.data.region_map_json, golden_df=golden_df)


def _load_classifier(cfg: AnalyzeRunConfig, device, logger):
    return load_classifier(
        checkpoint_path=cfg.classifier.checkpoint_path,
        device=device,
        logger=logger,
    )


def _format_gallery_labels(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    cols = [c for c in ["image_path", "image_id"] if c in df.columns]
    if not cols:
        return []
    data = df[cols].dropna(how="all").copy()
    if data.empty:
        return []
    data = data.drop_duplicates()
    labels: list[str] = []
    if "image_path" in data.columns and "image_id" in data.columns:
        for _, row in data.iterrows():
            path = str(row.get("image_path", "")).strip()
            img_id = str(row.get("image_id", "")).strip()
            if path and img_id:
                labels.append(f"{path} ({img_id})")
            elif path:
                labels.append(path)
            elif img_id:
                labels.append(img_id)
    elif "image_path" in data.columns:
        labels = [str(p).strip() for p in data["image_path"].tolist() if str(p).strip()]
    else:
        labels = [str(p).strip() for p in data["image_id"].tolist() if str(p).strip()]
    return labels


def _render_expand_list(title: str, items: list[str]) -> str:
    if not items:
        return ""
    escaped = _escape_html("\n".join(items))
    return (
        "<details class='expand'>"
        f"<summary>{_escape_html(title)}</summary>"
        f"<pre>{escaped}</pre>"
        "</details>"
    )


def _load_tb_scalars(tb_dir: Path, logger) -> dict[str, list[tuple[int, float]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        logger.info("TensorBoard not available; skipping scalar plots.")
        return {}

    if not tb_dir.exists():
        return {}

    try:
        accumulator = EventAccumulator(str(tb_dir))
        accumulator.Reload()
    except Exception as exc:
        logger.info("Failed to read TensorBoard logs from %s: %s", tb_dir, exc)
        return {}

    tags = accumulator.Tags().get("scalars", [])
    scalars: dict[str, list[tuple[int, float]]] = {}
    for tag in tags:
        events = accumulator.Scalars(tag)
        if not events:
            continue
        scalars[tag] = [(int(e.step), float(e.value)) for e in events]
    return scalars


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
        main_val_df = main_val_df.copy()
        if "country" not in main_val_df.columns and id_to_country is not None:
            main_val_df["country"] = main_val_df[LABEL_COL].apply(
                lambda idx: id_to_country[int(idx)] if int(idx) < len(id_to_country) else str(idx)
            )
        if region_map and "country" in main_val_df.columns:
            main_val_df = _maybe_add_region_by_country(main_val_df, "country", region_map)

    report_sections: list[str] = []

    # Dataset analyzer: main
    main_stats = {}
    if not main_df.empty:
        main_df = main_df.copy()
        if "country" not in main_df.columns and id_to_country is not None:
            main_df["country"] = main_df[LABEL_COL].apply(
                lambda idx: id_to_country[int(idx)] if int(idx) < len(id_to_country) else str(idx)
            )

        if region_map and "country" in main_df.columns:
            main_df = _maybe_add_region_by_country(main_df, "country", region_map)

        main_stats = _dataset_summary(main_df, "country", "region" if "region" in main_df.columns else None)
        _save_json(main_stats, run_dir / "artifacts" / "main" / "summary.json")
        main_counts = _value_counts(main_df, "country") if "country" in main_df.columns else pd.DataFrame()
        main_region_counts = _value_counts(main_df, "region") if "region" in main_df.columns else pd.DataFrame()
        main_image_counts = _image_counts_by_country(main_df, "country") if "country" in main_df.columns else pd.DataFrame()
        main_image_dist = _image_count_distribution(main_image_counts)

        _save_csv(main_counts, run_dir / "artifacts" / "main" / "country_counts.csv")
        _save_csv(main_region_counts, run_dir / "artifacts" / "main" / "region_counts.csv")
        _save_csv(main_image_counts, run_dir / "artifacts" / "main" / "images_per_country.csv")
        _save_csv(main_image_dist, run_dir / "artifacts" / "main" / "images_per_country_dist.csv")

        if not main_image_counts.empty:
            _plot_count_hist(
                main_image_counts["image_count"],
                run_dir / "artifacts" / "main" / "images_per_country_hist.png",
                "Images per country",
                "images per country",
            )

        if "cls" in main_df.columns:
            main_df["class_name"] = main_df["cls"].astype(int).apply(
                lambda x: _class_name(x, cfg.detector.class_names)
            )
        else:
            main_df["class_name"] = cfg.data.golden_default_category
        class_counts = _compute_class_counts(
            main_df,
            default_class=cfg.data.golden_default_category,
            class_names=cfg.detector.class_names,
        )
        _save_csv(class_counts, run_dir / "artifacts" / "main" / "class_counts.csv")

        area_aspect = _calc_bbox_area_aspect(main_df, "bbox")
        if not area_aspect.empty:
            _plot_hist(area_aspect["bbox_area"], run_dir / "artifacts" / "main" / "bbox_area_hist.png", "BBox area", "area fraction")
            _plot_hist(area_aspect["bbox_aspect"], run_dir / "artifacts" / "main" / "bbox_aspect_hist.png", "BBox aspect", "aspect ratio")

        top_country, bottom_country = _top_bottom(main_df, "country") if "country" in main_df.columns else (pd.DataFrame(), pd.DataFrame())

        main_section = ""
        main_section += f"<p>Images: {main_stats.get('n_images', 0)} | Objects: {main_stats.get('n_objects', 0)} | Objects/image: {main_stats.get('objects_per_image', 0):.2f}</p>"
        main_section += f"<p>Countries: {main_stats.get('n_countries', 0)}"
        if "n_regions" in main_stats:
            main_section += f" | Regions: {main_stats['n_regions']}"
        main_section += "</p>"
        main_section += "<h3>Top/bottom countries</h3>"
        main_section += _render_table_grid([
            ("Top countries", top_country),
            ("Bottom countries", bottom_country),
        ])
        if not main_region_counts.empty:
            main_section += "<h3>Regions distribution</h3>"
            main_section += _render_table(main_region_counts)
        if (run_dir / "artifacts" / "main" / "images_per_country_hist.png").exists():
            main_section += "<h3>Images per country</h3>"
            main_section += "<img src='artifacts/main/images_per_country_hist.png' width='420'>"
        main_section += "<h3>Geometry</h3>"
        if (run_dir / "artifacts" / "main" / "bbox_area_hist.png").exists():
            main_section += "<img src='artifacts/main/bbox_area_hist.png' width='420'>"
        if (run_dir / "artifacts" / "main" / "bbox_aspect_hist.png").exists():
            main_section += "<img src='artifacts/main/bbox_aspect_hist.png' width='420'>"
        report_sections.append(_build_report_section("Main dataset", main_section))

    # Dataset analyzer: golden
    if golden_df is not None and not golden_df.empty:
        golden_stats = {}
        golden_df = golden_df.copy()
        if "country" in golden_df.columns:
            golden_df["country_code"] = golden_df["country"].apply(golden_country_to_code)
            code_series = golden_df["country_code"].fillna("")
            canonical = code_series.map(country_code_to_name)
            canonical = canonical.where(canonical != code_series, None)
            golden_df["country"] = canonical.fillna(golden_df["country"])
            golden_df["country_code"] = code_series
        golden_df["class_name"] = cfg.data.golden_default_category
        golden_df = _maybe_add_region_by_country(golden_df, "country_code", region_map, out_col="region")
        if "region" not in golden_df.columns and "continent" in golden_df.columns:
            golden_df["region"] = golden_df["continent"]

        golden_stats = _dataset_summary(golden_df, "country", "region" if "region" in golden_df.columns else None)
        _save_json(golden_stats, run_dir / "artifacts" / "golden" / "summary.json")
        golden_country_counts = _value_counts(golden_df, "country")
        golden_region_counts = _value_counts(golden_df, "region") if "region" in golden_df.columns else pd.DataFrame()
        golden_image_counts = _image_counts_by_country(golden_df, "country")
        golden_image_dist = _image_count_distribution(golden_image_counts)
        _save_csv(golden_country_counts, run_dir / "artifacts" / "golden" / "country_counts.csv")
        _save_csv(golden_region_counts, run_dir / "artifacts" / "golden" / "region_counts.csv")
        _save_csv(golden_image_counts, run_dir / "artifacts" / "golden" / "images_per_country.csv")
        _save_csv(golden_image_dist, run_dir / "artifacts" / "golden" / "images_per_country_dist.csv")
        class_counts = _compute_class_counts(
            golden_df,
            default_class=cfg.data.golden_default_category,
            class_names=cfg.detector.class_names,
        )
        _save_csv(class_counts, run_dir / "artifacts" / "golden" / "class_counts.csv")

        golden_geom = _calc_crop_area_aspect(golden_df, "crop")
        if not golden_geom.empty:
            _plot_hist(golden_geom["crop_area"], run_dir / "artifacts" / "golden" / "bbox_area_hist.png", "BBox area", "area fraction")
            _plot_hist(golden_geom["crop_aspect"], run_dir / "artifacts" / "golden" / "bbox_aspect_hist.png", "BBox aspect", "aspect ratio")
        if not golden_image_counts.empty:
            _plot_count_hist(
                golden_image_counts["image_count"],
                run_dir / "artifacts" / "golden" / "images_per_country_hist.png",
                "Images per country",
                "images per country",
            )

        top_country, bottom_country = _top_bottom(golden_df, "country")

        golden_section = ""
        golden_section += f"<p>Images: {golden_stats.get('n_images', 0)} | Objects: {golden_stats.get('n_objects', 0)} | Objects/image: {golden_stats.get('objects_per_image', 0):.2f}</p>"
        golden_section += f"<p>Countries: {golden_stats.get('n_countries', 0)}"
        if "n_regions" in golden_stats:
            golden_section += f" | Regions: {golden_stats['n_regions']}"
        golden_section += "</p>"
        golden_section += "<h3>Top/bottom countries</h3>"
        golden_section += _render_table_grid([
            ("Top countries", top_country),
            ("Bottom countries", bottom_country),
        ])
        if not golden_region_counts.empty:
            golden_section += "<h3>Regions distribution</h3>"
            golden_section += _render_table(golden_region_counts)
        if (run_dir / "artifacts" / "golden" / "images_per_country_hist.png").exists():
            golden_section += "<h3>Images per country</h3>"
            golden_section += "<img src='artifacts/golden/images_per_country_hist.png' width='420'>"
        golden_section += "<h3>Geometry</h3>"
        if (run_dir / "artifacts" / "golden" / "bbox_area_hist.png").exists():
            golden_section += "<img src='artifacts/golden/bbox_area_hist.png' width='420'>"
        if (run_dir / "artifacts" / "golden" / "bbox_aspect_hist.png").exists():
            golden_section += "<img src='artifacts/golden/bbox_aspect_hist.png' width='420'>"
        report_sections.append(_build_report_section("Golden dataset", golden_section))

    # Detector prediction analyzer
    det_df = pd.DataFrame()
    if cfg.detector.enabled:
        logger.info("Running detector on main dataset")
        det_df = _run_detector(cfg, logger, main_df, Path(cfg.data.main_img_root), run_dir)
        if not det_df.empty:
            det_df["class_name"] = det_df["cls"].astype(int).apply(
                lambda x: _class_name(x, cfg.detector.class_names)
            )

        det_section = ""
        det_counts_path = run_dir / "artifacts" / "detector" / "detections_per_image.csv"
        det_conf_hist = run_dir / "artifacts" / "detector" / "conf_hist.png"
        det_class_counts = _value_counts(det_df, "class_name") if not det_df.empty else pd.DataFrame({"class_name": [], "count": []})
        if "class_name" not in det_class_counts.columns:
            det_class_counts = pd.DataFrame({"class_name": [], "count": []})
        _save_csv(det_class_counts, run_dir / "artifacts" / "detector" / "class_counts.csv")

        if det_counts_path.exists():
            det_counts = pd.read_csv(det_counts_path)["detections_per_image"].tolist()
            _plot_hist(det_counts, run_dir / "artifacts" / "detector" / "detections_hist.png", "Detections per image", "detections")

        if not det_df.empty:
            _plot_hist(det_df["conf"], det_conf_hist, "Detection confidence", "confidence")
            for cls_name in det_class_counts["class_name"].head(cfg.output.max_categories_plots).tolist():
                subset = det_df[det_df["class_name"] == cls_name]
                safe_name = cls_name.replace("/", "_")
                _plot_hist(
                    subset["conf"],
                    run_dir / "artifacts" / "detector" / f"conf_hist_{safe_name}.png",
                    f"Confidence ({cls_name})",
                    "confidence",
                )

        det_section += "<h3>Detections per image</h3>"
        if (run_dir / "artifacts" / "detector" / "detections_hist.png").exists():
            det_section += "<img src='artifacts/detector/detections_hist.png' width='420'>"
        det_section += "<h3>Confidence distribution</h3>"
        if det_conf_hist.exists():
            det_section += "<img src='artifacts/detector/conf_hist.png' width='420'>"
        if not det_class_counts.empty:
            det_section += "<h3>Confidence by class</h3>"
            for cls_name in det_class_counts["class_name"].head(cfg.output.max_categories_plots).tolist():
                safe_name = cls_name.replace("/", "_")
                conf_path = run_dir / "artifacts" / "detector" / f"conf_hist_{safe_name}.png"
                if conf_path.exists():
                    det_section += f"<img src='artifacts/detector/conf_hist_{safe_name}.png' width='420'>"

        if not det_class_counts.empty:
            det_section += "<h3>Class counts</h3>" + _render_table(det_class_counts.head(20))

        # Per-category galleries
        if not det_df.empty:
            img_root = Path(cfg.data.main_img_root)
            color_map = _make_color_map(det_df["cls"].astype(int).tolist())
            gallery_by_class: dict[str, list[str]] = {}
            gallery_labels_by_class: dict[str, list[str]] = {}
            for cls_name in det_class_counts["class_name"].head(cfg.output.max_categories_plots).tolist():
                cls_subset = det_df[det_df["class_name"] == cls_name]
                img_ids = cls_subset["image_path"].dropna().unique().tolist()
                rng.shuffle(img_ids)
                img_ids = img_ids[: cfg.output.gallery_per_category]
                gallery_dir = run_dir / "artifacts" / "detector" / "gallery" / cls_name.replace("/", "_")
                gallery_dir.mkdir(parents=True, exist_ok=True)
                gallery_by_class[cls_name] = []
                gallery_labels_by_class[cls_name] = []
                for i, rel_path in enumerate(img_ids):
                    img_path = img_root / rel_path
                    if not img_path.exists():
                        continue
                    try:
                        with Image.open(img_path) as im:
                            im = im.convert("RGB")
                            boxes = cls_subset[cls_subset["image_path"] == rel_path][
                                ["x1", "y1", "x2", "y2", "cls"]
                            ].to_dict("records")
                            annotated = _draw_boxes(im, boxes, color_map)
                            out_path = gallery_dir / f"det_{i:03d}.jpg"
                            annotated.save(out_path)
                            gallery_by_class[cls_name].append(str(out_path))
                            gallery_labels_by_class[cls_name].append(str(rel_path))
                    except Exception:
                        continue

            if gallery_by_class:
                det_section += "<h3>Sample detections by class</h3>"
                for cls_name, paths in gallery_by_class.items():
                    if not paths:
                        continue
                    rel = _relative_paths(paths, run_dir)
                    det_section += f"<h4>{cls_name}</h4><div class='grid'>"
                    det_section += "".join([f"<img src='{p}' loading='lazy'>" for p in rel])
                    det_section += "</div>"
                    labels = gallery_labels_by_class.get(cls_name, [])
                    det_section += _render_expand_list("Image files", labels)

        report_sections.append(_build_report_section("Detector predictions (main)", det_section))

    # Classifier evaluation
    if id_to_country is None or country_to_id is None:
        logger.info("Skipping classifier eval (missing country map)")
    else:
        class_sections = []
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
            golden_preds_path = run_dir / "artifacts" / "classifier" / "golden" / "predictions.csv"
            _save_csv(golden_preds, golden_preds_path)
            metrics = _compute_metrics(golden_preds)
            _save_json(metrics, run_dir / "artifacts" / "classifier" / "golden" / "metrics.json")

            country_groups = _group_accuracy(golden_preds, "country", cfg.output.min_support)
            region_groups = _group_accuracy(golden_preds, "region", cfg.output.min_support)
            _save_csv(country_groups, run_dir / "artifacts" / "classifier" / "golden" / "country_groups.csv")
            _save_csv(region_groups, run_dir / "artifacts" / "classifier" / "golden" / "region_groups.csv")

            country_conf = _confusion_pairs(golden_preds, "country", "pred_country", cfg.output.top_k)
            region_conf = _confusion_pairs(golden_preds, "region", "pred_region", cfg.output.top_k)
            _save_csv(country_conf, run_dir / "artifacts" / "classifier" / "golden" / "confusion_country.csv")
            _save_csv(region_conf, run_dir / "artifacts" / "classifier" / "golden" / "confusion_region.csv")

            correct = golden_preds[golden_preds["correct_top1"]].sample(
                n=min(cfg.output.gallery_size, len(golden_preds[golden_preds["correct_top1"]])),
                random_state=cfg.seed,
            ) if not golden_preds.empty else pd.DataFrame()
            incorrect = golden_preds[~golden_preds["correct_top1"]]
            incorrect_sample = incorrect.sample(
                n=min(cfg.output.gallery_size, len(incorrect)),
                random_state=cfg.seed,
            ) if not incorrect.empty else pd.DataFrame()
            high_conf_wrong = incorrect.sort_values("top1_conf", ascending=False).head(cfg.output.gallery_size)

            galleries = {}
            img_root = Path(cfg.data.golden_img_root or Path(cfg.data.golden_csv or ".").parent)
            galleries["correct"] = _save_gallery(
                correct,
                img_root,
                run_dir / "artifacts" / "classifier" / "golden" / "gallery_correct",
                "correct",
                cfg.classifier.expand,
                cfg.output.gallery_size,
                cfg.classifier.img_size,
            )
            galleries["incorrect"] = _save_gallery(
                incorrect_sample,
                img_root,
                run_dir / "artifacts" / "classifier" / "golden" / "gallery_incorrect",
                "incorrect",
                cfg.classifier.expand,
                cfg.output.gallery_size,
                cfg.classifier.img_size,
            )
            galleries["high_conf_wrong"] = _save_gallery(
                high_conf_wrong,
                img_root,
                run_dir / "artifacts" / "classifier" / "golden" / "gallery_high_conf_wrong",
                "highconf_wrong",
                cfg.classifier.expand,
                cfg.output.gallery_size,
                cfg.classifier.img_size,
            )

            section = ""
            section += "<p>Top-1 country: {:.3f} | Top-5 country: {:.3f}</p>".format(metrics.get("top1_country", 0.0), metrics.get("top5_country", 0.0))
            if "top1_region" in metrics:
                section += "<p>Top-1 region: {:.3f} | Top-5 region: {:.3f}</p>".format(metrics.get("top1_region", 0.0), metrics.get("top5_region", 0.0))
            section += "<h3>Best/worst countries</h3>"
            if not country_groups.empty:
                worst_countries = country_groups.sort_values("top1_accuracy", ascending=True).head(cfg.output.top_k)
                best_countries = country_groups.sort_values("top1_accuracy", ascending=False).head(cfg.output.top_k)
            else:
                worst_countries = country_groups
                best_countries = country_groups
            section += _render_table_grid([
                ("Worst", worst_countries),
                ("Best", best_countries),
            ])
            if not region_groups.empty:
                section += "<h3>Best/worst regions</h3>"
                worst_regions = region_groups.sort_values("top1_accuracy", ascending=True).head(cfg.output.top_k)
                best_regions = region_groups.sort_values("top1_accuracy", ascending=False).head(cfg.output.top_k)
                section += _render_table_grid([
                    ("Worst", worst_regions),
                    ("Best", best_regions),
                ])
            section += "<h3>Top confusion pairs (country)</h3>" + _render_table(country_conf)
            if not region_conf.empty:
                section += "<h3>Top confusion pairs (region)</h3>" + _render_table(region_conf)

            if galleries["correct"]:
                rel = _relative_paths(galleries["correct"], run_dir)
                section += "<h3>Correct samples</h3><div class='grid'>" + "".join([f"<img src='{p}' loading='lazy'>" for p in rel]) + "</div>"
                section += _render_expand_list("Image files", _format_gallery_labels(correct))
            if galleries["incorrect"]:
                rel = _relative_paths(galleries["incorrect"], run_dir)
                section += "<h3>Incorrect samples</h3><div class='grid'>" + "".join([f"<img src='{p}' loading='lazy'>" for p in rel]) + "</div>"
                section += _render_expand_list("Image files", _format_gallery_labels(incorrect_sample))
            if galleries["high_conf_wrong"]:
                rel = _relative_paths(galleries["high_conf_wrong"], run_dir)
                section += "<h3>High-confidence wrong</h3><div class='grid'>" + "".join([f"<img src='{p}' loading='lazy'>" for p in rel]) + "</div>"
                section += _render_expand_list("Image files", _format_gallery_labels(high_conf_wrong))

            class_sections.append(_build_report_section("Classifier (golden)", section))

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
                main_preds_path = run_dir / "artifacts" / "classifier" / "main" / "predictions.csv"
                _save_csv(main_preds, main_preds_path)
                metrics = _compute_metrics(main_preds)
                _save_json(metrics, run_dir / "artifacts" / "classifier" / "main" / "metrics.json")

                country_groups = _group_accuracy(main_preds, "country", cfg.output.min_support)
                region_groups = _group_accuracy(main_preds, "region", cfg.output.min_support)
                _save_csv(country_groups, run_dir / "artifacts" / "classifier" / "main" / "country_groups.csv")
                _save_csv(region_groups, run_dir / "artifacts" / "classifier" / "main" / "region_groups.csv")

                country_conf = _confusion_pairs(main_preds, "country", "pred_country", cfg.output.top_k)
                region_conf = _confusion_pairs(main_preds, "region", "pred_region", cfg.output.top_k)
                _save_csv(country_conf, run_dir / "artifacts" / "classifier" / "main" / "confusion_country.csv")
                _save_csv(region_conf, run_dir / "artifacts" / "classifier" / "main" / "confusion_region.csv")

                correct = main_preds[main_preds["correct_top1"]].sample(
                    n=min(cfg.output.gallery_size, len(main_preds[main_preds["correct_top1"]])),
                    random_state=cfg.seed,
                ) if not main_preds.empty else pd.DataFrame()
                incorrect = main_preds[~main_preds["correct_top1"]]
                incorrect_sample = incorrect.sample(
                    n=min(cfg.output.gallery_size, len(incorrect)),
                    random_state=cfg.seed,
                ) if not incorrect.empty else pd.DataFrame()
                high_conf_wrong = incorrect.sort_values("top1_conf", ascending=False).head(cfg.output.gallery_size)

                galleries = {}
                img_root = Path(cfg.data.main_img_root)
                galleries["correct"] = _save_gallery(
                    correct,
                    img_root,
                    run_dir / "artifacts" / "classifier" / "main" / "gallery_correct",
                    "correct",
                    cfg.classifier.expand,
                    cfg.output.gallery_size,
                    cfg.classifier.img_size,
                )
                galleries["incorrect"] = _save_gallery(
                    incorrect_sample,
                    img_root,
                    run_dir / "artifacts" / "classifier" / "main" / "gallery_incorrect",
                    "incorrect",
                    cfg.classifier.expand,
                    cfg.output.gallery_size,
                    cfg.classifier.img_size,
                )
                galleries["high_conf_wrong"] = _save_gallery(
                    high_conf_wrong,
                    img_root,
                    run_dir / "artifacts" / "classifier" / "main" / "gallery_high_conf_wrong",
                    "highconf_wrong",
                    cfg.classifier.expand,
                    cfg.output.gallery_size,
                    cfg.classifier.img_size,
                )

                section = ""
                if cfg.data.main_val_csv:
                    section += f"<p>Eval set: {_escape_html(cfg.data.main_val_csv)}</p>"
                section += "<p>Top-1 country: {:.3f} | Top-5 country: {:.3f}</p>".format(metrics.get("top1_country", 0.0), metrics.get("top5_country", 0.0))
                if "top1_region" in metrics:
                    section += "<p>Top-1 region: {:.3f} | Top-5 region: {:.3f}</p>".format(metrics.get("top1_region", 0.0), metrics.get("top5_region", 0.0))
                section += "<h3>Best/worst countries</h3>"
                if not country_groups.empty:
                    worst_countries = country_groups.sort_values("top1_accuracy", ascending=True).head(cfg.output.top_k)
                    best_countries = country_groups.sort_values("top1_accuracy", ascending=False).head(cfg.output.top_k)
                else:
                    worst_countries = country_groups
                    best_countries = country_groups
                section += _render_table_grid([
                    ("Worst", worst_countries),
                    ("Best", best_countries),
                ])
                if not region_groups.empty:
                    section += "<h3>Best/worst regions</h3>"
                    worst_regions = region_groups.sort_values("top1_accuracy", ascending=True).head(cfg.output.top_k)
                    best_regions = region_groups.sort_values("top1_accuracy", ascending=False).head(cfg.output.top_k)
                    section += _render_table_grid([
                        ("Worst", worst_regions),
                        ("Best", best_regions),
                    ])
                section += "<h3>Top confusion pairs (country)</h3>" + _render_table(country_conf)
                if not region_conf.empty:
                    section += "<h3>Top confusion pairs (region)</h3>" + _render_table(region_conf)

                if galleries["correct"]:
                    rel = _relative_paths(galleries["correct"], run_dir)
                    section += "<h3>Correct samples</h3><div class='grid'>" + "".join([f"<img src='{p}' loading='lazy'>" for p in rel]) + "</div>"
                    section += _render_expand_list("Image files", _format_gallery_labels(correct))
                if galleries["incorrect"]:
                    rel = _relative_paths(galleries["incorrect"], run_dir)
                    section += "<h3>Incorrect samples</h3><div class='grid'>" + "".join([f"<img src='{p}' loading='lazy'>" for p in rel]) + "</div>"
                    section += _render_expand_list("Image files", _format_gallery_labels(incorrect_sample))
                if galleries["high_conf_wrong"]:
                    rel = _relative_paths(galleries["high_conf_wrong"], run_dir)
                    section += "<h3>High-confidence wrong</h3><div class='grid'>" + "".join([f"<img src='{p}' loading='lazy'>" for p in rel]) + "</div>"
                    section += _render_expand_list("Image files", _format_gallery_labels(high_conf_wrong))

                class_sections.append(_build_report_section("Classifier (main)", section))

        report_sections.extend(class_sections)

    tb_section = ""
    tb_dir = None
    if training_run_dir:
        tb_dir = training_run_dir / "tb"
    elif cfg.classifier.checkpoint_path:
        tb_dir = Path(cfg.classifier.checkpoint_path).parent / "tb"

    if tb_dir:
        scalars = _load_tb_scalars(tb_dir, logger)
        if scalars:
            tb_section += f"<p class='meta'>Source: {_escape_html(str(tb_dir))}</p>"
            used_names: set[str] = set()
            for tag, points in sorted(scalars.items(), key=lambda item: item[0]):
                if not points:
                    continue
                steps, values = zip(*points)
                safe_tag = tag.replace("/", "_").replace(" ", "_")
                if safe_tag in used_names:
                    suffix = 2
                    while f"{safe_tag}_{suffix}" in used_names:
                        suffix += 1
                    safe_tag = f"{safe_tag}_{suffix}"
                used_names.add(safe_tag)
                out_path = run_dir / "artifacts" / "tensorboard" / f"{safe_tag}.png"
                _plot_scalar_series(steps, values, out_path, tag)
                if out_path.exists():
                    tb_section += f"<h3>{_escape_html(tag)}</h3>"
                    tb_section += f"<img src='artifacts/tensorboard/{safe_tag}.png' width='520'>"

    if tb_section:
        report_sections.append(_build_report_section("TensorBoard scalars", tb_section))

    css = """
    <style>
    :root {
      --bg: #f4f0e6;
      --bg-accent: #e7f0eb;
      --ink: #1f2b24;
      --muted: #5c6b5f;
      --card: #ffffff;
      --accent: #2c6f5f;
      --accent-2: #c89b3c;
      --border: #e2ddd2;
      --shadow: 0 14px 28px rgba(20, 26, 21, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Source Sans 3", "Noto Sans", sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at top left, #faf6ef 0%, var(--bg) 45%, var(--bg-accent) 100%);
    }
    .page { max-width: 1200px; margin: 0 auto; padding: 28px; }
    .page-header {
      padding: 24px 26px;
      border-radius: 18px;
      border: 1px solid var(--border);
      background: linear-gradient(135deg, #fef9f0 0%, #f0f7f4 100%);
      box-shadow: var(--shadow);
      margin-bottom: 26px;
    }
    .eyebrow { text-transform: uppercase; letter-spacing: 0.14em; font-size: 11px; color: var(--muted); margin: 0 0 8px; }
    h1 {
      font-family: "Fraunces", "Iowan Old Style", "Georgia", serif;
      font-size: 34px;
      margin: 0 0 8px;
      letter-spacing: -0.02em;
      color: var(--ink);
    }
    h2 {
      font-size: 22px;
      margin: 0 0 14px;
      color: var(--accent);
    }
    h3 { font-size: 18px; margin: 20px 0 10px; }
    h4 {
      margin: 4px 0 8px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }
    p { margin: 8px 0; color: var(--ink); }
    .meta { color: var(--muted); margin: 0; }
    .badge {
      display: inline-block;
      padding: 2px 10px;
      border-radius: 999px;
      background: rgba(44, 111, 95, 0.12);
      color: var(--accent);
      font-weight: 600;
      font-size: 12px;
    }
    section.section-card {
      background: var(--card);
      border-radius: 16px;
      padding: 20px 22px;
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }
    .table-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px;
      margin-bottom: 10px;
    }
    .table-card {
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      box-shadow: 0 8px 18px rgba(15, 20, 17, 0.06);
    }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 8px 10px; border-bottom: 1px solid var(--border); }
    th {
      text-align: left;
      background: #f7f4ea;
      font-weight: 600;
      color: #2d3b32;
    }
    tr:nth-child(even) { background: #fbfaf7; }
    details.expand {
      margin: 10px 0 16px;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px dashed var(--border);
      background: #fbf7ee;
    }
    details.expand summary {
      cursor: pointer;
      font-weight: 600;
      color: var(--accent);
    }
    details.expand pre {
      white-space: pre-wrap;
      margin: 8px 0 0;
      font-size: 12px;
      color: var(--muted);
    }
    img {
      margin: 8px 8px 8px 0;
      border-radius: 10px;
      border: 1px solid var(--border);
      box-shadow: 0 10px 20px rgba(18, 24, 20, 0.08);
      max-width: 100%;
      height: auto;
      background: #fff;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 10px;
    }
    .grid img { width: 100%; height: auto; display: block; margin: 0; }
    @media (max-width: 720px) {
      .page { padding: 18px; }
      .page-header { padding: 18px; }
      h1 { font-size: 28px; }
    }
    </style>
    """

    header = (
        "<header class='page-header'>"
        "<p class='eyebrow'>Bollards report</p>"
        "<h1>Single-run analysis</h1>"
        f"<p class='meta'>Run <span class='badge'>{run_dir.name}</span></p>"
        f"<p class='meta'>{region_note}</p>"
        "</header>"
    )
    body = "<div class='page'>" + header + "\n".join(report_sections) + "</div>"
    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"{css}</head><body>{body}</body></html>"
    )
    report_path = run_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")

    logger.info("Report saved: %s", report_path)
