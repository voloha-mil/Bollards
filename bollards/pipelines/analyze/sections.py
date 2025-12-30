from __future__ import annotations

import logging
from html import escape as escape_html
from pathlib import Path
from random import Random
from typing import Callable

import pandas as pd

from bollards.pipelines.analyze.config import AnalyzeRunConfig
from bollards.pipelines.analyze.gallery import save_detection_galleries, save_gallery
from bollards.utils.visuals.boxes import make_color_map
from bollards.pipelines.analyze.reporting import (
    build_report_section,
    relative_paths,
    render_table,
    render_table_grid,
    save_csv,
    save_json,
)
from bollards.pipelines.analyze.mappings import class_name
from bollards.pipelines.analyze.metrics import (
    calc_bbox_area_aspect,
    calc_crop_area_aspect,
    confusion_pairs,
    compute_metrics,
    dataset_summary,
    group_accuracy,
    image_count_distribution,
    image_counts_by_country,
    top_bottom,
    value_counts,
)
from bollards.utils.visuals.plots import plot_count_hist, plot_hist, plot_scalar_series
from bollards.pipelines.analyze.report import format_gallery_labels, render_expand_list


def compute_class_counts(
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
            counts["class_name"] = counts["class_id"].apply(lambda x: class_name(x, class_names))
            return counts[["class_name", "count"]]

    if "class_name" in df.columns:
        counts = value_counts(df, "class_name")
        if not counts.empty and "class_name" in counts.columns:
            return counts[["class_name", "count"]]

    if default_class:
        return pd.DataFrame({"class_name": [default_class], "count": [len(df)]})

    return pd.DataFrame({"class_name": [], "count": []})


def _plot_geometry(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    calc_fn: Callable[[pd.DataFrame, str], pd.DataFrame],
    column_prefix: str,
    output_prefix: str,
) -> None:
    geom = calc_fn(df, column_prefix)
    if geom.empty:
        return
    area_col = f"{column_prefix}_area"
    aspect_col = f"{column_prefix}_aspect"
    if area_col in geom.columns:
        plot_hist(
            geom[area_col],
            out_dir / f"{output_prefix}_area_hist.png",
            "BBox area",
            "area fraction",
        )
    if aspect_col in geom.columns:
        plot_hist(
            geom[aspect_col],
            out_dir / f"{output_prefix}_aspect_hist.png",
            "BBox aspect",
            "aspect ratio",
        )


def build_dataset_section(
    df: pd.DataFrame,
    *,
    run_dir: Path,
    dataset_key: str,
    title: str,
    default_class: str,
    class_names: list[str] | None,
    geom_calc: Callable[[pd.DataFrame, str], pd.DataFrame],
    geom_prefix: str,
    geom_output_prefix: str,
) -> str:
    if df.empty:
        return ""

    out_dir = run_dir / "artifacts" / dataset_key
    stats = dataset_summary(df, "country", "region" if "region" in df.columns else None)
    save_json(stats, out_dir / "summary.json")

    country_counts = value_counts(df, "country")
    region_counts = value_counts(df, "region")
    image_counts = image_counts_by_country(df, "country")
    image_dist = image_count_distribution(image_counts)

    save_csv(country_counts, out_dir / "country_counts.csv")
    save_csv(region_counts, out_dir / "region_counts.csv")
    save_csv(image_counts, out_dir / "images_per_country.csv")
    save_csv(image_dist, out_dir / "images_per_country_dist.csv")

    if not image_counts.empty:
        plot_count_hist(
            image_counts["image_count"],
            out_dir / "images_per_country_hist.png",
            "Images per country",
            "images per country",
        )

    class_counts = compute_class_counts(
        df,
        default_class=default_class,
        class_names=class_names,
    )
    save_csv(class_counts, out_dir / "class_counts.csv")

    _plot_geometry(
        df,
        out_dir=out_dir,
        calc_fn=geom_calc,
        column_prefix=geom_prefix,
        output_prefix=geom_output_prefix,
    )

    top_country, bottom_country = (
        top_bottom(df, "country") if "country" in df.columns else (pd.DataFrame(), pd.DataFrame())
    )

    section = ""
    section += (
        f"<p>Images: {stats.get('n_images', 0)} | Objects: {stats.get('n_objects', 0)} "
        f"| Objects/image: {stats.get('objects_per_image', 0):.2f}</p>"
    )
    section += f"<p>Countries: {stats.get('n_countries', 0)}"
    if "n_regions" in stats:
        section += f" | Regions: {stats['n_regions']}"
    section += "</p>"
    section += "<h3>Top/bottom countries</h3>"
    section += render_table_grid([
        ("Top countries", top_country),
        ("Bottom countries", bottom_country),
    ])
    if not region_counts.empty:
        section += "<h3>Regions distribution</h3>"
        section += render_table(region_counts)
    images_hist = out_dir / "images_per_country_hist.png"
    if images_hist.exists():
        section += "<h3>Images per country</h3>"
        section += f"<img src='artifacts/{dataset_key}/images_per_country_hist.png' width='420'>"
    section += "<h3>Geometry</h3>"
    area_hist = out_dir / f"{geom_output_prefix}_area_hist.png"
    aspect_hist = out_dir / f"{geom_output_prefix}_aspect_hist.png"
    if area_hist.exists():
        section += f"<img src='artifacts/{dataset_key}/{geom_output_prefix}_area_hist.png' width='420'>"
    if aspect_hist.exists():
        section += f"<img src='artifacts/{dataset_key}/{geom_output_prefix}_aspect_hist.png' width='420'>"

    return build_report_section(title, section)


def build_main_dataset_section(
    df: pd.DataFrame,
    *,
    run_dir: Path,
    default_class: str,
    class_names: list[str] | None,
) -> str:
    return build_dataset_section(
        df,
        run_dir=run_dir,
        dataset_key="main",
        title="Main dataset",
        default_class=default_class,
        class_names=class_names,
        geom_calc=calc_bbox_area_aspect,
        geom_prefix="bbox",
        geom_output_prefix="bbox",
    )


def build_golden_dataset_section(
    df: pd.DataFrame,
    *,
    run_dir: Path,
    default_class: str,
    class_names: list[str] | None,
) -> str:
    return build_dataset_section(
        df,
        run_dir=run_dir,
        dataset_key="golden",
        title="Golden dataset",
        default_class=default_class,
        class_names=class_names,
        geom_calc=calc_crop_area_aspect,
        geom_prefix="crop",
        geom_output_prefix="bbox",
    )


def build_detector_section(
    det_df: pd.DataFrame,
    *,
    cfg: AnalyzeRunConfig,
    rng: Random,
    run_dir: Path,
    img_root: Path,
) -> str:
    det_df_local = det_df
    if not det_df.empty and "class_name" not in det_df.columns and "cls" in det_df.columns:
        det_df_local = det_df.copy()
        det_df_local["class_name"] = det_df_local["cls"].astype(int).apply(
            lambda x: class_name(x, cfg.detector.class_names)
        )

    det_counts_path = run_dir / "artifacts" / "detector" / "detections_per_image.csv"
    det_conf_hist = run_dir / "artifacts" / "detector" / "conf_hist.png"
    det_class_counts = value_counts(det_df_local, "class_name")
    save_csv(det_class_counts, run_dir / "artifacts" / "detector" / "class_counts.csv")

    if det_counts_path.exists():
        det_counts = pd.read_csv(det_counts_path)["detections_per_image"].tolist()
        plot_hist(
            det_counts,
            run_dir / "artifacts" / "detector" / "detections_hist.png",
            "Detections per image",
            "detections",
        )

    if not det_df_local.empty:
        plot_hist(det_df_local["conf"], det_conf_hist, "Detection confidence", "confidence")
        for cls_name in det_class_counts["class_name"].head(cfg.output.max_categories_plots).tolist():
            subset = det_df_local[det_df_local["class_name"] == cls_name]
            safe_name = cls_name.replace("/", "_")
            plot_hist(
                subset["conf"],
                run_dir / "artifacts" / "detector" / f"conf_hist_{safe_name}.png",
                f"Confidence ({cls_name})",
                "confidence",
            )

    det_section = ""
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
        det_section += "<h3>Class counts</h3>" + render_table(det_class_counts.head(20))

    if not det_df_local.empty:
        color_map = make_color_map(det_df_local["cls"].astype(int).tolist())
        class_names = det_class_counts["class_name"].head(cfg.output.max_categories_plots).tolist()
        gallery_by_class, labels_by_class = save_detection_galleries(
            det_df_local,
            img_root,
            run_dir / "artifacts" / "detector" / "gallery",
            class_names,
            rng,
            cfg.output.gallery_per_category,
            color_map,
        )
        if gallery_by_class:
            det_section += "<h3>Sample detections by class</h3>"
            for cls_name, paths in gallery_by_class.items():
                if not paths:
                    continue
                rel = relative_paths(paths, run_dir)
                det_section += f"<h4>{cls_name}</h4><div class='grid'>"
                det_section += "".join([f"<img src='{p}' loading='lazy'>" for p in rel])
                det_section += "</div>"
                labels = labels_by_class.get(cls_name, [])
                det_section += render_expand_list("Image files", labels)

    return build_report_section("Detector predictions (main)", det_section)


def build_classifier_section(
    preds: pd.DataFrame,
    *,
    cfg: AnalyzeRunConfig,
    run_dir: Path,
    dataset_key: str,
    title: str,
    img_root: Path,
    eval_label: str | None = None,
) -> str:
    out_dir = run_dir / "artifacts" / "classifier" / dataset_key
    save_csv(preds, out_dir / "predictions.csv")
    metrics = compute_metrics(preds)
    save_json(metrics, out_dir / "metrics.json")

    country_groups = group_accuracy(preds, "country", cfg.output.min_support)
    region_groups = group_accuracy(preds, "region", cfg.output.min_support)
    save_csv(country_groups, out_dir / "country_groups.csv")
    save_csv(region_groups, out_dir / "region_groups.csv")

    country_conf = confusion_pairs(preds, "country", "pred_country", cfg.output.top_k)
    region_conf = confusion_pairs(preds, "region", "pred_region", cfg.output.top_k)
    save_csv(country_conf, out_dir / "confusion_country.csv")
    save_csv(region_conf, out_dir / "confusion_region.csv")

    correct = (
        preds[preds["correct_top1"]].sample(
            n=min(cfg.output.gallery_size, len(preds[preds["correct_top1"]])),
            random_state=cfg.seed,
        )
        if not preds.empty
        else pd.DataFrame()
    )
    incorrect = preds[~preds["correct_top1"]]
    incorrect_sample = (
        incorrect.sample(
            n=min(cfg.output.gallery_size, len(incorrect)),
            random_state=cfg.seed,
        )
        if not incorrect.empty
        else pd.DataFrame()
    )
    high_conf_wrong = incorrect.sort_values("top1_conf", ascending=False).head(cfg.output.gallery_size)

    galleries = {
        "correct": save_gallery(
            correct,
            img_root,
            out_dir / "gallery_correct",
            "correct",
            cfg.classifier.expand,
            cfg.output.gallery_size,
            cfg.classifier.img_size,
        ),
        "incorrect": save_gallery(
            incorrect_sample,
            img_root,
            out_dir / "gallery_incorrect",
            "incorrect",
            cfg.classifier.expand,
            cfg.output.gallery_size,
            cfg.classifier.img_size,
        ),
        "high_conf_wrong": save_gallery(
            high_conf_wrong,
            img_root,
            out_dir / "gallery_high_conf_wrong",
            "highconf_wrong",
            cfg.classifier.expand,
            cfg.output.gallery_size,
            cfg.classifier.img_size,
        ),
    }

    section = ""
    if eval_label:
        section += f"<p>Eval set: {escape_html(eval_label)}</p>"
    section += "<p>Top-1 country: {:.3f} | Top-5 country: {:.3f}</p>".format(
        metrics.get("top1_country", 0.0),
        metrics.get("top5_country", 0.0),
    )
    if "top1_region" in metrics:
        section += "<p>Top-1 region: {:.3f} | Top-5 region: {:.3f}</p>".format(
            metrics.get("top1_region", 0.0),
            metrics.get("top5_region", 0.0),
        )
    section += "<h3>Best/worst countries</h3>"
    if not country_groups.empty:
        worst_countries = country_groups.sort_values("top1_accuracy", ascending=True).head(cfg.output.top_k)
        best_countries = country_groups.sort_values("top1_accuracy", ascending=False).head(cfg.output.top_k)
    else:
        worst_countries = country_groups
        best_countries = country_groups
    section += render_table_grid([
        ("Worst", worst_countries),
        ("Best", best_countries),
    ])
    if not region_groups.empty:
        section += "<h3>Best/worst regions</h3>"
        worst_regions = region_groups.sort_values("top1_accuracy", ascending=True).head(cfg.output.top_k)
        best_regions = region_groups.sort_values("top1_accuracy", ascending=False).head(cfg.output.top_k)
        section += render_table_grid([
            ("Worst", worst_regions),
            ("Best", best_regions),
        ])
    section += "<h3>Top confusion pairs (country)</h3>" + render_table(country_conf)
    if not region_conf.empty:
        section += "<h3>Top confusion pairs (region)</h3>" + render_table(region_conf)

    if galleries["correct"]:
        rel = relative_paths(galleries["correct"], run_dir)
        section += "<h3>Correct samples</h3><div class='grid'>"
        section += "".join([f"<img src='{p}' loading='lazy'>" for p in rel])
        section += "</div>"
        section += render_expand_list("Image files", format_gallery_labels(correct))
    if galleries["incorrect"]:
        rel = relative_paths(galleries["incorrect"], run_dir)
        section += "<h3>Incorrect samples</h3><div class='grid'>"
        section += "".join([f"<img src='{p}' loading='lazy'>" for p in rel])
        section += "</div>"
        section += render_expand_list("Image files", format_gallery_labels(incorrect_sample))
    if galleries["high_conf_wrong"]:
        rel = relative_paths(galleries["high_conf_wrong"], run_dir)
        section += "<h3>High-confidence wrong</h3><div class='grid'>"
        section += "".join([f"<img src='{p}' loading='lazy'>" for p in rel])
        section += "</div>"
        section += render_expand_list("Image files", format_gallery_labels(high_conf_wrong))

    return build_report_section(title, section)


def load_tb_scalars(
    tb_dir: Path,
    logger: logging.Logger,
) -> dict[str, list[tuple[int, float]]]:
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


def build_tensorboard_section(
    tb_dir: Path,
    *,
    run_dir: Path,
    logger: logging.Logger,
) -> str:
    scalars = load_tb_scalars(tb_dir, logger)
    if not scalars:
        return ""

    section = f"<p class='meta'>Source: {escape_html(str(tb_dir))}</p>"
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
        plot_scalar_series(steps, values, out_path, tag)
        if out_path.exists():
            section += f"<h3>{escape_html(tag)}</h3>"
            section += f"<img src='artifacts/tensorboard/{safe_tag}.png' width='520'>"

    return build_report_section("TensorBoard scalars", section)
