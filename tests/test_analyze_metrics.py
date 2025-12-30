from __future__ import annotations

import unittest

import pandas as pd

from bollards.constants import PATH_COL
from bollards.pipelines.analyze.metrics import (
    calc_bbox_area_aspect,
    calc_crop_area_aspect,
    compute_metrics,
    confusion_pairs,
    dataset_summary,
    group_accuracy,
    image_count_distribution,
    image_counts_by_country,
)


class TestAnalyzeCounts(unittest.TestCase):
    def test_dataset_summary_counts_unique_images(self) -> None:
        df = pd.DataFrame(
            {
                PATH_COL: [
                    "images/1.jpg",
                    "images/1.jpg",
                    "images/2.jpg",
                    "images/3.jpg",
                ],
                "country": ["US", "US", "DE", "DE"],
            }
        )
        out = dataset_summary(df, "country", None)
        self.assertEqual(out["n_images"], 3)
        self.assertEqual(out["n_objects"], 4)
        self.assertAlmostEqual(out["objects_per_image"], 4 / 3)
        self.assertEqual(out["n_countries"], 2)

    def test_image_counts_by_country_uses_unique_images(self) -> None:
        df = pd.DataFrame(
            {
                PATH_COL: ["images/1.jpg", "images/1.jpg", "images/2.jpg", "images/3.jpg"],
                "country": ["US", "US", "DE", "DE"],
            }
        )
        counts = image_counts_by_country(df, "country")
        got = dict(zip(counts["country"], counts["image_count"]))
        self.assertEqual(got, {"DE": 2, "US": 1})
        self.assertEqual(int(counts.iloc[0]["image_count"]), 2)

    def test_image_count_distribution(self) -> None:
        image_counts = pd.DataFrame(
            {"country": ["A", "B", "C", "D"], "image_count": [1, 1, 2, 5]}
        )
        dist = image_count_distribution(image_counts)
        got = dict(zip(dist["images_per_country"], dist["n_countries"]))
        self.assertEqual(got, {1: 2, 2: 1, 5: 1})


class TestAnalyzeGeometry(unittest.TestCase):
    def test_calc_bbox_area_aspect(self) -> None:
        df = pd.DataFrame({"x1": [0.2], "y1": [0.2], "x2": [0.7], "y2": [0.6]})
        out = calc_bbox_area_aspect(df, "bbox")
        self.assertAlmostEqual(float(out["bbox_area"].iloc[0]), 0.2)
        self.assertAlmostEqual(float(out["bbox_aspect"].iloc[0]), 1.25)

    def test_calc_crop_area_aspect(self) -> None:
        df = pd.DataFrame({"crop_w": [50], "crop_h": [25], "orig_w": [100], "orig_h": [100]})
        out = calc_crop_area_aspect(df, "crop")
        self.assertAlmostEqual(float(out["crop_area"].iloc[0]), 0.125)
        self.assertAlmostEqual(float(out["crop_aspect"].iloc[0]), 2.0)


class TestAnalyzeMetrics(unittest.TestCase):
    def test_compute_metrics_with_region(self) -> None:
        df = pd.DataFrame(
            {
                "correct_top1": [True, False, False, True],
                "correct_top5": [True, True, False, True],
                "region": ["EU", "EU", None, "NA"],
                "correct_region_top1": [True, False, False, True],
                "correct_region_top5": [True, True, False, True],
            }
        )
        metrics = compute_metrics(df)
        self.assertAlmostEqual(metrics["top1_country"], 0.5)
        self.assertAlmostEqual(metrics["top5_country"], 0.75)
        self.assertAlmostEqual(metrics["top1_region"], 2 / 3)
        self.assertAlmostEqual(metrics["top5_region"], 1.0)

    def test_group_accuracy_filters_and_labels(self) -> None:
        df = pd.DataFrame(
            {
                "country": ["A", "A", "B", "B", "B", "C"],
                "correct_top1": [1, 0, 1, 1, 0, 1],
            }
        )
        grouped = group_accuracy(df, "country", min_support=2)
        self.assertIn("top1_accuracy", grouped.columns)
        self.assertIn("support", grouped.columns)
        got = {row["country"]: (row["top1_accuracy"], row["support"]) for _, row in grouped.iterrows()}
        self.assertEqual(got, {"A": (0.5, 2), "B": (2 / 3, 3)})

    def test_confusion_pairs_orders_by_count(self) -> None:
        df = pd.DataFrame(
            {
                "country": ["A", "A", "A", "B", "B"],
                "pred_country": ["B", "B", "C", "A", "A"],
            }
        )
        pairs = confusion_pairs(df, "country", "pred_country", top_k=2)
        self.assertEqual(int(pairs.iloc[0]["count"]), 2)
        self.assertIn(("A", "B"), set(zip(pairs["country"], pairs["pred_country"])))


if __name__ == "__main__":
    unittest.main()
