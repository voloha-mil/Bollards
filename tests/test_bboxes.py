import unittest

import pandas as pd

from bollards.data.bboxes import (
    bbox_xyxy_norm_to_center,
    bbox_xyxy_norm_to_pixels,
    compute_avg_bbox_wh,
    expand_bbox_xyxy_norm,
    normalize_bbox_xyxy_px,
)


class TestComputeAvgBBoxWH(unittest.TestCase):
    def test_basic_average(self) -> None:
        df = pd.DataFrame(
            {
                "x1": [0.0, 1.0],
                "y1": [0.0, 2.0],
                "x2": [2.0, 4.0],
                "y2": [3.0, 6.0],
            }
        )
        avg_w, avg_h = compute_avg_bbox_wh(df, label="train")
        self.assertAlmostEqual(avg_w, 2.5)
        self.assertAlmostEqual(avg_h, 3.5)

    def test_clips_negative_sizes(self) -> None:
        df = pd.DataFrame(
            {
                "x1": [5.0, 0.0],
                "y1": [5.0, 0.0],
                "x2": [3.0, 2.0],
                "y2": [4.0, 3.0],
            }
        )
        avg_w, avg_h = compute_avg_bbox_wh(df, label="train")
        self.assertAlmostEqual(avg_w, 1.0)
        self.assertAlmostEqual(avg_h, 1.5)

    def test_missing_columns_raises(self) -> None:
        df = pd.DataFrame({"x1": [0.0], "x2": [1.0]})
        with self.assertRaises(ValueError):
            compute_avg_bbox_wh(df, label="train")

    def test_no_valid_values_raises(self) -> None:
        df = pd.DataFrame(
            {
                "x1": [None],
                "y1": [None],
                "x2": [None],
                "y2": [None],
            }
        )
        with self.assertRaises(ValueError):
            compute_avg_bbox_wh(df, label="train")


class TestBBoxHelpers(unittest.TestCase):
    def test_normalize_bbox_xyxy_px_clamps_and_sorts(self) -> None:
        x1n, y1n, x2n, y2n = normalize_bbox_xyxy_px(
            x1=10.0,
            y1=-5.0,
            x2=-2.0,
            y2=20.0,
            w=10,
            h=10,
        )
        self.assertEqual((x1n, y1n, x2n, y2n), (0.0, 0.0, 1.0, 1.0))

    def test_normalize_bbox_xyxy_px_min_size(self) -> None:
        x1n, y1n, x2n, y2n = normalize_bbox_xyxy_px(
            x1=5.0,
            y1=5.0,
            x2=5.0,
            y2=5.0,
            w=10,
            h=10,
        )
        self.assertGreater(x2n, x1n)
        self.assertGreater(y2n, y1n)

    def test_bbox_xyxy_norm_to_center(self) -> None:
        xc, yc, bw, bh = bbox_xyxy_norm_to_center(0.0, 0.0, 1.0, 0.5)
        self.assertAlmostEqual(xc, 0.5)
        self.assertAlmostEqual(yc, 0.25)
        self.assertAlmostEqual(bw, 1.0)
        self.assertAlmostEqual(bh, 0.5)

    def test_expand_bbox_xyxy_norm_clamps(self) -> None:
        ex1, ey1, ex2, ey2 = expand_bbox_xyxy_norm(0.0, 0.0, 0.1, 0.1, expand=3.0)
        self.assertGreaterEqual(ex1, 0.0)
        self.assertGreaterEqual(ey1, 0.0)
        self.assertLessEqual(ex2, 1.0)
        self.assertLessEqual(ey2, 1.0)

    def test_bbox_xyxy_norm_to_pixels_min_size(self) -> None:
        px1, py1, px2, py2 = bbox_xyxy_norm_to_pixels(0.5, 0.5, 0.5, 0.5, w=10, h=10)
        self.assertEqual((px1, py1), (5, 5))
        self.assertGreater(px2, px1)
        self.assertGreater(py2, py1)


if __name__ == "__main__":
    unittest.main()
