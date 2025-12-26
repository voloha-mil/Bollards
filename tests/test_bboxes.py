import unittest

import pandas as pd

from bollards.data.bboxes import compute_avg_bbox_wh


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


if __name__ == "__main__":
    unittest.main()
