import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bollards.io.csv import (
    append_csv_row,
    append_processed_ids,
    ensure_csv_header,
    load_processed_ids,
)


class TestCsvIo(unittest.TestCase):
    def test_load_and_append_processed_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.txt"
            path.write_text("a\n\n b \n", encoding="utf-8")
            self.assertEqual(load_processed_ids(path), {"a", "b"})

            append_processed_ids(path, ["c", "d"])
            self.assertEqual(load_processed_ids(path), {"a", "b", "c", "d"})

    def test_ensure_header_and_append_row(self) -> None:
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "data.csv"
            fieldnames = ["a", "b"]
            ensure_csv_header(csv_path, fieldnames)
            append_csv_row(csv_path, fieldnames, {"a": 1})

            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["a"], "1")
            self.assertEqual(rows[0]["b"], "")

            lines = csv_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "a,b")


if __name__ == "__main__":
    unittest.main()
