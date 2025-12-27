import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bollards.io.fs import ensure_dir


class TestEnsureDir(unittest.TestCase):
    def test_creates_nested_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "a" / "b" / "c"
            ensure_dir(path)
            self.assertTrue(path.exists())
            self.assertTrue(path.is_dir())


if __name__ == "__main__":
    unittest.main()
