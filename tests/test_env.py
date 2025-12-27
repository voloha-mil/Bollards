import importlib.util
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bollards.utils.env import load_env


class TestLoadEnv(unittest.TestCase):
    def setUp(self) -> None:
        if importlib.util.find_spec("dotenv") is None:
            self.skipTest("python-dotenv not installed")

    def test_loads_env_file(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("BOLLARDS_TEST_ENV=hello\n", encoding="utf-8")

            key = "BOLLARDS_TEST_ENV"
            prev = os.environ.pop(key, None)
            try:
                load_env(str(path))
                self.assertEqual(os.environ.get(key), "hello")
            finally:
                if prev is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prev

    def test_does_not_override_existing_env(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("BOLLARDS_TEST_ENV=from_file\n", encoding="utf-8")

            key = "BOLLARDS_TEST_ENV"
            prev = os.environ.get(key)
            os.environ[key] = "from_env"
            try:
                load_env(str(path))
                self.assertEqual(os.environ.get(key), "from_env")
            finally:
                if prev is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prev


if __name__ == "__main__":
    unittest.main()
