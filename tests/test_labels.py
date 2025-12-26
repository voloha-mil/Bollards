import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bollards.data.labels import load_id_to_country


class TestLoadIdToCountry(unittest.TestCase):
    def test_none_path_returns_none(self) -> None:
        self.assertIsNone(load_id_to_country(None))

    def test_load_mapping(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "country_map.json"
            path.write_text(json.dumps({"US": 0, "DE": 2}), encoding="utf-8")
            result = load_id_to_country(str(path))
        self.assertEqual(result, ["US", "", "DE"])


if __name__ == "__main__":
    unittest.main()
