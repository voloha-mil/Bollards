from __future__ import annotations

import json
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from bollards.config import apply_overrides, load_config


@dataclass
class InnerConfig:
    flag: bool = False
    count: int = 1
    ratio: float = 0.5
    names: list[str] = field(default_factory=lambda: ["a"])


@dataclass
class OuterConfig:
    inner: InnerConfig = field(default_factory=InnerConfig)
    text: str = "hi"


class DummyConfig:
    def __init__(self, data: dict) -> None:
        self.data = data

    @classmethod
    def from_dict(cls, data: dict) -> "DummyConfig":
        return cls(data)


class TestConfigOverrides(unittest.TestCase):
    def test_apply_overrides_coerces_types(self) -> None:
        cfg = OuterConfig()
        apply_overrides(
            cfg,
            [
                "inner.flag=true",
                "inner.count=4",
                "inner.ratio=2.5",
                "inner.names=a, b ,c",
                "text=hello",
            ],
        )
        self.assertTrue(cfg.inner.flag)
        self.assertEqual(cfg.inner.count, 4)
        self.assertAlmostEqual(cfg.inner.ratio, 2.5)
        self.assertEqual(cfg.inner.names, ["a", "b", "c"])
        self.assertEqual(cfg.text, "hello")

    def test_apply_overrides_unknown_key_raises(self) -> None:
        cfg = OuterConfig()
        with self.assertRaises(KeyError):
            apply_overrides(cfg, ["inner.missing=1"])


class TestConfigIncludes(unittest.TestCase):
    def test_load_config_resolves_includes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.json").write_text(
                json.dumps(
                    {
                        "value": 10,
                        "list": [1, 2],
                        "nested": {"from_a": True, "child": 1},
                    }
                ),
                encoding="utf-8",
            )
            (root / "b.json").write_text(
                json.dumps({"child": 5, "extra": "yes"}),
                encoding="utf-8",
            )
            (root / "base.json").write_text(
                json.dumps(
                    {
                        "include": "a.json",
                        "value": 3,
                        "nested": {"include": "b.json", "child": 7},
                    }
                ),
                encoding="utf-8",
            )
            cfg = load_config(root / "base.json", DummyConfig)

        self.assertEqual(cfg.data["value"], 3)
        self.assertEqual(cfg.data["list"], [1, 2])
        self.assertEqual(cfg.data["nested"], {"child": 7, "extra": "yes"})

    def test_load_config_includes_list(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.json").write_text(
                json.dumps({"value": 1, "from_a": True}),
                encoding="utf-8",
            )
            (root / "b.json").write_text(
                json.dumps({"value": 2, "from_b": True}),
                encoding="utf-8",
            )
            (root / "base.json").write_text(
                json.dumps({"include": ["a.json", "b.json"], "extra": "yes"}),
                encoding="utf-8",
            )
            cfg = load_config(root / "base.json", DummyConfig)

        self.assertEqual(cfg.data["value"], 2)
        self.assertTrue(cfg.data["from_a"])
        self.assertTrue(cfg.data["from_b"])
        self.assertEqual(cfg.data["extra"], "yes")


if __name__ == "__main__":
    unittest.main()
