from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

T = TypeVar("T")


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_includes(data: Any, base_dir: Path) -> Any:
    if isinstance(data, dict):
        merged: dict[str, Any] = {}
        include = data.get("include")
        if include:
            include_list = include if isinstance(include, list) else [include]
            for inc in include_list:
                inc_path = Path(inc)
                if not inc_path.is_absolute():
                    inc_path = base_dir / inc_path
                with inc_path.open("r", encoding="utf-8") as f:
                    inc_data = json.load(f)
                inc_data = _resolve_includes(inc_data, inc_path.parent)
                if isinstance(inc_data, dict):
                    merged = _merge_dicts(merged, inc_data)
        for key, value in data.items():
            if key == "include":
                continue
            merged[key] = _resolve_includes(value, base_dir)
        return merged
    if isinstance(data, list):
        return [_resolve_includes(item, base_dir) for item in data]
    return data


def load_config(path: Path, cls: Type[T]) -> T:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data = _resolve_includes(data, path.parent)
    return cls.from_dict(data)  # type: ignore[attr-defined]


def resolve_config_path(config_path: Optional[str], default_name: str) -> Path:
    if config_path:
        return Path(config_path)
    default_path = Path("configs") / default_name
    if default_path.exists():
        return default_path
    raise SystemExit(f"Missing config. Provide --config or create {default_path}.")


def _parse_override_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        lowered = raw.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        return raw


def _coerce_value(value: Any, current: Any) -> Any:
    if current is None:
        return value
    if isinstance(current, bool):
        return bool(value)
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value.strip():
            return [v.strip() for v in value.split(",") if v.strip()]
        return value
    if isinstance(current, str):
        return str(value)
    return value


def apply_overrides(cfg: T, overrides: list[str]) -> T:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item}. Use key=value")
        key, raw = item.split("=", 1)
        value = _parse_override_value(raw)

        target: Any = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise KeyError(f"Unknown config key: {key}")
            target = getattr(target, part)

        attr = parts[-1]
        if not hasattr(target, attr):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(target, attr)
        setattr(target, attr, _coerce_value(value, current))

    return cfg
