from __future__ import annotations

from pathlib import Path
from typing import Optional


def load_env(env_path: Optional[str] = None) -> None:
    """
    Best-effort .env loader. If python-dotenv is missing, this is a no-op.
    """
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    if env_path:
        path = Path(env_path)
        if path.exists():
            load_dotenv(dotenv_path=path, override=False)
            return

    load_dotenv(override=False)
