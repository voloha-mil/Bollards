#!/usr/bin/env python3
from __future__ import annotations

import sys

from bollards.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["prepare-local", *sys.argv[1:]]))
