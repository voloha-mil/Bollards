#!/usr/bin/env python3
from __future__ import annotations

import sys

from bollards.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["mine-osv5m", *sys.argv[1:]]))
