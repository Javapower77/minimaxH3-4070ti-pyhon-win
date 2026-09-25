#!/usr/bin/env python3
"""Download MiniMax-H3 FL2VA models by type.

Examples:
    python scripts/download_models.py --type 12gb
    python scripts/download_models.py --type pruned
    python scripts/download_models.py --type taomate
    python scripts/download_models.py --type dasiwa
    python scripts/download_models.py --type h100
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimax_h3_fl2v.download import main


if __name__ == "__main__":
    main()
