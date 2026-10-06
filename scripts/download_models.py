#!/usr/bin/env python3
"""Download MiniMax-H3 FL2VA models by type.

Missing Hugging Face/Civitai tokens are requested using masked terminal input.
Press Enter for anonymous downloads; --no-input disables prompts. Existing
environment or Hugging Face login credentials are reused; entered tokens are
temporary and never saved to disk.

Examples:
    python scripts/download_models.py --type 12gb
    python scripts/download_models.py --type pruned
    python scripts/download_models.py --type latent_upscaler
    python scripts/download_models.py --type taomate
    python scripts/download_models.py --type dasiwa
    python scripts/download_models.py --type dasiwa_v2
    python scripts/download_models.py --type dmad
    python scripts/download_models.py --type dmad_hyperflow
    python scripts/download_models.py --type pdmd_dmad
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
