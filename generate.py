#!/usr/bin/env python3
"""CLI wrapper: python generate.py --prompt '...' --first-image a.png --last-image b.png"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from minimax_h3_fl2v.offline import enforce_offline_runtime

enforce_offline_runtime()

from minimax_h3_fl2v.cli import main

if __name__ == "__main__":
    main()
