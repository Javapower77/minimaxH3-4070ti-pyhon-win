#!/usr/bin/env python3
"""Launch the MiniMax-H3 FL2VA Gradio studio (offline, unfiltered)."""

import os

# Must be set before importing torch (package imports execute __init__.py first).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from minimax_h3_fl2v.offline import enforce_offline_runtime

enforce_offline_runtime()

from minimax_h3_fl2v.ui import main

if __name__ == "__main__":
    main()
