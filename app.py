#!/usr/bin/env python3
"""Launch the MiniMax-H3 FL2VA Gradio studio (offline, unfiltered)."""

import os
import sys

# Must be set before importing torch (package imports execute __init__.py first).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Gradio/Uvicorn only needs socket I/O in this process. Windows' default
# Proactor loop can emit a noisy WinError 10054 callback when a browser closes a
# preview/SSE connection before transport shutdown. The ComfyUI subprocess keeps
# its own default loop and is unaffected.
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from minimax_h3_fl2v.offline import enforce_offline_runtime

enforce_offline_runtime()

from minimax_h3_fl2v.ui import main

if __name__ == "__main__":
    main()
