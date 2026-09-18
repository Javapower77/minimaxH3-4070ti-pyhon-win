#!/usr/bin/env python3
"""Container health probe for the local Gradio server."""

from __future__ import annotations

import sys
import urllib.error
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:7860/", timeout=4) as response:
        if 200 <= response.status < 500:
            raise SystemExit(0)
except (OSError, urllib.error.URLError):
    pass

print("Gradio is not responding", file=sys.stderr)
raise SystemExit(1)
