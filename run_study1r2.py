#!/usr/bin/env python3
"""Command-line entry point for exploratory Study 1-R2."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from egms_study1r2.runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
