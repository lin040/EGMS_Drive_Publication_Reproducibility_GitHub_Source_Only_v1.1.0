#!/usr/bin/env python3
"""Public entry point for the frozen Study 1 evaluation code.

Reader-facing outputs use the two method names ``Baseline B`` and
``Structured fusion``.  The imported implementation retains its archived
machine identifier so the frozen source hashes and checkpoint provenance can
be verified exactly.
"""

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
