#!/usr/bin/env python3
"""Run, reanalyze, or validate the controlled synthetic Studies 2-3."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egms_studies23.runner import main  # noqa: E402


if __name__ == "__main__":
    main()
