"""Compatibility namespace for the frozen Study 1 source manifest.

The public comparison is Baseline B versus Structured fusion.  This namespace
keeps the original method-blind generator path and console entry point valid;
the complete frozen implementation lives in :mod:`egms_study1r2` under its
archived machine identifier.
"""

from egms_study1r2.common import ACTION_NAMES

__all__ = ["ACTION_NAMES"]
