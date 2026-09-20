"""Study 1-R controlled-synthetic paired refit package."""

from .common import ACTION_NAMES, MODALITIES, SEMANTIC_VARIABLES, STATE_COLUMNS
from .generator import (
    ObservationBatch,
    ScenarioTape,
    generate_all_offline,
    generate_offline_split,
    make_scenario_tape,
    observe_state,
    oracle_actions,
    state_diagnostics,
    step_dynamics,
)

__all__ = [
    "ACTION_NAMES",
    "MODALITIES",
    "SEMANTIC_VARIABLES",
    "STATE_COLUMNS",
    "ObservationBatch",
    "ScenarioTape",
    "generate_all_offline",
    "generate_offline_split",
    "make_scenario_tape",
    "observe_state",
    "oracle_actions",
    "state_diagnostics",
    "step_dynamics",
]

__version__ = "2.0.0"
