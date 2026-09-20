"""Shared constants and deterministic helpers for Study 1-R.

The data generator, model trainers, analysis, and validator import the column
contract from this module.  Keeping the contract in one small module prevents
the two candidate pipelines from silently receiving different data.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import yaml


ACTION_NAMES = ("KEEP", "SLOW", "YIELD", "STOP")
MODALITIES = ("camera", "lidar", "radar", "ego")
SEMANTIC_VARIABLES = (
    "gap_m",
    "closing_speed_mps",
    "ego_speed_mps",
    "crossing_risk",
    "route_urgency",
)
STATE_COLUMNS = (
    "gap_m",
    "closing_speed_mps",
    "ego_speed_mps",
    "acceleration_mps2",
    "progress_m",
    "crossing_risk",
    "route_urgency",
)
OBSERVATION_LAGS = ("prev", "current")

STATE_INDEX = {name: index for index, name in enumerate(STATE_COLUMNS)}
SEMANTIC_STATE_INDICES = np.asarray(
    [STATE_INDEX[name] for name in SEMANTIC_VARIABLES], dtype=int
)


@dataclass(frozen=True, order=True)
class ScenarioCell:
    """One cell in the frozen family × environment × density matrix."""

    family: str
    environment: str
    density: str

    @property
    def cell_id(self) -> str:
        return f"{self.family}__{self.environment}__{self.density}"


def stable_seed(*parts: object) -> int:
    """Create a stable NumPy seed without relying on Python's salted hash."""

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
        2**32 - 1
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(*arrays: np.ndarray) -> str:
    """Hash array content together with dtype and shape metadata."""

    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.maximum(
        exponentials.sum(axis=axis, keepdims=True), np.finfo(float).tiny
    )


def load_protocol(path: str | Path) -> dict:
    """Read and validate the frozen YAML protocol."""

    with Path(path).open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    if not isinstance(protocol, dict):
        raise ValueError("Study 1-R protocol must be a YAML mapping")
    validate_protocol(protocol)
    return protocol


def scenario_cells(protocol: Mapping[str, object]) -> tuple[ScenarioCell, ...]:
    matrix = protocol["scenario_matrix"]
    if not isinstance(matrix, Mapping):
        raise ValueError("scenario_matrix must be a mapping")
    return tuple(
        ScenarioCell(str(family), str(environment), str(density))
        for family, environment, density in product(
            matrix["families"], matrix["environments"], matrix["densities"]
        )
    )


def observation_value_column(modality: str, lag: str, variable: str) -> str:
    return f"{modality}_{lag}_{variable}"


def observation_quality_column(modality: str, lag: str) -> str:
    return f"q_{modality}_{lag}"


def observation_availability_column(modality: str, lag: str) -> str:
    return f"available_{modality}_{lag}"


def observation_columns(
    *, include_quality: bool = True, include_availability: bool = True
) -> list[str]:
    """Return the canonical temporal observation columns in stable order."""

    columns: list[str] = []
    for lag in OBSERVATION_LAGS:
        for modality in MODALITIES:
            columns.extend(
                observation_value_column(modality, lag, variable)
                for variable in SEMANTIC_VARIABLES
            )
            if include_quality:
                columns.append(observation_quality_column(modality, lag))
            if include_availability:
                columns.append(observation_availability_column(modality, lag))
    return columns


def latent_columns() -> list[str]:
    return [f"latent_{name}" for name in STATE_COLUMNS]


def validate_protocol(protocol: Mapping[str, object]) -> None:
    """Fail early if a run no longer matches the frozen Study 1-R contract."""

    actions = protocol.get("actions", {})
    if tuple(actions.get("names", ())) != ACTION_NAMES:
        raise ValueError(f"actions.names must be exactly {ACTION_NAMES}")
    if tuple(int(value) for value in actions.get("class_ids", ())) != tuple(
        range(len(ACTION_NAMES))
    ):
        raise ValueError("actions.class_ids must be the contiguous values 0..3")

    generator = protocol.get("generator", {})
    if tuple(generator.get("modalities", ())) != MODALITIES:
        raise ValueError(f"generator.modalities must be exactly {MODALITIES}")
    if tuple(generator.get("semantic_variables", ())) != SEMANTIC_VARIABLES:
        raise ValueError(
            f"generator.semantic_variables must be exactly {SEMANTIC_VARIABLES}"
        )

    cells = scenario_cells(protocol)
    expected_cells = int(protocol["scenario_matrix"]["expected_cells"])
    if len(cells) != expected_cells or expected_cells != 45:
        raise ValueError("the frozen scenario matrix must contain exactly 45 cells")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise ValueError("scenario cells are not unique")

    data = protocol.get("data", {})
    required_exact = {
        "frames_per_episode": 16,
        "rollout_frames": 20,
        "train_episodes_per_cell": 4,
        "validation_episodes_per_cell": 2,
        "test_episodes_per_cell": 8,
        "rollout_episodes_per_cell": 8,
        "formal_training_replicates": 10,
    }
    for key, expected in required_exact.items():
        if int(data.get(key, -1)) != expected:
            raise ValueError(f"data.{key} must equal the frozen value {expected}")
    if not np.isclose(float(data.get("dt_s", np.nan)), 0.2):
        raise ValueError("data.dt_s must equal 0.2 s")

    training_data_seeds = tuple(int(value) for value in data.get("training_data_seeds", ()))
    if len(training_data_seeds) != 10 or len(set(training_data_seeds)) != 10:
        raise ValueError("exactly 10 unique formal training-data seeds are required")
    pilot_seeds = set(int(value) for value in data.get("pilot_training_data_seeds", ()))
    if pilot_seeds.intersection(training_data_seeds):
        raise ValueError("pilot and formal training-data seeds must not overlap")
    fixed_seeds = {
        int(data.get("fixed_validation_seed", -1)),
        int(data.get("fixed_test_seed", -1)),
        int(data.get("fixed_rollout_seed", -1)),
    }
    if len(fixed_seeds) != 3 or fixed_seeds.intersection(training_data_seeds):
        raise ValueError(
            "fixed validation/test/rollout seeds must be unique and formal-data disjoint"
        )

    model_seeds = tuple(
        int(value) for value in protocol.get("models", {}).get("model_training_seeds", ())
    )
    if len(model_seeds) != 10 or len(set(model_seeds)) != 10:
        raise ValueError("exactly 10 unique model-training seeds are required")

    quality = generator.get("environment_quality", {})
    for cell in cells:
        if cell.environment not in quality:
            raise ValueError(f"missing quality profile for {cell.environment}")
        if set(quality[cell.environment]) != set(MODALITIES):
            raise ValueError(
                f"quality profile for {cell.environment} must cover all modalities"
            )

    noise = generator.get("base_noise_sd", {})
    if set(noise) != set(MODALITIES):
        raise ValueError("base_noise_sd must cover every modality")
    for modality in MODALITIES:
        if set(noise[modality]) != set(SEMANTIC_VARIABLES):
            raise ValueError(
                f"base_noise_sd.{modality} must cover every semantic variable"
            )
        if any(float(noise[modality][name]) <= 0 for name in SEMANTIC_VARIABLES):
            raise ValueError("all base observation-noise scales must be positive")

    density_profiles = generator.get("density_profiles", {})
    expected_densities = set(protocol["scenario_matrix"]["densities"])
    if set(density_profiles) != expected_densities:
        raise ValueError("density_profiles must cover the frozen density levels")
    quality_reduction = generator.get("density_quality_reduction", {})
    if set(quality_reduction) != expected_densities:
        raise ValueError("density_quality_reduction must cover the frozen density levels")

    family_profiles = generator.get("family_profiles", {})
    expected_families = set(protocol["scenario_matrix"]["families"])
    if set(family_profiles) != expected_families:
        raise ValueError("family_profiles must cover the frozen scenario families")
    required_family_values = {
        "gap_adjustment_m",
        "initial_closing_speed_mps",
        "lead_acceleration_peak_mps2",
        "gap_impulse_m",
        "crossing_risk_peak",
        "route_urgency_peak",
    }
    for family, profile in family_profiles.items():
        if set(profile) != required_family_values:
            raise ValueError(f"family profile {family} does not match the frozen schema")

    shared_head = protocol.get("models", {}).get("shared_head", {})
    if tuple(int(value) for value in shared_head.get("hidden_layer_sizes", ())) != (48, 24):
        raise ValueError("models.shared_head.hidden_layer_sizes must be [48, 24]")
    if int(shared_head.get("epochs", -1)) != 50:
        raise ValueError("models.shared_head.epochs must equal 50")
    if shared_head.get("class_weighting") != "inverse_frequency_balanced_from_training_split":
        raise ValueError("models.shared_head.class_weighting differs from the frozen rule")

    metrics = protocol.get("metrics", {})
    if metrics.get("ttc_conditioning") != "jointly_collision_free_episode_pairs_only":
        raise ValueError("metrics.ttc_conditioning differs from the frozen paired estimand")
    if int(metrics.get("ece_bins", -1)) != 15 or metrics.get("ece_binning") != "equal_width":
        raise ValueError("Study 1-R requires 15 equal-width ECE bins")

    statistics = protocol.get("statistics", {})
    frozen_statistics = {
        "estimand": "structured_r2_minus_baseline_b",
        "bootstrap_type": "paired_crossed_cluster_percentile",
        "bootstrap_repetitions": 10000,
        "quantile_algorithm": "numpy_linear",
        "sign_flip_tests": "exact_all_1024_sign_patterns",
        "multiplicity_adjustment": "holm",
    }
    for key, expected in frozen_statistics.items():
        if statistics.get(key) != expected:
            raise ValueError(f"statistics.{key} must equal the frozen value {expected!r}")
    bootstrap_seed = int(statistics.get("bootstrap_seed", -1))
    if bootstrap_seed <= 0:
        raise ValueError("statistics.bootstrap_seed must be a positive recorded integer")


def require_columns(columns: Iterable[str], required: Iterable[str], label: str) -> None:
    present = set(columns)
    missing = sorted(set(required).difference(present))
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")
