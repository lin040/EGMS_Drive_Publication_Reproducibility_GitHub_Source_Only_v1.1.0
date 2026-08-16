"""Protocol configuration loading and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .statistics import feasible_correlation_bounds


DISCLAIMER = (
    "All numerical outputs are conditional design quantities generated under "
    "prespecified event-rate and dependence assumptions. They are not empirical "
    "performance estimates of EGMS-Drive or any implemented driving system."
)


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML protocol."""

    protocol_path = Path(path).expanduser().resolve()
    if not protocol_path.is_file():
        raise FileNotFoundError(f"protocol file not found: {protocol_path}")
    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a YAML mapping")
    errors = validate_protocol(protocol)
    if errors:
        raise ValueError("Invalid protocol:\n- " + "\n- ".join(errors))
    protocol["_meta"] = {
        "path": str(protocol_path),
        "sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
    }
    return protocol


def _require(mapping: dict[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def validate_protocol(protocol: dict[str, Any]) -> list[str]:
    """Return protocol-validation errors; an empty list means valid."""

    errors: list[str] = []
    required = [
        "project.version",
        "project.status",
        "project.empirical_data_present",
        "project.empirical_performance_claims",
        "primary_design.endpoint",
        "primary_design.estimand",
        "primary_design.control_rate_assumption",
        "primary_design.candidate_rate_assumption",
        "primary_design.alpha",
        "primary_design.target_power",
        "simulation.master_seed",
        "simulation.canonical_repetitions",
        "paired_design.correlations",
        "clustered_design.mean_cluster_size",
        "clustered_design.icc_values",
        "scenario_matrix.allocation_block",
    ]
    values: dict[str, Any] = {}
    for key in required:
        try:
            values[key] = _require(protocol, key)
        except KeyError:
            errors.append(f"missing required key: {key}")
    if errors:
        return errors

    if values["project.status"] != "prospective_design_only":
        errors.append("project.status must be 'prospective_design_only'")
    if values["project.empirical_data_present"] is not False:
        errors.append("project.empirical_data_present must be false")
    if values["project.empirical_performance_claims"] is not False:
        errors.append("project.empirical_performance_claims must be false")

    p0 = values["primary_design.control_rate_assumption"]
    p1 = values["primary_design.candidate_rate_assumption"]
    alpha = values["primary_design.alpha"]
    power = values["primary_design.target_power"]
    for name, value in [("control rate", p0), ("candidate rate", p1)]:
        if not isinstance(value, (int, float)) or not 0 < value < 1:
            errors.append(f"{name} must be numeric and in (0, 1)")
    if isinstance(p0, (int, float)) and isinstance(p1, (int, float)) and p0 == p1:
        errors.append("control and candidate rate assumptions must differ")
    if not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        errors.append("alpha must be in (0, 1)")
    if not isinstance(power, (int, float)) or not 0 < power < 1:
        errors.append("target_power must be in (0, 1)")

    repetitions = values["simulation.canonical_repetitions"]
    if not isinstance(repetitions, int) or repetitions < 10_000:
        errors.append("canonical_repetitions must be an integer >= 10000")
    if not isinstance(values["simulation.master_seed"], int):
        errors.append("master_seed must be an integer")

    if (
        isinstance(p0, (int, float))
        and isinstance(p1, (int, float))
        and 0 < p0 < 1
        and 0 < p1 < 1
    ):
        lower, upper = feasible_correlation_bounds(float(p0), float(p1))
        correlations = values["paired_design.correlations"]
        if not isinstance(correlations, list) or not correlations:
            errors.append("paired_design.correlations must be a non-empty list")
        else:
            for correlation in correlations:
                if not isinstance(correlation, (int, float)):
                    errors.append("all paired correlations must be numeric")
                elif not lower <= correlation <= upper:
                    errors.append(
                        f"paired correlation {correlation} is outside feasible "
                        f"range [{lower:.6f}, {upper:.6f}]"
                    )

    mean_cluster_size = values["clustered_design.mean_cluster_size"]
    if not isinstance(mean_cluster_size, (int, float)) or mean_cluster_size <= 0:
        errors.append("mean_cluster_size must be positive")
    iccs = values["clustered_design.icc_values"]
    if not isinstance(iccs, list) or not iccs:
        errors.append("clustered_design.icc_values must be a non-empty list")
    else:
        for icc in iccs:
            if not isinstance(icc, (int, float)) or not 0 <= icc < 1:
                errors.append("all ICC values must be numeric and in [0, 1)")

    block = values["scenario_matrix.allocation_block"]
    if not isinstance(block, int) or block <= 0:
        errors.append("scenario_matrix.allocation_block must be a positive integer")
    return errors


def canonical_protocol_json(protocol: dict[str, Any]) -> str:
    """Serialize protocol content without local metadata for hashing."""

    payload = {key: value for key, value in protocol.items() if key != "_meta"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
