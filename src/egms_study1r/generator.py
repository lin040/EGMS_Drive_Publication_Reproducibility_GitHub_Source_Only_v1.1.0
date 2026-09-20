"""Deterministic controlled-synthetic data generator for Study 1-R.

This module owns only the data-generating process and common vehicle dynamics.
It has no dependency on either candidate pipeline.  A split or scenario tape
is generated once and is subsequently shared by both fitted pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .common import (
    ACTION_NAMES,
    MODALITIES,
    SEMANTIC_STATE_INDICES,
    SEMANTIC_VARIABLES,
    STATE_COLUMNS,
    STATE_INDEX,
    array_sha256,
    load_protocol,
    observation_availability_column,
    observation_quality_column,
    observation_value_column,
    scenario_cells,
    stable_seed,
    validate_protocol,
)


@dataclass(frozen=True)
class ObservationBatch:
    """Sensor observations for every episode at one time index."""

    values: np.ndarray  # [episode, modality, semantic variable]
    quality: np.ndarray  # [episode, modality]
    available: np.ndarray  # [episode, modality]

    def validate(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("observation values must be [episode, modality, variable]")
        expected = (self.values.shape[0], len(MODALITIES), len(SEMANTIC_VARIABLES))
        if self.values.shape != expected:
            raise ValueError(f"observation values have shape {self.values.shape}; expected {expected}")
        if self.quality.shape != expected[:2] or self.available.shape != expected[:2]:
            raise ValueError("observation quality/availability shape mismatch")
        if np.any(~np.isfinite(self.quality)) or np.any(
            (self.quality < 0.0) | (self.quality > 1.0)
        ):
            raise ValueError("observed quality must be finite and in [0,1]")
        if np.any(np.isfinite(self.values[~self.available])):
            raise ValueError("unavailable observations must be represented by NaN")
        if np.any(~np.isfinite(self.values[self.available])):
            raise ValueError("available observations must be finite")


@dataclass(frozen=True)
class ScenarioTape:
    """Fixed exogenous inputs used by a vectorized closed-loop evaluation."""

    split: str
    data_seed: int
    training_replicate: int
    dt_s: float
    steps: int
    episode_id: np.ndarray
    cell_id: np.ndarray
    family: np.ndarray
    environment: np.ndarray
    density: np.ndarray
    episode_index: np.ndarray
    episode_seed: np.ndarray
    initial_state: np.ndarray
    desired_speed_mps: np.ndarray
    route_goal_m: np.ndarray
    lead_acceleration_mps2: np.ndarray
    gap_impulse_m: np.ndarray
    crossing_risk: np.ndarray
    route_urgency: np.ndarray
    quality_true: np.ndarray
    quality_observed: np.ndarray
    standard_noise: np.ndarray
    dropout_uniform: np.ndarray
    outlier_uniform: np.ndarray
    outlier_noise: np.ndarray

    @property
    def n_episodes(self) -> int:
        return int(len(self.episode_id))

    def validate(self) -> None:
        n = self.n_episodes
        time_points = self.steps + 1
        if self.initial_state.shape != (n, len(STATE_COLUMNS)):
            raise ValueError("initial_state shape does not follow STATE_COLUMNS")
        for name in (
            "episode_id",
            "cell_id",
            "family",
            "environment",
            "density",
            "episode_index",
            "episode_seed",
            "desired_speed_mps",
            "route_goal_m",
        ):
            if np.asarray(getattr(self, name)).shape != (n,):
                raise ValueError(f"{name} must have one value per episode")
        if self.lead_acceleration_mps2.shape != (n, self.steps):
            raise ValueError("lead_acceleration_mps2 shape mismatch")
        if self.gap_impulse_m.shape != (n, self.steps):
            raise ValueError("gap_impulse_m shape mismatch")
        if self.crossing_risk.shape != (n, time_points):
            raise ValueError("crossing_risk shape mismatch")
        if self.route_urgency.shape != (n, time_points):
            raise ValueError("route_urgency shape mismatch")
        quality_shape = (n, time_points, len(MODALITIES))
        if self.quality_true.shape != quality_shape:
            raise ValueError("quality_true shape mismatch")
        if self.quality_observed.shape != quality_shape:
            raise ValueError("quality_observed shape mismatch")
        observation_shape = quality_shape + (len(SEMANTIC_VARIABLES),)
        if self.standard_noise.shape != observation_shape:
            raise ValueError("standard_noise shape mismatch")
        if self.outlier_noise.shape != observation_shape:
            raise ValueError("outlier_noise shape mismatch")
        if self.dropout_uniform.shape != quality_shape:
            raise ValueError("dropout_uniform shape mismatch")
        if self.outlier_uniform.shape != quality_shape:
            raise ValueError("outlier_uniform shape mismatch")
        if self.steps <= 0 or self.dt_s <= 0:
            raise ValueError("tape steps and dt_s must be positive")
        if len(set(self.episode_id.tolist())) != n:
            raise ValueError("episode_id values must be unique")

    def digest(self) -> str:
        """Content digest used to prove paired use of one immutable tape."""

        text_arrays = (
            np.char.encode(np.asarray([self.split], dtype=str), "utf-8"),
            np.char.encode(self.episode_id.astype(str), "utf-8"),
            np.char.encode(self.cell_id.astype(str), "utf-8"),
            np.char.encode(self.family.astype(str), "utf-8"),
            np.char.encode(self.environment.astype(str), "utf-8"),
            np.char.encode(self.density.astype(str), "utf-8"),
        )
        return array_sha256(
            *text_arrays,
            np.asarray(
                [
                    float(self.data_seed),
                    float(self.training_replicate),
                    float(self.dt_s),
                    float(self.steps),
                ],
                dtype=np.float64,
            ),
            self.episode_index,
            self.episode_seed,
            self.initial_state,
            self.desired_speed_mps,
            self.route_goal_m,
            self.lead_acceleration_mps2,
            self.gap_impulse_m,
            self.crossing_risk,
            self.route_urgency,
            self.quality_true,
            self.quality_observed,
            self.standard_noise,
            self.dropout_uniform,
            self.outlier_uniform,
            self.outlier_noise,
        )


def _as_protocol(protocol: Mapping[str, object] | str | Path) -> dict:
    if isinstance(protocol, (str, Path)):
        return load_protocol(protocol)
    value = dict(protocol)
    validate_protocol(value)
    return value


def _split_specification(
    protocol: Mapping[str, object], split: str, training_replicate: int | None
) -> tuple[int, int, int]:
    data = protocol["data"]
    if split == "train":
        if training_replicate is None:
            raise ValueError("training_replicate is required for the train split")
        replicate = int(training_replicate)
        seeds = tuple(int(value) for value in data["training_data_seeds"])
        if not 0 <= replicate < len(seeds):
            raise ValueError(f"training_replicate must be in [0,{len(seeds) - 1}]")
        return replicate, seeds[replicate], int(data["train_episodes_per_cell"])
    if training_replicate is not None:
        raise ValueError("training_replicate is only valid for the train split")
    if split == "validation":
        return -1, int(data["fixed_validation_seed"]), int(
            data["validation_episodes_per_cell"]
        )
    if split == "test":
        return -1, int(data["fixed_test_seed"]), int(data["test_episodes_per_cell"])
    if split == "rollout":
        return -1, int(data["fixed_rollout_seed"]), int(
            data["rollout_episodes_per_cell"]
        )
    raise ValueError("split must be train, validation, test, or rollout")


def _profile_array(profile: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(profile[name]) for name in SEMANTIC_VARIABLES], dtype=float)


def make_scenario_tape(
    protocol: Mapping[str, object] | str | Path,
    split: str,
    training_replicate: int | None = None,
    steps: int | None = None,
) -> ScenarioTape:
    """Generate all exogenous values before any fitted pipeline is evaluated."""

    cfg = _as_protocol(protocol)
    replicate, data_seed, episodes_per_cell = _split_specification(
        cfg, split, training_replicate
    )
    if steps is None:
        steps = (
            int(cfg["data"]["rollout_frames"])
            if split == "rollout"
            else int(cfg["data"]["frames_per_episode"]) + 1
        )
    steps = int(steps)
    if steps <= 0:
        raise ValueError("steps must be positive")

    cells = scenario_cells(cfg)
    n = len(cells) * episodes_per_cell
    n_time = steps + 1
    n_modalities = len(MODALITIES)
    n_variables = len(SEMANTIC_VARIABLES)
    dt_s = float(cfg["data"]["dt_s"])
    generator_cfg = cfg["generator"]
    dynamics_cfg = cfg["dynamics"]

    episode_ids: list[str] = []
    cell_ids: list[str] = []
    families: list[str] = []
    environments: list[str] = []
    densities: list[str] = []
    episode_indices = np.empty(n, dtype=np.int16)
    episode_seeds = np.empty(n, dtype=np.uint32)
    initial_state = np.empty((n, len(STATE_COLUMNS)), dtype=float)
    desired_speed = np.empty(n, dtype=float)
    route_goal = np.empty(n, dtype=float)
    lead_acceleration = np.empty((n, steps), dtype=float)
    gap_impulse = np.zeros((n, steps), dtype=float)
    crossing_risk = np.empty((n, n_time), dtype=float)
    route_urgency = np.empty((n, n_time), dtype=float)
    quality_true = np.empty((n, n_time, n_modalities), dtype=float)
    quality_observed = np.empty_like(quality_true)
    standard_noise = np.empty((n, n_time, n_modalities, n_variables), dtype=float)
    dropout_uniform = np.empty((n, n_time, n_modalities), dtype=float)
    outlier_uniform = np.empty_like(dropout_uniform)
    outlier_noise = np.empty_like(standard_noise)

    density_profiles = generator_cfg["density_profiles"]
    family_profiles = generator_cfg["family_profiles"]
    quality_profiles = generator_cfg["environment_quality"]
    quality_reduction = generator_cfg["density_quality_reduction"]
    if split == "train":
        prefix = f"train_r{replicate:02d}"
    elif split in {"test", "rollout"}:
        # Offline and closed-loop endpoints use the same fixed 360 episode IDs
        # so crossed resampling addresses a common evaluation-cell index.
        prefix = "fixed_test"
    else:
        prefix = split
    row_index = 0
    normalized_time = np.linspace(0.0, 1.0, n_time)

    for cell in cells:
        density_profile = density_profiles[cell.density]
        family_profile = family_profiles[cell.family]
        for episode_index in range(episodes_per_cell):
            episode_seed = stable_seed(
                "study1r",
                data_seed,
                split,
                replicate,
                cell.family,
                cell.environment,
                cell.density,
                episode_index,
            )
            rng = np.random.default_rng(episode_seed)
            episode_id = f"{prefix}__{cell.cell_id}__e{episode_index:02d}"
            episode_ids.append(episode_id)
            cell_ids.append(cell.cell_id)
            families.append(cell.family)
            environments.append(cell.environment)
            densities.append(cell.density)
            episode_indices[row_index] = episode_index
            episode_seeds[row_index] = episode_seed

            ego_speed = max(
                3.0,
                float(density_profile["initial_ego_speed_mps"])
                + rng.normal(0.0, float(generator_cfg["initial_speed_sd_mps"])),
            )
            closing_speed = (
                float(family_profile["initial_closing_speed_mps"])
                + rng.normal(
                    0.0, float(generator_cfg["initial_closing_speed_sd_mps"])
                )
            )
            gap = max(
                5.0,
                float(density_profile["initial_gap_m"])
                + float(family_profile["gap_adjustment_m"])
                + rng.normal(0.0, float(generator_cfg["initial_gap_sd_m"])),
            )
            desired = max(ego_speed + rng.normal(1.2, 0.35), ego_speed)
            desired_speed[row_index] = desired
            route_goal[row_index] = (
                desired
                * steps
                * dt_s
                * float(dynamics_cfg["route_goal_fraction_of_nominal_progress"])
            )

            center = float(
                np.clip(
                    float(generator_cfg["event_center_fraction"])
                    + rng.normal(
                        0.0, float(generator_cfg["event_center_jitter_sd"])
                    ),
                    0.30,
                    0.75,
                )
            )
            width = float(generator_cfg["event_width_fraction"])
            pulse = np.exp(-0.5 * ((normalized_time - center) / width) ** 2)
            background_risk = float(density_profile["background_crossing_risk"])
            risk_peak = float(family_profile["crossing_risk_peak"])
            risk_noise = rng.normal(0.0, 0.012, size=n_time)
            crossing = background_risk + (risk_peak - background_risk) * pulse + risk_noise
            crossing = np.clip(crossing, 0.0, 1.0)
            crossing_risk[row_index] = crossing

            urgency_peak = float(family_profile["route_urgency_peak"])
            logistic = 1.0 / (1.0 + np.exp(-10.0 * (normalized_time - center)))
            urgency = 0.06 + (urgency_peak - 0.06) * logistic
            urgency += rng.normal(0.0, 0.008, size=n_time)
            route_urgency[row_index] = np.clip(urgency, 0.0, 1.0)

            lead_acceleration[row_index] = (
                float(family_profile["lead_acceleration_peak_mps2"]) * pulse[:-1]
                + rng.normal(
                    0.0,
                    float(generator_cfg["exogenous_acceleration_noise_sd_mps2"]),
                    size=steps,
                )
            )
            total_gap_impulse = float(family_profile["gap_impulse_m"])
            if total_gap_impulse != 0.0:
                pulse_steps = pulse[:-1]
                gap_impulse[row_index] = total_gap_impulse * pulse_steps / np.maximum(
                    pulse_steps.sum(), np.finfo(float).tiny
                )

            initial_state[row_index] = np.asarray(
                [
                    gap,
                    closing_speed,
                    ego_speed,
                    0.0,
                    0.0,
                    crossing[0],
                    route_urgency[row_index, 0],
                ],
                dtype=float,
            )

            episode_health = rng.uniform(
                float(generator_cfg["episode_health_low"]),
                float(generator_cfg["episode_health_high"]),
                size=n_modalities,
            )
            for modality_index, modality in enumerate(MODALITIES):
                base_quality = (
                    float(quality_profiles[cell.environment][modality])
                    - float(quality_reduction[cell.density])
                )
                innovations = rng.normal(0.0, 0.018, size=n_time)
                quality_path = np.empty(n_time, dtype=float)
                quality_path[0] = base_quality * episode_health[modality_index] + innovations[0]
                for time_index in range(1, n_time):
                    quality_path[time_index] = (
                        0.82 * quality_path[time_index - 1]
                        + 0.18 * base_quality * episode_health[modality_index]
                        + innovations[time_index]
                    )
                quality_true[row_index, :, modality_index] = np.clip(
                    quality_path, 0.08, 0.995
                )
            quality_observed[row_index] = np.clip(
                quality_true[row_index]
                + rng.normal(
                    0.0,
                    float(generator_cfg["quality_observation_sd"]),
                    size=(n_time, n_modalities),
                ),
                0.05,
                1.0,
            )
            standard_noise[row_index] = rng.normal(
                0.0, 1.0, size=(n_time, n_modalities, n_variables)
            )
            dropout_uniform[row_index] = rng.random((n_time, n_modalities))
            outlier_uniform[row_index] = rng.random((n_time, n_modalities))
            outlier_noise[row_index] = rng.normal(
                0.0, 1.0, size=(n_time, n_modalities, n_variables)
            )
            row_index += 1

    tape = ScenarioTape(
        split=split,
        data_seed=data_seed,
        training_replicate=replicate,
        dt_s=dt_s,
        steps=steps,
        episode_id=np.asarray(episode_ids, dtype=str),
        cell_id=np.asarray(cell_ids, dtype=str),
        family=np.asarray(families, dtype=str),
        environment=np.asarray(environments, dtype=str),
        density=np.asarray(densities, dtype=str),
        episode_index=episode_indices,
        episode_seed=episode_seeds,
        initial_state=initial_state,
        desired_speed_mps=desired_speed,
        route_goal_m=route_goal,
        lead_acceleration_mps2=lead_acceleration,
        gap_impulse_m=gap_impulse,
        crossing_risk=crossing_risk,
        route_urgency=route_urgency,
        quality_true=quality_true,
        quality_observed=quality_observed,
        standard_noise=standard_noise,
        dropout_uniform=dropout_uniform,
        outlier_uniform=outlier_uniform,
        outlier_noise=outlier_noise,
    )
    tape.validate()
    return tape


def _base_noise_matrix(protocol: Mapping[str, object]) -> np.ndarray:
    noise = protocol["generator"]["base_noise_sd"]
    return np.asarray(
        [
            [float(noise[modality][variable]) for variable in SEMANTIC_VARIABLES]
            for modality in MODALITIES
        ],
        dtype=float,
    )


def observe_state(
    state: np.ndarray,
    tape: ScenarioTape,
    frame_index: int,
    protocol: Mapping[str, object] | str | Path,
) -> ObservationBatch:
    """Observe possibly divergent states using the same pre-generated noise tape."""

    cfg = _as_protocol(protocol)
    values = np.asarray(state, dtype=float)
    if values.shape != (tape.n_episodes, len(STATE_COLUMNS)):
        raise ValueError(
            f"state has shape {values.shape}; expected {(tape.n_episodes, len(STATE_COLUMNS))}"
        )
    index = int(frame_index)
    if not 0 <= index <= tape.steps:
        raise ValueError(f"frame_index must be in [0,{tape.steps}]")
    generator_cfg = cfg["generator"]
    semantic_state = values[:, SEMANTIC_STATE_INDICES]
    quality_true = tape.quality_true[:, index, :]
    quality_observed = tape.quality_observed[:, index, :]
    noise_scale = _base_noise_matrix(cfg)[None, :, :] / (
        0.25 + 0.75 * quality_true[:, :, None]
    )
    observed = semantic_state[:, None, :] + (
        tape.standard_noise[:, index, :, :] * noise_scale
    )

    outlier_probability = float(generator_cfg["outlier_base_probability"]) + float(
        generator_cfg["outlier_quality_slope"]
    ) * (1.0 - quality_true)
    outlier_mask = tape.outlier_uniform[:, index, :] < outlier_probability
    observed += (
        outlier_mask[:, :, None]
        * tape.outlier_noise[:, index, :, :]
        * noise_scale
        * float(generator_cfg["outlier_scale"])
    )

    for variable_index, variable in enumerate(SEMANTIC_VARIABLES):
        low, high = (float(value) for value in generator_cfg["observation_clips"][variable])
        observed[:, :, variable_index] = np.clip(
            observed[:, :, variable_index], low, high
        )

    dropout_probability = float(generator_cfg["dropout_base_probability"]) + float(
        generator_cfg["dropout_quality_slope"]
    ) * (1.0 - quality_true)
    # Ego-state availability is physically much more stable than exteroceptive
    # sensing, but remains nonzero and is generated before evaluation.
    dropout_probability[:, MODALITIES.index("ego")] *= 0.10
    available = tape.dropout_uniform[:, index, :] >= dropout_probability
    observed[~available] = np.nan
    batch = ObservationBatch(observed, quality_observed.copy(), available)
    batch.validate()
    return batch


def state_diagnostics(
    state: np.ndarray, protocol: Mapping[str, object] | str | Path
) -> dict[str, np.ndarray]:
    cfg = _as_protocol(protocol)
    values = np.asarray(state, dtype=float)
    gap = values[:, STATE_INDEX["gap_m"]]
    closing = values[:, STATE_INDEX["closing_speed_mps"]]
    speed = values[:, STATE_INDEX["ego_speed_mps"]]
    oracle_cfg = cfg["oracle"]
    ttc_cap = float(oracle_cfg["ttc_cap_s"])
    ttc = np.full(len(values), ttc_cap, dtype=float)
    closing_mask = closing > 0.1
    ttc[closing_mask] = np.clip(
        gap[closing_mask] / closing[closing_mask], 0.0, ttc_cap
    )
    stopping_margin = gap - (
        float(oracle_cfg["standstill_gap_m"])
        + float(oracle_cfg["reaction_time_s"]) * speed
        + np.maximum(closing, 0.0) ** 2
        / (2.0 * float(oracle_cfg["comfortable_deceleration_mps2"]))
    )
    return {"ttc_s": ttc, "stopping_margin_m": stopping_margin}


def oracle_actions(
    state: np.ndarray, protocol: Mapping[str, object] | str | Path
) -> np.ndarray:
    """Assign one of four ordered expert actions from latent physical state."""

    cfg = _as_protocol(protocol)
    values = np.asarray(state, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(STATE_COLUMNS):
        raise ValueError("state must have shape [episode, len(STATE_COLUMNS)]")
    diagnostics = state_diagnostics(values, cfg)
    ttc = diagnostics["ttc_s"]
    margin = diagnostics["stopping_margin_m"]
    gap = values[:, STATE_INDEX["gap_m"]]
    crossing = values[:, STATE_INDEX["crossing_risk"]]
    oracle_cfg = cfg["oracle"]

    actions = np.zeros(len(values), dtype=np.int8)
    slow = (
        (ttc < float(oracle_cfg["slow_ttc_s"]))
        | (margin < float(oracle_cfg["slow_stopping_margin_m"]))
        | (crossing > float(oracle_cfg["slow_crossing_risk"]))
    )
    actions[slow] = 1
    yielding = (
        (ttc < float(oracle_cfg["yield_ttc_s"]))
        | (margin < float(oracle_cfg["yield_stopping_margin_m"]))
        | (crossing > float(oracle_cfg["yield_crossing_risk"]))
    )
    actions[yielding] = 2
    stopping = (
        (ttc < float(oracle_cfg["stop_ttc_s"]))
        | (crossing > float(oracle_cfg["stop_crossing_risk"]))
        | (gap <= float(cfg["dynamics"]["collision_gap_m"]))
    )
    actions[stopping] = 3
    return actions


def step_dynamics(
    state: np.ndarray,
    executed_actions: np.ndarray,
    tape: ScenarioTape,
    frame_index: int,
    protocol: Mapping[str, object] | str | Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Advance all episodes one step with common action-to-control dynamics."""

    cfg = _as_protocol(protocol)
    values = np.asarray(state, dtype=float)
    if values.shape != (tape.n_episodes, len(STATE_COLUMNS)):
        raise ValueError("state shape does not match the scenario tape")
    actions = np.asarray(executed_actions, dtype=int)
    if actions.shape != (tape.n_episodes,) or np.any((actions < 0) | (actions >= 4)):
        raise ValueError("executed_actions must be one action ID per episode in [0,3]")
    index = int(frame_index)
    if not 0 <= index < tape.steps:
        raise ValueError(f"frame_index must be in [0,{tape.steps - 1}]")

    dt_s = float(cfg["data"]["dt_s"])
    dynamics_cfg = cfg["dynamics"]
    gap = values[:, STATE_INDEX["gap_m"]]
    closing = values[:, STATE_INDEX["closing_speed_mps"]]
    speed = values[:, STATE_INDEX["ego_speed_mps"]]
    acceleration = values[:, STATE_INDEX["acceleration_mps2"]]
    progress = values[:, STATE_INDEX["progress_m"]]
    lead_speed = np.maximum(0.0, speed - closing)

    keep_command = np.clip(
        float(dynamics_cfg["keep_speed_gain"]) * (tape.desired_speed_mps - speed),
        -float(dynamics_cfg["keep_acceleration_limit_mps2"]),
        float(dynamics_cfg["keep_acceleration_limit_mps2"]),
    )
    commands = keep_command.copy()
    commands[actions == 1] = float(dynamics_cfg["slow_command_mps2"])
    commands[actions == 2] = float(dynamics_cfg["yield_command_mps2"])
    commands[actions == 3] = float(dynamics_cfg["stop_command_mps2"])
    lag = float(dynamics_cfg["acceleration_lag"])
    next_acceleration = np.clip(
        lag * acceleration + (1.0 - lag) * commands,
        float(dynamics_cfg["maximum_deceleration_mps2"]),
        float(dynamics_cfg["maximum_acceleration_mps2"]),
    )
    next_speed = np.maximum(0.0, speed + next_acceleration * dt_s)
    next_lead_speed = np.maximum(
        0.0, lead_speed + tape.lead_acceleration_mps2[:, index] * dt_s
    )
    next_closing = next_speed - next_lead_speed
    next_gap = (
        gap
        + 0.5 * ((lead_speed - speed) + (next_lead_speed - next_speed)) * dt_s
        + tape.gap_impulse_m[:, index]
    )
    next_progress = progress + 0.5 * (speed + next_speed) * dt_s
    next_state = np.column_stack(
        [
            next_gap,
            next_closing,
            next_speed,
            next_acceleration,
            next_progress,
            tape.crossing_risk[:, index + 1],
            tape.route_urgency[:, index + 1],
        ]
    )

    diagnostics = state_diagnostics(next_state, cfg)
    crossing = next_state[:, STATE_INDEX["crossing_risk"]]
    longitudinal_collision = next_gap <= float(dynamics_cfg["collision_gap_m"])
    crossing_collision = (
        (crossing >= float(dynamics_cfg["crossing_collision_risk"]))
        & (next_speed >= float(dynamics_cfg["crossing_collision_speed_mps"]))
    )
    collision = longitudinal_collision | crossing_collision
    near_miss = (~collision) & (
        (diagnostics["ttc_s"] < float(dynamics_cfg["near_miss_ttc_s"]))
        | (next_gap < float(dynamics_cfg["near_miss_gap_m"]))
        | (
            (crossing >= float(dynamics_cfg["crossing_near_miss_risk"]))
            & (next_speed > 2.0)
        )
    )
    critical = collision | near_miss
    events = {
        "command_acceleration_mps2": commands,
        "jerk_mps3": (next_acceleration - acceleration) / dt_s,
        "ttc_s": diagnostics["ttc_s"],
        "stopping_margin_m": diagnostics["stopping_margin_m"],
        "collision": collision,
        "near_miss": near_miss,
        "critical_event": critical,
        "route_goal_reached": next_progress >= tape.route_goal_m,
    }
    return next_state, events


def _append_observation(
    row: dict[str, object], batch: ObservationBatch, episode_index: int, lag: str
) -> None:
    for modality_index, modality in enumerate(MODALITIES):
        for variable_index, variable in enumerate(SEMANTIC_VARIABLES):
            row[observation_value_column(modality, lag, variable)] = float(
                batch.values[episode_index, modality_index, variable_index]
            )
        row[observation_quality_column(modality, lag)] = float(
            batch.quality[episode_index, modality_index]
        )
        row[observation_availability_column(modality, lag)] = int(
            batch.available[episode_index, modality_index]
        )


def generate_offline_split(
    protocol: Mapping[str, object] | str | Path,
    split: str,
    training_replicate: int | None = None,
) -> pd.DataFrame:
    """Generate one labeled temporal split from the frozen physical process."""

    if split == "rollout":
        raise ValueError("use make_scenario_tape for closed-loop rollout")
    cfg = _as_protocol(protocol)
    frames = int(cfg["data"]["frames_per_episode"])
    tape = make_scenario_tape(
        cfg, split, training_replicate=training_replicate, steps=frames + 1
    )
    state = tape.initial_state.copy()
    previous_observation = observe_state(state, tape, 0, cfg)
    burn_in_action = oracle_actions(state, cfg)
    state, _ = step_dynamics(state, burn_in_action, tape, 0, cfg)
    rows: list[dict[str, object]] = []

    for frame_index in range(frames):
        tape_index = frame_index + 1
        current_observation = observe_state(state, tape, tape_index, cfg)
        labels = oracle_actions(state, cfg)
        diagnostics = state_diagnostics(state, cfg)
        for episode_position in range(tape.n_episodes):
            row: dict[str, object] = {
                "split": split,
                "training_replicate": int(tape.training_replicate),
                "data_seed": int(tape.data_seed),
                "cell_id": str(tape.cell_id[episode_position]),
                "episode_id": str(tape.episode_id[episode_position]),
                "episode_index": int(tape.episode_index[episode_position]),
                "episode_seed": int(tape.episode_seed[episode_position]),
                "family": str(tape.family[episode_position]),
                "environment": str(tape.environment[episode_position]),
                "density": str(tape.density[episode_position]),
                "frame": int(frame_index),
                "time_s": float(frame_index * tape.dt_s),
                "sample_id": f"{tape.episode_id[episode_position]}__f{frame_index:02d}",
                "has_previous": 1,
                "latent_ttc_s": float(diagnostics["ttc_s"][episode_position]),
                "latent_stopping_margin_m": float(
                    diagnostics["stopping_margin_m"][episode_position]
                ),
                "y_true": int(labels[episode_position]),
                "action_name": ACTION_NAMES[int(labels[episode_position])],
            }
            for state_index, name in enumerate(STATE_COLUMNS):
                row[f"latent_{name}"] = float(state[episode_position, state_index])
            _append_observation(row, previous_observation, episode_position, "prev")
            _append_observation(row, current_observation, episode_position, "current")
            rows.append(row)

        state, _ = step_dynamics(state, labels, tape, tape_index, cfg)
        previous_observation = current_observation

    frame = pd.DataFrame.from_records(rows)
    frame = frame.sort_values(["cell_id", "episode_id", "frame"], kind="stable")
    frame = frame.reset_index(drop=True)
    observed_classes = set(int(value) for value in frame["y_true"].unique())
    if observed_classes != set(range(len(ACTION_NAMES))):
        raise RuntimeError(
            f"{split} split generated action classes {sorted(observed_classes)}; expected 0..3"
        )
    if frame["sample_id"].duplicated().any():
        raise RuntimeError("sample_id values are not unique")
    return frame


def generate_all_offline(
    protocol: Mapping[str, object] | str | Path,
) -> dict[str, pd.DataFrame]:
    """Generate the ten formal training samples and the two fixed splits."""

    cfg = _as_protocol(protocol)
    outputs = {
        f"train_{replicate:02d}": generate_offline_split(
            cfg, "train", training_replicate=replicate
        )
        for replicate in range(int(cfg["data"]["formal_training_replicates"]))
    }
    outputs["validation"] = generate_offline_split(cfg, "validation")
    outputs["test"] = generate_offline_split(cfg, "test")
    return outputs
