"""Paired closed-loop evaluation and raw episode reconstruction for Study 1-R."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .common import (
    ACTION_NAMES,
    MODALITIES,
    SEMANTIC_VARIABLES,
    STATE_COLUMNS,
    STATE_INDEX,
    softmax,
)
from .generator import (
    ObservationBatch,
    ScenarioTape,
    observe_state,
    oracle_actions,
    step_dynamics,
)
from .models import MODEL_BASELINE_B, MODEL_STRUCTURED, Study1RClassifier


EVENT_COLUMNS = (
    "replicate_id",
    "training_seed",
    "method",
    "scenario_cell",
    "episode_seed",
    "episode_id",
    "frame_id",
    "time_s",
    "event_type",
    "event_value",
    "threshold",
    "source",
)


def _model_frame(previous: ObservationBatch, current: ObservationBatch) -> pd.DataFrame:
    """Build the exact method-blind temporal observation frame used by both models."""

    rows: dict[str, np.ndarray] = {}
    for lag, batch in (("prev", previous), ("current", current)):
        for modality_index, modality in enumerate(MODALITIES):
            for variable_index, variable in enumerate(SEMANTIC_VARIABLES):
                rows[f"{modality}_{lag}_{variable}"] = batch.values[
                    :, modality_index, variable_index
                ]
            rows[f"q_{modality}_{lag}"] = batch.quality[:, modality_index]
            rows[f"available_{modality}_{lag}"] = batch.available[
                :, modality_index
            ].astype(np.int8)
    return pd.DataFrame(rows)


def _state_columns(state: np.ndarray) -> dict[str, np.ndarray]:
    return {
        name: state[:, index].astype(float, copy=False)
        for index, name in enumerate(STATE_COLUMNS)
    }


def run_paired_rollouts(
    protocol: Mapping[str, object],
    tape: ScenarioTape,
    models: Mapping[str, Study1RClassifier],
    *,
    replicate_id: int,
    training_seed: int,
    protocol_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate both fitted methods on one immutable exogenous scenario tape.

    The state trajectory may diverge after the first action, but every method receives
    the same initial conditions and indexed exogenous noise.  Fixed-width step rows are
    retained after termination with ``valid_step=0`` so completeness is auditable.
    """

    expected_methods = {MODEL_BASELINE_B, MODEL_STRUCTURED}
    if set(models) != expected_methods:
        raise ValueError(f"models must contain exactly {sorted(expected_methods)}")
    tape.validate()
    dt_s = float(protocol["data"]["dt_s"])
    action_rows: list[pd.DataFrame] = []
    event_rows: list[dict[str, object]] = []

    for method in (MODEL_BASELINE_B, MODEL_STRUCTURED):
        model = models[method]
        state = tape.initial_state.copy()
        previous = observe_state(state, tape, 0, protocol)
        terminated = np.zeros(tape.n_episodes, dtype=bool)
        termination_reason = np.full(tape.n_episodes, "", dtype=object)
        collision_seen = np.zeros(tape.n_episodes, dtype=bool)
        near_seen = np.zeros(tape.n_episodes, dtype=bool)
        critical_seen = np.zeros(tape.n_episodes, dtype=bool)
        completion_seen = np.zeros(tape.n_episodes, dtype=bool)

        for step in range(tape.steps):
            # At step zero this is an explicit identical-observation warm start.
            # Thereafter ``previous`` is the prior pre-action state and ``current``
            # is aligned to this pre-action state and tape index.
            current = observe_state(state, tape, step, protocol)
            model_input = _model_frame(previous, current)
            logits = model.predict_logits(model_input)
            probabilities = softmax(logits / float(model.temperature), axis=1)
            predicted = probabilities.argmax(axis=1).astype(np.int8)
            oracle = oracle_actions(state, protocol).astype(np.int8)
            valid_before = ~terminated
            executed = predicted.copy()
            executed[terminated] = ACTION_NAMES.index("STOP")

            next_state, events = step_dynamics(state, executed, tape, step, protocol)
            for name, values in events.items():
                events[name] = np.asarray(values).copy()
            for name in events:
                if name in {
                    "collision",
                    "near_miss",
                    "critical_event",
                    "route_goal_reached",
                }:
                    events[name] = events[name].astype(bool) & valid_before
                else:
                    events[name][~valid_before] = 0.0
            next_state[~valid_before] = state[~valid_before]

            collision_now = events["collision"] & ~collision_seen
            near_now = events["near_miss"] & ~near_seen & ~collision_seen
            critical_now = events["critical_event"] & ~critical_seen
            complete_now = (
                events["route_goal_reached"]
                & ~completion_seen
                & ~collision_seen
                & ~collision_now
            )
            final_timeout = (step == tape.steps - 1) & valid_before & ~collision_now & ~complete_now
            terminal_now = collision_now | complete_now | final_timeout

            collision_seen |= events["collision"]
            near_seen |= events["near_miss"]
            critical_seen |= events["critical_event"]
            completion_seen |= complete_now
            termination_reason[collision_now] = "collision"
            termination_reason[complete_now] = "route_complete"
            termination_reason[final_timeout] = "timeout"

            state_payload = _state_columns(next_state)
            payload: dict[str, object] = {
                "protocol_sha256": np.repeat(protocol_sha256, tape.n_episodes),
                "replicate_id": np.repeat(int(replicate_id), tape.n_episodes),
                "training_seed": np.repeat(int(training_seed), tape.n_episodes),
                "method": np.repeat(method, tape.n_episodes),
                "scenario_cell": tape.cell_id,
                "episode_seed": tape.episode_seed,
                "episode_id": tape.episode_id,
                "route_goal_m": tape.route_goal_m,
                "frame_id": np.repeat(step, tape.n_episodes),
                # State and event fields below describe the post-action state.
                "time_s": np.repeat(float((step + 1) * dt_s), tape.n_episodes),
                "valid_step": valid_before.astype(np.int8),
                "oracle_label": oracle,
                "predicted_action": predicted,
                "executed_action": executed,
                "predicted_action_name": np.asarray(ACTION_NAMES, dtype=object)[predicted],
                "executed_action_name": np.asarray(ACTION_NAMES, dtype=object)[executed],
                "temperature": np.repeat(float(model.temperature), tape.n_episodes),
                "commanded_acceleration_mps2": events["command_acceleration_mps2"],
                "applied_acceleration_mps2": next_state[:, STATE_INDEX["acceleration_mps2"]],
                "jerk_mps3": events["jerk_mps3"],
                "ttc_s": events["ttc_s"],
                "stopping_margin_m": events["stopping_margin_m"],
                "collision_event": collision_now.astype(np.int8),
                "near_miss_condition": events["near_miss"].astype(np.int8),
                "critical_condition": events["critical_event"].astype(np.int8),
                "route_complete_event": complete_now.astype(np.int8),
                "terminal": terminal_now.astype(np.int8),
                "termination_reason": np.where(terminal_now, termination_reason, ""),
                **state_payload,
            }
            for class_index, action_name in enumerate(ACTION_NAMES):
                payload[f"logit_{action_name}"] = logits[:, class_index]
                payload[f"p_{action_name}"] = probabilities[:, class_index]
            # Preserve the exact temporal observations consumed by the checkpoint
            # so every stored rollout logit can be replayed without regenerating
            # endogenous states.
            for column in model_input.columns:
                payload[column] = model_input[column].to_numpy(copy=False)
            action_rows.append(pd.DataFrame(payload))

            # Safety events can be triggered by more than one physical condition,
            # so their ledger uses a neutral true/threshold pair instead of a
            # potentially misleading single-condition value.
            ones = np.ones(tape.n_episodes, dtype=float)
            for event_name, mask, values, threshold in (
                ("collision", collision_now, ones, 1.0),
                ("near_miss_entry", near_now, ones, 1.0),
                ("critical_entry", critical_now, ones, 1.0),
                ("route_complete", complete_now, next_state[:, STATE_INDEX["progress_m"]], tape.route_goal_m),
                ("timeout", final_timeout, next_state[:, STATE_INDEX["progress_m"]], tape.route_goal_m),
            ):
                for episode_index in np.flatnonzero(mask):
                    threshold_value = (
                        float(threshold[episode_index])
                        if isinstance(threshold, np.ndarray)
                        else float(threshold)
                    )
                    event_rows.append(
                        {
                            "replicate_id": int(replicate_id),
                            "training_seed": int(training_seed),
                            "method": method,
                            "scenario_cell": str(tape.cell_id[episode_index]),
                            "episode_seed": int(tape.episode_seed[episode_index]),
                            "episode_id": str(tape.episode_id[episode_index]),
                            "frame_id": int(step),
                            "time_s": float((step + 1) * dt_s),
                            "event_type": event_name,
                            "event_value": float(values[episode_index]),
                            "threshold": threshold_value,
                            "source": "raw_rollout_recomputation",
                        }
                    )

            terminated |= terminal_now
            state = next_state
            previous = current

    actions = pd.concat(action_rows, ignore_index=True)
    actions = actions.sort_values(
        ["replicate_id", "method", "scenario_cell", "episode_id", "frame_id"],
        kind="stable",
    ).reset_index(drop=True)
    events = pd.DataFrame.from_records(event_rows, columns=EVENT_COLUMNS)
    if len(events):
        events = events.sort_values(
            ["replicate_id", "method", "scenario_cell", "episode_id", "frame_id", "event_type"],
            kind="stable",
        ).reset_index(drop=True)
    return actions, events


def derive_episode_metrics(action_records: pd.DataFrame, protocol: Mapping[str, object]) -> pd.DataFrame:
    """Recompute every episode endpoint exclusively from persisted action rows."""

    dt_s = float(protocol["data"]["dt_s"])
    keys = [
        "replicate_id",
        "training_seed",
        "method",
        "scenario_cell",
        "episode_seed",
        "episode_id",
    ]
    rows: list[dict[str, object]] = []
    for key_values, group in action_records.groupby(keys, sort=False):
        valid = group.loc[group["valid_step"].astype(int).eq(1)].sort_values("frame_id")
        if valid.empty:
            raise ValueError(f"Episode has no valid steps: {key_values}")
        collision = bool(valid["collision_event"].astype(bool).any())
        near_miss = bool((not collision) and valid["near_miss_condition"].astype(bool).any())
        critical = bool(valid["critical_condition"].astype(bool).any() or collision or near_miss)
        route_completed = bool((not collision) and valid["route_complete_event"].astype(bool).any())
        ttc = pd.to_numeric(valid["ttc_s"], errors="raise").to_numpy(float)
        ttc = ttc[np.isfinite(ttc) & (ttc >= 0.0)]
        jerk = np.abs(pd.to_numeric(valid["jerk_mps3"], errors="raise").to_numpy(float))
        if len(ttc) == 0:
            raise ValueError(f"Episode has no finite TTC values: {key_values}")
        if len(jerk) == 0:
            raise ValueError(f"Episode has no finite jerk values: {key_values}")
        jerk_p95 = float(np.quantile(jerk, 0.95, method="linear"))
        terminal_rows = valid.loc[valid["terminal"].astype(bool)]
        reason = str(terminal_rows.iloc[0]["termination_reason"]) if len(terminal_rows) else ""
        row = dict(zip(keys, key_values))
        row.update(
            {
                "n_steps": int(len(valid)),
                "collision": int(collision),
                "near_miss": int(near_miss),
                "critical_event": int(critical),
                "route_completed": int(route_completed),
                "ttc_p5": float(np.quantile(ttc, 0.05, method="linear")),
                "jerk_p95": jerk_p95,
                "termination_reason": reason,
                "dt_s": dt_s,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["replicate_id", "method", "scenario_cell", "episode_id"], kind="stable"
    ).reset_index(drop=True)


__all__ = ["EVENT_COLUMNS", "run_paired_rollouts", "derive_episode_metrics"]
