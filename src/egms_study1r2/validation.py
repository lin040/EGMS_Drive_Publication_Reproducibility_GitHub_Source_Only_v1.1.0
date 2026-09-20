"""Direction-neutral integrity validator for exploratory Study 1-R2."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
import zlib

import numpy as np
import pandas as pd

from .common import ACTION_NAMES, load_protocol, sha256_file, softmax
from .models import MODEL_BASELINE_B, MODEL_STRUCTURED, Study1RClassifier
from .reporting import build_figure_inputs, build_table2_source
from .rollout import EVENT_COLUMNS, derive_episode_metrics
from .statistics import summarize_study1r


EXPECTED_EVIDENCE = "EXPLORATORY_CONTROLLED_SYNTHETIC_STUDY1R2_NOT_REAL_WORLD_SAFETY_EVIDENCE"
EXPECTED_METHODS = (MODEL_BASELINE_B, MODEL_STRUCTURED)


def _append_if(errors: list[str], condition: bool, message: str) -> None:
    if condition:
        errors.append(message)


def _valid_png(path: Path) -> bool:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return False
    position = 8
    while position + 12 <= len(data):
        length = int.from_bytes(data[position : position + 4], "big")
        kind = data[position + 4 : position + 8]
        payload_end = position + 8 + length
        chunk_end = payload_end + 4
        if chunk_end > len(data):
            return False
        expected = int.from_bytes(data[payload_end:chunk_end], "big")
        observed = zlib.crc32(kind + data[position + 8 : payload_end])
        if expected != observed:
            return False
        position = chunk_end
        if kind == b"IEND":
            return position == len(data)
    return False


def _compare_frames(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    key: list[str],
    label: str,
    atol: float = 1.0e-10,
) -> list[str]:
    errors: list[str] = []
    if set(observed.columns) != set(expected.columns):
        return [
            f"{label} columns differ; observed-only={sorted(set(observed.columns)-set(expected.columns))}, "
            f"expected-only={sorted(set(expected.columns)-set(observed.columns))}"
        ]
    left = observed.sort_values(key, kind="stable").reset_index(drop=True)
    right = expected.sort_values(key, kind="stable").reset_index(drop=True)
    if len(left) != len(right):
        return [f"{label} row count differs: {len(left)} versus {len(right)}"]
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
            a = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
            b = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
            if not np.allclose(a, b, atol=atol, rtol=0.0, equal_nan=True):
                errors.append(f"{label}.{column} fails raw recomputation")
        elif not left[column].fillna("").astype(str).equals(right[column].fillna("").astype(str)):
            errors.append(f"{label}.{column} differs after recomputation")
    return errors


def _validate_artifact_manifest(output: Path, errors: list[str]) -> int:
    path = output / "artifact_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("files", {})
    for relative, expected in records.items():
        candidate = output / relative
        _append_if(errors, not candidate.is_file(), f"Manifest file missing: {relative}")
        if candidate.is_file():
            _append_if(
                errors,
                candidate.stat().st_size != int(expected["bytes"]),
                f"Manifest byte count differs: {relative}",
            )
            _append_if(
                errors,
                sha256_file(candidate) != expected["sha256"],
                f"Manifest SHA-256 differs: {relative}",
            )
    return len(records)


def _method_blind_contract(protocol: dict, errors: list[str]) -> None:
    import egms_study1r2.generator as generator_module

    forbidden = ("method", "baseline", "structured", "advantage", "bonus", "penalty")
    generator_text = json.dumps(protocol.get("generator", {}), sort_keys=True).lower()
    _append_if(
        errors,
        any(token in generator_text for token in forbidden),
        "Generator configuration contains a method/performance-specific token",
    )
    for function in (
        generator_module.make_scenario_tape,
        generator_module.generate_offline_split,
        generator_module.observe_state,
        generator_module.oracle_actions,
        generator_module.step_dynamics,
    ):
        parameters = {name.lower() for name in inspect.signature(function).parameters}
        _append_if(
            errors,
            any("method" in name or "baseline" in name or "structured" in name for name in parameters),
            f"Method-blind generator API violation: {function.__name__}",
        )


def _reconstruct_event_records(action: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the sparse event ledger exclusively from persisted action rows."""

    keys = [
        "replicate_id",
        "training_seed",
        "method",
        "scenario_cell",
        "episode_seed",
        "episode_id",
    ]
    rows: list[dict[str, object]] = []
    for key_values, group in action.groupby(keys, sort=False):
        ordered = group.sort_values("frame_id", kind="stable")
        collision_seen = False
        near_seen = False
        critical_seen = False
        for record in ordered.itertuples(index=False):
            if not bool(record.valid_step):
                continue
            entries: list[tuple[str, float, float]] = []
            collision_now = bool(record.collision_event) and not collision_seen
            near_now = (
                bool(record.near_miss_condition)
                and not near_seen
                and not collision_seen
            )
            critical_now = bool(record.critical_condition) and not critical_seen
            if collision_now:
                entries.append(("collision", 1.0, 1.0))
            if near_now:
                entries.append(("near_miss_entry", 1.0, 1.0))
            if critical_now:
                entries.append(("critical_entry", 1.0, 1.0))
            if bool(record.route_complete_event):
                entries.append(
                    ("route_complete", float(record.progress_m), float(record.route_goal_m))
                )
            if bool(record.terminal) and str(record.termination_reason) == "timeout":
                entries.append(("timeout", float(record.progress_m), float(record.route_goal_m)))
            for event_type, value, threshold in entries:
                row = dict(zip(keys, key_values))
                row.update(
                    {
                        "frame_id": int(record.frame_id),
                        "time_s": float(record.time_s),
                        "event_type": event_type,
                        "event_value": value,
                        "threshold": threshold,
                        "source": "raw_rollout_recomputation",
                    }
                )
                rows.append(row)
            collision_seen |= bool(record.collision_event)
            near_seen |= bool(record.near_miss_condition)
            critical_seen |= bool(record.critical_condition)
    result = pd.DataFrame.from_records(rows, columns=EVENT_COLUMNS)
    if len(result):
        result = result.sort_values(
            ["replicate_id", "method", "scenario_cell", "episode_id", "frame_id", "event_type"],
            kind="stable",
        ).reset_index(drop=True)
    return result


def _validate_checkpoint_replay(
    output: Path,
    model_manifest: pd.DataFrame,
    offline: pd.DataFrame,
    action: pd.DataFrame,
    errors: list[str],
) -> None:
    """Reload every checkpoint and reproduce fixed-test and rollout logits."""

    test = pd.read_csv(output / "generated" / "test.csv.gz")
    if not test["sample_id"].is_unique:
        errors.append("Generated fixed-test sample IDs are not unique")
        return
    ordered_ids = test["sample_id"].astype(str).tolist()
    probability_columns = [f"p_{name}" for name in ACTION_NAMES]
    logit_columns = [f"logit_{name}" for name in ACTION_NAMES]
    for manifest_row in model_manifest.itertuples(index=False):
        replicate_id = int(manifest_row.replicate_id)
        method = str(manifest_row.method)
        label = f"checkpoint replay replicate={replicate_id}, method={method}"
        try:
            model = Study1RClassifier.load_checkpoint(
                output / str(manifest_row.checkpoint_path)
            )
            logits = model.predict_logits(test)
            probabilities = softmax(logits / float(model.temperature), axis=1)
        except Exception as exc:  # preserve all validation findings in one report
            errors.append(f"{label} could not be loaded/evaluated: {exc}")
            continue
        block = offline.loc[
            offline["replicate_id"].astype(int).eq(replicate_id)
            & offline["method"].astype(str).eq(method)
        ].copy()
        if len(block) != len(test) or not block["sample_id"].is_unique:
            errors.append(f"{label} has an incomplete or duplicate raw-output grid")
            continue
        block["sample_id"] = block["sample_id"].astype(str)
        block = block.set_index("sample_id").loc[ordered_ids]
        _append_if(
            errors,
            not np.allclose(
                block[logit_columns].to_numpy(float), logits, atol=1.0e-10, rtol=0.0
            ),
            f"{label} logits differ from archived raw output",
        )
        _append_if(
            errors,
            not np.allclose(
                block[probability_columns].to_numpy(float),
                probabilities,
                atol=1.0e-10,
                rtol=0.0,
            ),
            f"{label} probabilities differ from archived raw output",
        )
        _append_if(
            errors,
            not np.array_equal(block["y_pred"].to_numpy(int), probabilities.argmax(axis=1)),
            f"{label} predicted classes differ from archived raw output",
        )
        rollout_block = action.loc[
            action["replicate_id"].astype(int).eq(replicate_id)
            & action["method"].astype(str).eq(method)
        ].sort_values(["scenario_cell", "episode_id", "frame_id"], kind="stable")
        try:
            rollout_logits = model.predict_logits(rollout_block)
            rollout_probabilities = softmax(
                rollout_logits / float(model.temperature), axis=1
            )
        except Exception as exc:
            errors.append(f"{label} rollout replay failed: {exc}")
            continue
        _append_if(
            errors,
            not np.allclose(
                rollout_block[logit_columns].to_numpy(float),
                rollout_logits,
                atol=1.0e-10,
                rtol=0.0,
            ),
            f"{label} rollout logits differ from archived action records",
        )
        _append_if(
            errors,
            not np.allclose(
                rollout_block[probability_columns].to_numpy(float),
                rollout_probabilities,
                atol=1.0e-10,
                rtol=0.0,
            ),
            f"{label} rollout probabilities differ from archived action records",
        )


def validate_output(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    required = [
        output / "config_resolved.yaml",
        output / "run_manifest.json",
        output / "artifact_manifest.json",
        output / "manifests" / "split_manifest.csv",
        output / "manifests" / "seed_registry.csv",
        output / "manifests" / "model_manifest.csv",
        output / "logs" / "training_history.csv",
        output / "logs" / "model_architecture.csv",
        output / "raw" / "offline_frame_predictions.csv.gz",
        output / "raw" / "action_records.csv.gz",
        output / "raw" / "event_records.csv.gz",
        output / "raw" / "episode_metrics.csv.gz",
        output / "provenance" / "freeze_manifest.json",
        output / "provenance" / "development_manifest.json",
        output / "provenance" / "selection.json",
        output / "provenance" / "candidate_ledger.csv",
        output / "provenance" / "evaluation_access_record.json",
        output / "analysis" / "data" / "study1r2_metric_summary.csv",
        output / "analysis" / "data" / "study1r2_paired_contrasts.csv",
        output / "analysis" / "data" / "study1r2_replicate_effects.csv",
        output / "analysis" / "data" / "study1r2_bootstrap_draws.csv",
        output / "analysis" / "data" / "Figure_2B_Study1R2_Exploratory_inputs.csv",
        output / "analysis" / "tables" / "Table_2B_Study1R2_Exploratory.csv",
        output / "analysis" / "figures" / "Figure_2B_Study1R2_Exploratory.png",
        output / "analysis" / "figures" / "Figure_2B_Study1R2_Exploratory.pdf",
        output / "analysis" / "figures" / "Figure_2B_Study1R2_Exploratory.svg",
    ]
    errors = [f"Missing required file: {path.relative_to(output)}" for path in required if not path.is_file()]
    if errors:
        raise RuntimeError("Validation failed: " + "; ".join(errors))

    protocol = load_protocol(output / "config_resolved.yaml")
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    _append_if(errors, manifest.get("evidence_label") != EXPECTED_EVIDENCE, "Incorrect Study 1-R2 evidence label")
    _append_if(errors, manifest.get("analysis_status") != "post_hoc_exploratory_final_evaluation", "Incorrect R2 analysis status")
    _append_if(errors, int(manifest.get("final_training_replicates", -1)) != 10, "Exactly 10 final training replicates are required")
    _append_if(errors, manifest.get("methods") != list(EXPECTED_METHODS), "Run manifest method order differs")
    freeze_path = output / "provenance" / "freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    _append_if(errors, sha256_file(freeze_path) != manifest.get("freeze_manifest_sha256"), "Freeze manifest hash differs")
    _append_if(errors, freeze.get("final_source_sha256") != manifest.get("source_sha256"), "Run and freeze source sets differ")
    _append_if(errors, bool(freeze.get("validation_rule_requires_improvement")), "Freeze manifest incorrectly requires improvement")
    _append_if(errors, sha256_file(output / "config_resolved.yaml") != manifest.get("protocol_sha256"), "Resolved protocol hash differs")
    source_root = Path(__file__).resolve().parents[2]
    source_hashes = manifest.get("source_sha256", {})
    _append_if(errors, not isinstance(source_hashes, dict) or not source_hashes, "Run manifest lacks source hashes")
    if isinstance(source_hashes, dict):
        for relative, expected_hash in source_hashes.items():
            source_path = source_root / str(relative)
            _append_if(errors, not source_path.is_file(), f"Formal-run source file missing: {relative}")
            if source_path.is_file():
                _append_if(
                    errors,
                    sha256_file(source_path) != str(expected_hash),
                    f"Formal-run source SHA-256 differs: {relative}",
                )
    _method_blind_contract(protocol, errors)

    split = pd.read_csv(output / "manifests" / "split_manifest.csv")
    split_counts = split.groupby("split")["episode_id"].nunique().to_dict()
    _append_if(errors, split_counts.get("validation") != 90, "Validation split must contain 90 episodes")
    _append_if(errors, split_counts.get("test") != 360, "Offline test must contain 360 fixed episodes")
    _append_if(errors, split_counts.get("rollout") != 360, "Rollout test must contain 360 fixed episodes")
    train = split[split["split"].eq("train")]
    _append_if(errors, train["replicate_id"].nunique() != 10, "Training split lacks ten replicates")
    train_counts = train.groupby("replicate_id")["episode_id"].nunique()
    _append_if(errors, not train_counts.eq(180).all(), "Each training replicate must contain 180 episodes")
    validation_ids = set(split.loc[split["split"].eq("validation"), "episode_id"].astype(str))
    test_ids = set(split.loc[split["split"].eq("test"), "episode_id"].astype(str))
    rollout_ids = set(split.loc[split["split"].eq("rollout"), "episode_id"].astype(str))
    train_ids = set(train["episode_id"].astype(str))
    _append_if(errors, bool(train_ids & validation_ids) or bool(train_ids & test_ids), "Training episode IDs leak into validation/test")
    _append_if(errors, test_ids != rollout_ids, "Offline-test and rollout canonical episode IDs differ")
    for relative, group in split.groupby("relative_data_path"):
        path = output / str(relative)
        _append_if(errors, not path.is_file(), f"Split data file missing: {relative}")
        if path.is_file():
            _append_if(errors, group["data_sha256"].nunique() != 1 or group["data_sha256"].iloc[0] != sha256_file(path), f"Split data hash differs: {relative}")

    for replicate_id in range(10):
        train_path = output / "generated" / f"train_replicate_{replicate_id:02d}.csv.gz"
        if train_path.is_file():
            generated_train = pd.read_csv(train_path, usecols=["y_true"])
            _append_if(
                errors,
                set(generated_train["y_true"].astype(int)) != set(range(len(ACTION_NAMES))),
                f"Training replicate {replicate_id} lacks at least one action class",
            )

    tape_path = output / "generated" / "rollout_scenario_tape.npz"
    try:
        with np.load(tape_path, allow_pickle=False) as tape_payload:
            required_metadata = {
                "metadata_split",
                "metadata_data_seed",
                "metadata_training_replicate",
                "metadata_dt_s",
                "metadata_steps",
            }
            _append_if(
                errors,
                not required_metadata.issubset(set(tape_payload.files)),
                "Rollout tape lacks self-contained scalar metadata",
            )
    except Exception as exc:
        errors.append(f"Rollout tape cannot be loaded: {exc}")

    seeds = pd.read_csv(output / "manifests" / "seed_registry.csv")
    _append_if(errors, len(seeds) != 10 or seeds["replicate_id"].nunique() != 10, "Seed registry must contain ten unique replicate rows")
    _append_if(errors, seeds["training_data_seed"].nunique() != 10, "Training data seeds are not unique")
    model_manifest = pd.read_csv(output / "manifests" / "model_manifest.csv")
    _append_if(errors, len(model_manifest) != 20, "Model manifest must contain 20 fitted checkpoints")
    _append_if(errors, set(model_manifest["method"]) != set(EXPECTED_METHODS), "Model manifest methods differ")
    _append_if(errors, model_manifest.groupby("replicate_id")["training_data_sha256"].nunique().max() != 1, "Paired methods used different training data")
    _append_if(errors, model_manifest.drop_duplicates("replicate_id")["training_data_sha256"].nunique() != 10, "Training replicate data hashes are not unique")
    _append_if(errors, model_manifest["validation_data_sha256"].nunique() != 1, "Validation data hash is not fixed")
    for row in model_manifest.itertuples(index=False):
        for path_column, hash_column in (
            ("checkpoint_path", "checkpoint_sha256"),
            ("training_log_path", "training_log_sha256"),
            ("resolved_training_path", "resolved_training_sha256"),
            ("resolved_config_path", "resolved_config_sha256"),
        ):
            path = output / str(getattr(row, path_column))
            _append_if(errors, not path.is_file(), f"Model artifact missing: {path_column}={path}")
            if path.is_file():
                _append_if(errors, sha256_file(path) != str(getattr(row, hash_column)), f"Model artifact hash differs: {path}")

    history = pd.read_csv(output / "logs" / "training_history.csv")
    expected_history = 10 * (
        int(protocol["models"]["baseline_b_training"]["epochs"])
        + int(protocol["models"]["structured_r2_training"]["epochs"])
    )
    _append_if(errors, len(history) != expected_history, "Training history grid is incomplete")
    expected_epochs = {
        MODEL_BASELINE_B: int(protocol["models"]["baseline_b_training"]["epochs"]),
        MODEL_STRUCTURED: int(protocol["models"]["structured_r2_training"]["epochs"]),
    }
    for method, epochs in expected_epochs.items():
        counts = history.loc[history["method"].eq(method)].groupby("training_replicate")["epoch"].nunique()
        _append_if(errors, len(counts) != 10 or not counts.eq(epochs).all(), f"Training history epochs differ for {method}")
    _append_if(errors, not history.groupby(["training_replicate", "method"])["selected_checkpoint"].sum().eq(1).all(), "Each run must select exactly one checkpoint")

    offline = pd.read_csv(output / "raw" / "offline_frame_predictions.csv.gz")
    action = pd.read_csv(output / "raw" / "action_records.csv.gz")
    event = pd.read_csv(output / "raw" / "event_records.csv.gz")
    episode = pd.read_csv(output / "raw" / "episode_metrics.csv.gz")
    _append_if(errors, len(offline) != 10 * 2 * 360 * 16, "Offline prediction grid has the wrong row count")
    _append_if(errors, len(action) != 10 * 2 * 360 * 20, "Action record grid has the wrong row count")
    _append_if(errors, len(episode) != 10 * 2 * 360, "Episode metric grid has the wrong row count")
    expected_time = (action["frame_id"].to_numpy(float) + 1.0) * float(protocol["data"]["dt_s"])
    _append_if(
        errors,
        not np.allclose(action["time_s"].to_numpy(float), expected_time, atol=1.0e-12, rtol=0.0),
        "Action records do not use the declared post-step timestamps",
    )
    expected_critical = (
        action["collision_event"].astype(bool)
        | action["near_miss_condition"].astype(bool)
    )
    _append_if(
        errors,
        not np.array_equal(action["critical_condition"].astype(bool), expected_critical),
        "Action-row critical condition is not collision OR near miss",
    )

    _validate_checkpoint_replay(output, model_manifest, offline, action, errors)

    probability_columns = [f"p_{name}" for name in ACTION_NAMES]
    logit_columns = [f"logit_{name}" for name in ACTION_NAMES]
    for frame, name, prediction_column in ((offline, "offline", "y_pred"), (action, "action", "predicted_action")):
        p = frame[probability_columns].to_numpy(float)
        z = frame[logit_columns].to_numpy(float)
        temperature = frame["temperature"].to_numpy(float)
        expected_probability = softmax(z / temperature[:, None], axis=1)
        _append_if(errors, not np.isfinite(p).all() or not np.isfinite(z).all(), f"{name} logits/probabilities are non-finite")
        _append_if(errors, not np.allclose(p.sum(axis=1), 1.0, atol=1.0e-8, rtol=0.0), f"{name} probability rows do not sum to one")
        _append_if(errors, not np.allclose(p, expected_probability, atol=5.0e-8, rtol=0.0), f"{name} probabilities do not equal softmax(logits/temperature)")
        _append_if(errors, not np.array_equal(frame[prediction_column].to_numpy(int), p.argmax(axis=1)), f"{name} predictions differ from argmax")

    recomputed_episode = derive_episode_metrics(action, protocol)
    errors.extend(
        _compare_frames(
            episode,
            recomputed_episode,
            key=["replicate_id", "method", "scenario_cell", "episode_id"],
            label="episode_metrics",
        )
    )
    recomputed_event = _reconstruct_event_records(action)
    errors.extend(
        _compare_frames(
            event,
            recomputed_event,
            key=["replicate_id", "method", "scenario_cell", "episode_id", "frame_id", "event_type"],
            label="event_records",
        )
    )

    recomputed = summarize_study1r(
        offline,
        recomputed_episode,
        repetitions=int(protocol["statistics"]["bootstrap_repetitions"]),
        confidence=float(protocol["statistics"]["confidence_level"]),
        bootstrap_seed=int(protocol["statistics"]["bootstrap_seed"]),
    )
    saved = {
        "metric_summary": pd.read_csv(output / "analysis" / "data" / "study1r2_metric_summary.csv"),
        "paired_contrasts": pd.read_csv(output / "analysis" / "data" / "study1r2_paired_contrasts.csv"),
        "replicate_effects": pd.read_csv(output / "analysis" / "data" / "study1r2_replicate_effects.csv"),
        "bootstrap_draws": pd.read_csv(output / "analysis" / "data" / "study1r2_bootstrap_draws.csv"),
    }
    keys = {
        "metric_summary": ["endpoint", "method"],
        "paired_contrasts": ["endpoint"],
        "replicate_effects": ["endpoint", "replicate_id"],
        "bootstrap_draws": ["endpoint", "draw"],
    }
    for name in saved:
        errors.extend(_compare_frames(saved[name], recomputed[name], key=keys[name], label=name))

    figure_inputs = pd.read_csv(output / "analysis" / "data" / "Figure_2B_Study1R2_Exploratory_inputs.csv")
    expected_figure_inputs = build_figure_inputs(recomputed["metric_summary"], recomputed["paired_contrasts"])
    errors.extend(_compare_frames(figure_inputs, expected_figure_inputs, key=["endpoint"], label="figure_inputs"))
    table2 = pd.read_csv(output / "analysis" / "tables" / "Table_2B_Study1R2_Exploratory.csv", keep_default_na=False)
    expected_table2 = build_table2_source(recomputed["paired_contrasts"])
    errors.extend(_compare_frames(table2, expected_table2, key=["comparison_endpoint"], label="table2"))

    png = output / "analysis" / "figures" / "Figure_2B_Study1R2_Exploratory.png"
    _append_if(errors, not _valid_png(png), "Figure 2 PNG is invalid")
    if _valid_png(png):
        data = png.read_bytes()
        _append_if(errors, tuple(int.from_bytes(data[index:index+4], "big") for index in (16,20)) != (3810,2522), "Figure 2 PNG dimensions differ")
    pdf = png.with_suffix(".pdf")
    _append_if(errors, not pdf.read_bytes().startswith(b"%PDF") or b"%%EOF" not in pdf.read_bytes()[-2048:], "Figure 2 PDF is invalid")
    try:
        ElementTree.parse(png.with_suffix(".svg"))
    except ElementTree.ParseError:
        errors.append("Figure 2 SVG is invalid")

    manifest_records = _validate_artifact_manifest(output, errors)
    report = {
        "passed": not errors,
        "errors": errors,
        "evidence_label": manifest.get("evidence_label"),
        "analysis_status": "post_hoc_exploratory_final_evaluation",
        "final_training_replicates": int(offline["replicate_id"].nunique()),
        "fitted_checkpoints": int(len(model_manifest)),
        "fixed_test_episodes": int(offline["episode_id"].nunique()),
        "offline_prediction_rows": int(len(offline)),
        "action_record_rows": int(len(action)),
        "event_record_rows": int(len(event)),
        "episode_metric_rows": int(len(episode)),
        "artifact_manifest_records": manifest_records,
        "paired_contrast_rows": int(len(recomputed["paired_contrasts"])),
        "bootstrap_draw_rows": int(len(recomputed["bootstrap_draws"])),
        "raw_recomputation": "All ten endpoints and intervals recomputed from archived raw rows.",
        "checkpoint_replay": "All 20 checkpoints reloaded and reproduced full fixed-test and rollout logits.",
        "validation_rule_note": "No rule requires Structured-R2 to outperform Baseline B.",
    }
    (output / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if errors:
        raise RuntimeError("Validation failed: " + "; ".join(errors))
    return report


__all__ = ["validate_output"]
