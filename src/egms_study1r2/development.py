"""Validation-only candidate development for exploratory Study 1-R2.

This module is deliberately unable to load or evaluate the final Study 1-R2
test seeds.  It retains every declared candidate and every development
replicate, including candidates that are adverse to Baseline B.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .common import load_protocol, sha256_file
from .generator import generate_offline_split, make_scenario_tape
from .models import MODEL_BASELINE_B, MODEL_STRUCTURED, multiclass_nll
from .rollout import derive_episode_metrics, run_paired_rollouts
from .training import train_paired_models


def _write_csv(frame: pd.DataFrame, path: Path, *, compressed: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {
        "index": False,
        "float_format": "%.17g",
        "lineterminator": "\n",
    }
    if compressed:
        with path.open("wb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
            ) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                    frame.to_csv(text, **options)
    else:
        frame.to_csv(path, **options)
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _candidate_protocol(
    protocol: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(protocol))
    training = dict(resolved["models"]["structured_r2_training"])
    for key in (
        "hidden_layer_sizes",
        "epochs",
        "learning_rate_init",
        "alpha",
    ):
        training[key] = candidate[key]
    training["candidate_id"] = str(candidate["candidate_id"])
    resolved["models"]["structured_r2_training"] = training
    return resolved


def _prepare_output(root: Path, request: Path, overwrite: bool) -> Path:
    allowed = (root / "outputs").resolve()
    output = request if request.is_absolute() else root / request
    output = output.resolve()
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError(f"Development output must be a named descendant of {allowed}")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Development output already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    return output


def _source_hashes(root: Path) -> dict[str, str]:
    candidates = [
        root / "configs" / "study1r2_development.yaml",
        *sorted((root / "src" / "egms_study1r2").glob("*.py")),
        *sorted((root / "src" / "egms_study1r").glob("*.py")),
    ]
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in candidates
        if path.is_file()
    }


def run_development(
    root: Path,
    config_request: Path,
    output_request: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate every declared candidate on development cohorts only."""

    config_path = (
        config_request if config_request.is_absolute() else root / config_request
    ).resolve()
    protocol = load_protocol(config_path)
    if protocol["project"]["protocol_status"] != "DEVELOPMENT_ONLY_NOT_FINAL_EVALUATION":
        raise ValueError("run_development accepts only the development protocol")
    dev = protocol.get("study1r2_development", {})
    if not bool(dev.get("final_evaluation_seeds_are_forbidden", False)):
        raise ValueError("Development protocol must explicitly forbid final seeds")
    candidates = list(dev.get("candidates", ()))
    n_replicates = int(dev.get("training_replicates_used_for_selection", 0))
    if not candidates or not 1 <= n_replicates <= 10:
        raise ValueError("Development requires candidates and 1..10 replicates")
    candidate_ids = [str(value["candidate_id"]) for value in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Development candidate IDs must be unique")

    output = _prepare_output(root, output_request, overwrite)
    started = datetime.now(timezone.utc)
    initial_hashes = _source_hashes(root)
    copied_config = output / "development_config_resolved.yaml"
    shutil.copy2(config_path, copied_config)
    protocol_sha256 = sha256_file(copied_config)

    validation = generate_offline_split(protocol, "validation")
    validation_path = _write_csv(
        validation, output / "generated" / "development_validation.csv.gz", compressed=True
    )
    rollout_tape = make_scenario_tape(protocol, "rollout")

    training_seeds = [int(v) for v in protocol["data"]["training_data_seeds"]]
    model_seeds = [int(v) for v in protocol["models"]["model_training_seeds"]]
    replicate_rows: list[dict[str, Any]] = []
    architecture_rows: list[pd.DataFrame] = []
    episode_frames: list[pd.DataFrame] = []

    for candidate_index, candidate in enumerate(candidates, start=1):
        candidate_id = str(candidate["candidate_id"])
        candidate_protocol = _candidate_protocol(protocol, candidate)
        print(
            f"candidate {candidate_index:02d}/{len(candidates):02d}: {candidate_id}",
            flush=True,
        )
        for replicate_id in range(n_replicates):
            print(f"  development replicate {replicate_id + 1:02d}/{n_replicates:02d}", flush=True)
            train = generate_offline_split(
                candidate_protocol, "train", training_replicate=replicate_id
            )
            train_path = output / "generated" / f"train_replicate_{replicate_id:02d}.csv.gz"
            if not train_path.exists():
                _write_csv(train, train_path, compressed=True)
            fitted = train_paired_models(
                train,
                validation,
                replicate_id=replicate_id,
                training_seed=model_seeds[replicate_id],
                config=candidate_protocol,
                output_root=output / "candidates" / candidate_id / "checkpoints",
            )
            architecture = fitted.architecture.copy()
            architecture.insert(0, "candidate_id", candidate_id)
            architecture.insert(1, "development_replicate", replicate_id)
            architecture_rows.append(architecture)

            method_metrics: dict[str, dict[str, float]] = {}
            labels = validation["y_true"].to_numpy(dtype=np.int64)
            for method in (MODEL_BASELINE_B, MODEL_STRUCTURED):
                model = fitted.models[method]
                probabilities = model.predict_proba(validation, calibrated=True)
                method_metrics[method] = {
                    "validation_nll": multiclass_nll(labels, probabilities),
                    "validation_macro_f1": float(
                        f1_score(
                            labels,
                            probabilities.argmax(axis=1),
                            labels=np.arange(4),
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }

            actions, _ = run_paired_rollouts(
                candidate_protocol,
                rollout_tape,
                fitted.models,
                replicate_id=replicate_id,
                training_seed=model_seeds[replicate_id],
                protocol_sha256=protocol_sha256,
            )
            episodes = derive_episode_metrics(actions, candidate_protocol)
            episodes.insert(0, "candidate_id", candidate_id)
            episode_frames.append(episodes)
            candidate_actions = output / "candidates" / candidate_id / "raw" / f"action_records_r{replicate_id:02d}.csv.gz"
            _write_csv(actions, candidate_actions, compressed=True)

            critical = episodes.groupby("method", sort=False)["critical_event"].mean()
            row: dict[str, Any] = {
                "candidate_id": candidate_id,
                "development_replicate": replicate_id,
                "training_data_seed": training_seeds[replicate_id],
                "model_training_seed": model_seeds[replicate_id],
                "development_validation_seed": int(protocol["data"]["fixed_validation_seed"]),
                "development_rollout_seed": int(protocol["data"]["fixed_rollout_seed"]),
                "train_data_sha256": sha256_file(train_path),
                "validation_data_sha256": sha256_file(validation_path),
                "baseline_validation_nll": method_metrics[MODEL_BASELINE_B]["validation_nll"],
                "structured_r2_validation_nll": method_metrics[MODEL_STRUCTURED]["validation_nll"],
                "baseline_validation_macro_f1": method_metrics[MODEL_BASELINE_B]["validation_macro_f1"],
                "structured_r2_validation_macro_f1": method_metrics[MODEL_STRUCTURED]["validation_macro_f1"],
                "baseline_critical_event_rate": float(critical.loc[MODEL_BASELINE_B]),
                "structured_r2_critical_event_rate": float(critical.loc[MODEL_STRUCTURED]),
            }
            replicate_rows.append(row)

    replicate_metrics = pd.DataFrame(replicate_rows)
    episode_metrics = pd.concat(episode_frames, ignore_index=True)
    architecture = pd.concat(architecture_rows, ignore_index=True)
    _write_csv(replicate_metrics, output / "candidate_replicate_metrics.csv")
    _write_csv(episode_metrics, output / "development_episode_metrics.csv.gz", compressed=True)
    _write_csv(architecture, output / "candidate_architecture.csv")

    ledger_rows: list[dict[str, Any]] = []
    for candidate_id, group in replicate_metrics.groupby("candidate_id", sort=False):
        architecture_group = architecture[
            (architecture["candidate_id"] == candidate_id)
            & (architecture["method"] == MODEL_STRUCTURED)
        ]
        baseline_critical = float(group["baseline_critical_event_rate"].mean())
        structured_critical = float(group["structured_r2_critical_event_rate"].mean())
        ledger_rows.append(
            {
                "candidate_id": candidate_id,
                "mean_baseline_validation_nll": float(group["baseline_validation_nll"].mean()),
                "mean_structured_r2_validation_nll": float(group["structured_r2_validation_nll"].mean()),
                "mean_validation_nll_difference": float(
                    (group["structured_r2_validation_nll"] - group["baseline_validation_nll"]).mean()
                ),
                "mean_baseline_macro_f1": float(group["baseline_validation_macro_f1"].mean()),
                "mean_structured_r2_macro_f1": float(group["structured_r2_validation_macro_f1"].mean()),
                "mean_macro_f1_difference": float(
                    (group["structured_r2_validation_macro_f1"] - group["baseline_validation_macro_f1"]).mean()
                ),
                "mean_baseline_critical_event_rate": baseline_critical,
                "mean_structured_r2_critical_event_rate": structured_critical,
                "mean_critical_event_rate_difference": structured_critical - baseline_critical,
                "safety_constraint_satisfied": structured_critical <= baseline_critical + 1.0e-12,
                "parameter_count": int(architecture_group["parameter_count"].iloc[0]),
            }
        )
    ledger = pd.DataFrame(ledger_rows)
    eligible = ledger[ledger["safety_constraint_satisfied"]].copy()
    if eligible.empty:
        raise RuntimeError(
            "No candidate satisfied the frozen development safety constraint; "
            "no candidate may be frozen."
        )
    selected = eligible.sort_values(
        [
            "mean_structured_r2_validation_nll",
            "mean_structured_r2_macro_f1",
            "parameter_count",
            "candidate_id",
        ],
        ascending=[True, False, True, True],
        kind="stable",
    ).iloc[0]
    selected_id = str(selected["candidate_id"])
    ledger["selected"] = ledger["candidate_id"].eq(selected_id)
    ledger_path = _write_csv(ledger, output / "candidate_ledger.csv")
    selected_spec = next(
        dict(value) for value in candidates if str(value["candidate_id"]) == selected_id
    )
    selection = {
        "schema": "egms-drive-study1r2-development-selection-1.0",
        "analysis_status": "post_hoc_exploratory_model_development",
        "selected_candidate_id": selected_id,
        "selected_candidate": selected_spec,
        "selection_rule": dev["selection_rule"],
        "development_replicates": n_replicates,
        "candidate_ledger_path": ledger_path.name,
        "candidate_ledger_sha256": sha256_file(ledger_path),
        "validation_data_sha256": sha256_file(validation_path),
        "final_test_or_rollout_accessed": False,
    }
    selection_path = _write_json(output / "selection.json", selection)

    final_hashes = _source_hashes(root)
    if initial_hashes != final_hashes:
        raise RuntimeError("Study 1-R2 source changed during development")
    finished = datetime.now(timezone.utc)
    manifest = {
        "schema": "egms-drive-study1r2-development-manifest-1.0",
        "project_id": protocol["project"]["id"],
        "analysis_status": "post_hoc_exploratory_model_development",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "development_protocol_sha256": protocol_sha256,
        "source_sha256": final_hashes,
        "candidate_ids": candidate_ids,
        "selection_sha256": sha256_file(selection_path),
        "selected_candidate_id": selected_id,
        "seed_firewall": {
            "development_training_data_seeds": training_seeds[:n_replicates],
            "development_model_training_seeds": model_seeds[:n_replicates],
            "development_validation_seed": int(protocol["data"]["fixed_validation_seed"]),
            "development_rollout_seed": int(protocol["data"]["fixed_rollout_seed"]),
            "final_evaluation_seeds_accessed": False,
        },
    }
    manifest_path = _write_json(output / "development_manifest.json", manifest)
    return {
        "status": "completed",
        "output": str(output),
        "selected_candidate_id": selected_id,
        "candidate_ledger": str(ledger_path),
        "development_manifest": str(manifest_path),
    }


__all__ = ["run_development"]
