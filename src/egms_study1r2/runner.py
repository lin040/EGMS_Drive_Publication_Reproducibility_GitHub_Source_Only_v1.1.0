"""Frozen final evaluation runner for post-hoc exploratory Study 1-R2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import io
import json
import os
import platform
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml

from .common import ACTION_NAMES, load_protocol, sha256_file, softmax
from .generator import generate_offline_split, make_scenario_tape
from .models import MODEL_BASELINE_B, MODEL_STRUCTURED
from .reporting import write_study1r2_reports
from .rollout import derive_episode_metrics, run_paired_rollouts
from .training import PairedTrainingResult, train_paired_models
from .validation import validate_output


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _write_csv(frame: pd.DataFrame, path: Path, *, compressed: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {
        "index": False,
        "float_format": "%.17g",
        "lineterminator": "\n",
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    if compressed:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                compresslevel=9,
                mtime=0,
            ) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                    frame.to_csv(text, **options)
        with gzip.open(temporary, "rb") as check:
            for _ in iter(lambda: check.read(1024 * 1024), b""):
                pass
    else:
        frame.to_csv(temporary, **options)
    temporary.replace(path)
    return path


def _source_hashes(root: Path) -> dict[str, str]:
    candidates = [
        root / "run_study1r2.py",
        root / "configs" / "study1r2_evaluation_frozen.yaml",
        root / "docs" / "STUDY1R2_EVALUATION_PROTOCOL_FROZEN.md",
        root / "docs" / "STUDY1R2_CLAIMS_BOUNDARY.md",
        root / "docs" / "STUDY1R2_DEVELOPMENT_LOG.md",
        root / "pyproject.toml",
        root / "requirements-lock.txt",
        *sorted((root / "src" / "egms_study1r2").glob("*.py")),
        root / "src" / "egms_study1r" / "generator.py",
    ]
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in candidates
        if path.is_file()
    }


def _prepare_output(root: Path, request: Path) -> Path:
    allowed = (root / "outputs").resolve()
    output = request if request.is_absolute() else root / request
    output = output.resolve()
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError(f"Output must be a named descendant of {allowed}")
    if output.is_symlink():
        raise ValueError("Refusing a symbolic-link output path")
    if output.exists():
        raise FileExistsError(f"Final Study 1-R2 output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


CANDIDATE_BEHAVIOUR_FILES = {
    "src/egms_study1r2/common.py",
    "src/egms_study1r2/generator.py",
    "src/egms_study1r2/models.py",
    "src/egms_study1r2/training.py",
    "src/egms_study1r2/rollout.py",
    "src/egms_study1r2/development.py",
}


def _seed_firewall(protocol: Mapping[str, Any]) -> dict[str, object]:
    expected = {
        "training_data_seeds": [8101, 8111, 8117, 8123, 8147, 8161, 8171, 8191, 8209, 8219],
        "model_training_seeds": [9103, 9127, 9133, 9151, 9173, 9181, 9199, 9209, 9221, 9239],
        "fixed_validation_seed": 31013,
        "fixed_test_seed": 31019,
        "fixed_rollout_seed": 31033,
        "bootstrap_seed": 171031,
    }
    observed = {
        "training_data_seeds": [int(v) for v in protocol["data"]["training_data_seeds"]],
        "model_training_seeds": [int(v) for v in protocol["models"]["model_training_seeds"]],
        "fixed_validation_seed": int(protocol["data"]["fixed_validation_seed"]),
        "fixed_test_seed": int(protocol["data"]["fixed_test_seed"]),
        "fixed_rollout_seed": int(protocol["data"]["fixed_rollout_seed"]),
        "bootstrap_seed": int(protocol["statistics"]["bootstrap_seed"]),
    }
    if observed != expected:
        raise ValueError("Final Study 1-R2 seeds differ from the frozen firewall")
    development = {
        5129, 5171, 5227, 5273, 5323, 6121, 6173, 6229, 6271, 6323,
        21013, 21017, 21019, 121031,
    }
    final_values = set(observed["training_data_seeds"]) | set(observed["model_training_seeds"])
    final_values |= {observed["fixed_validation_seed"], observed["fixed_test_seed"], observed["fixed_rollout_seed"], observed["bootstrap_seed"]}
    if final_values.intersection(development):
        raise ValueError("Development and final seed registries overlap")
    return observed


def build_freeze_manifest(
    root: Path,
    evaluation_config: Path,
    development_output: Path,
    destination: Path,
) -> Path:
    """Freeze the selected candidate and all final-evaluation source hashes."""

    if destination.exists():
        raise FileExistsError(f"Freeze manifest already exists: {destination}")
    protocol = load_protocol(evaluation_config)
    selection_path = development_output / "selection.json"
    development_manifest_path = development_output / "development_manifest.json"
    ledger_path = development_output / "candidate_ledger.csv"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    development_manifest = json.loads(development_manifest_path.read_text(encoding="utf-8"))
    selected = str(selection["selected_candidate_id"])
    if selected != "r2_c03" or selected != protocol["study1r2_evaluation"]["selected_candidate_id"]:
        raise ValueError("Evaluation config and development selection disagree")
    if sha256_file(ledger_path) != protocol["study1r2_evaluation"]["development_ledger_sha256"]:
        raise ValueError("Development ledger hash differs from the frozen config")
    current = _source_hashes(root)
    behaviour = {key: current[key] for key in sorted(CANDIDATE_BEHAVIOUR_FILES)}
    development_hashes = development_manifest["source_sha256"]
    if any(development_hashes.get(key) != value for key, value in behaviour.items()):
        raise ValueError("Candidate-behaviour source changed after development")
    payload = {
        "schema": "egms-drive-study1r2-freeze-manifest-1.0",
        "project_id": "EGMS_DRIVE_STUDY_1R2",
        "protocol_status": "FROZEN_AFTER_EXPLORATORY_DEVELOPMENT_BEFORE_FINAL_EVALUATION",
        "protocol_path": str(evaluation_config.relative_to(root)),
        "protocol_sha256": sha256_file(evaluation_config),
        "development_manifest_path": str(development_manifest_path),
        "development_manifest_sha256": sha256_file(development_manifest_path),
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "candidate_ledger_path": str(ledger_path),
        "candidate_ledger_sha256": sha256_file(ledger_path),
        "selected_candidate_id": selected,
        "selected_candidate_spec": selection["selected_candidate"],
        "candidate_behaviour_source_sha256": behaviour,
        "final_source_sha256": current,
        "seed_firewall": _seed_firewall(protocol),
        "original_study1r_generator_sha256": sha256_file(root / "src" / "egms_study1r" / "generator.py"),
        "r2_generator_sha256": sha256_file(root / "src" / "egms_study1r2" / "generator.py"),
        "post_hoc_after_study1r": True,
        "not_preregistered": True,
        "original_study1r_replaced": False,
        "final_evaluation_once_only": True,
        "validation_rule_requires_improvement": False,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
    }
    if payload["original_study1r_generator_sha256"] != payload["r2_generator_sha256"]:
        raise ValueError("Study 1-R2 generator source is not byte-identical to Study 1-R")
    return _json(destination, payload)


def _load_and_verify_freeze(root: Path, config_path: Path, freeze_path: Path) -> tuple[dict[str, Any], str]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != "egms-drive-study1r2-freeze-manifest-1.0":
        raise ValueError("Unsupported Study 1-R2 freeze manifest")
    if freeze.get("protocol_sha256") != sha256_file(config_path):
        raise ValueError("Frozen protocol hash mismatch")
    if freeze.get("final_source_sha256") != _source_hashes(root):
        raise ValueError("Final source differs from the frozen source set")
    if freeze.get("selected_candidate_id") != "r2_c03":
        raise ValueError("Unexpected frozen candidate")
    if any(not bool(freeze.get(k)) for k in ("post_hoc_after_study1r", "not_preregistered", "final_evaluation_once_only")):
        raise ValueError("Freeze manifest lacks required exploratory flags")
    if bool(freeze.get("original_study1r_replaced")) or bool(freeze.get("validation_rule_requires_improvement")):
        raise ValueError("Freeze manifest violates the claims or direction-neutral boundary")
    return freeze, sha256_file(freeze_path)


def _split_rows(
    frame: pd.DataFrame,
    *,
    relative_path: str,
    file_path: Path,
    replicate_id: int,
) -> list[dict[str, object]]:
    digest = sha256_file(file_path)
    size = file_path.stat().st_size
    rows = []
    for episode_id, group in frame.groupby("episode_id", sort=True):
        first = group.iloc[0]
        rows.append(
            {
                "split": str(first["split"]),
                "replicate_id": int(replicate_id),
                "scenario_cell": str(first["cell_id"]),
                "episode_seed": int(first["episode_seed"]),
                "episode_id": str(episode_id),
                "generator_seed": int(first["data_seed"]),
                "n_frames": int(len(group)),
                "relative_data_path": relative_path,
                "data_bytes": int(size),
                "data_sha256": digest,
            }
        )
    return rows


def _offline_predictions(
    test_frame: pd.DataFrame,
    fitted: PairedTrainingResult,
    *,
    replicate_id: int,
    training_seed: int,
    protocol_sha256: str,
) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "protocol_sha256": protocol_sha256,
            "split": "test",
            "replicate_id": int(replicate_id),
            "training_seed": int(training_seed),
            "scenario_cell": test_frame["cell_id"].astype(str),
            "episode_seed": test_frame["episode_seed"].astype(np.uint32),
            "episode_id": test_frame["episode_id"].astype(str),
            "frame_id": test_frame["frame"].astype(int),
            "time_s": test_frame["time_s"].astype(float),
            "sample_id": test_frame["sample_id"].astype(str),
            "y_true": test_frame["y_true"].astype(int),
        }
    )
    pieces = []
    for method in (MODEL_BASELINE_B, MODEL_STRUCTURED):
        model = fitted.results[method].model
        logits = model.predict_logits(test_frame)
        probabilities = softmax(logits / float(model.temperature), axis=1)
        predictions = probabilities.argmax(axis=1).astype(int)
        block = base.copy()
        block.insert(4, "method", method)
        block["y_pred"] = predictions
        block["y_true_name"] = np.asarray(ACTION_NAMES, dtype=object)[block["y_true"]]
        block["y_pred_name"] = np.asarray(ACTION_NAMES, dtype=object)[predictions]
        block["temperature"] = float(model.temperature)
        for class_index, action in enumerate(ACTION_NAMES):
            block[f"logit_{action}"] = logits[:, class_index]
            block[f"p_{action}"] = probabilities[:, class_index]
        pieces.append(block)
    return pd.concat(pieces, ignore_index=True)


def _save_rollout_tape(tape: Any, output: Path) -> tuple[Path, pd.DataFrame]:
    path = output / "generated" / "rollout_scenario_tape.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: value
        for name, value in vars(tape).items()
        if isinstance(value, np.ndarray)
    }
    arrays.update(
        {
            "metadata_split": np.asarray(tape.split),
            "metadata_data_seed": np.asarray(tape.data_seed, dtype=np.int64),
            "metadata_training_replicate": np.asarray(
                tape.training_replicate, dtype=np.int64
            ),
            "metadata_dt_s": np.asarray(tape.dt_s, dtype=np.float64),
            "metadata_steps": np.asarray(tape.steps, dtype=np.int64),
        }
    )
    np.savez_compressed(path, **arrays)
    manifest = pd.DataFrame(
        {
            "split": "rollout",
            "replicate_id": -1,
            "scenario_cell": tape.cell_id,
            "episode_seed": tape.episode_seed,
            "episode_id": tape.episode_id,
            "generator_seed": int(tape.data_seed),
            "n_frames": int(tape.steps),
            "relative_data_path": str(path.relative_to(output)),
            "data_bytes": int(path.stat().st_size),
            "data_sha256": sha256_file(path),
            "tape_content_sha256": tape.digest(),
        }
    )
    return path, manifest


def _model_manifest_rows(
    fitted: PairedTrainingResult,
    *,
    replicate_id: int,
    training_seed: int,
    training_data_path: Path,
    training_data_sha256: str,
    config_path: Path,
    output: Path,
) -> list[dict[str, object]]:
    architecture = fitted.architecture.set_index("method")
    rows = []
    for method in (MODEL_BASELINE_B, MODEL_STRUCTURED):
        result = fitted.results[method]
        checkpoint = result.checkpoint_path
        if checkpoint is None or result.checkpoint_sha256 is None:
            raise RuntimeError("Formal training did not persist its selected checkpoint")
        training_log = checkpoint.parent / "training_history.csv"
        resolved = checkpoint.parent / "resolved_training.json"
        row = architecture.loc[method]
        rows.append(
            {
                "replicate_id": int(replicate_id),
                "training_seed": int(training_seed),
                "method": method,
                "training_data_path": str(training_data_path.relative_to(output)),
                "training_data_sha256": training_data_sha256,
                "validation_data_sha256": "",
                "resolved_config_path": str(config_path.relative_to(output)),
                "resolved_config_sha256": sha256_file(config_path),
                "checkpoint_path": str(checkpoint.relative_to(output)),
                "checkpoint_bytes": int(checkpoint.stat().st_size),
                "checkpoint_sha256": result.checkpoint_sha256,
                "training_log_path": str(training_log.relative_to(output)),
                "training_log_sha256": sha256_file(training_log),
                "resolved_training_path": str(resolved.relative_to(output)),
                "resolved_training_sha256": sha256_file(resolved),
                "best_epoch": int(result.best_epoch),
                "best_validation_nll": float(result.best_validation_nll),
                "calibrated_validation_nll": float(result.calibrated_validation_nll),
                "temperature": float(result.model.temperature),
                "input_features": int(row["input_features"]),
                "hidden_layer_sizes": str(row["hidden_layer_sizes"]),
                "parameter_count": int(row["parameter_count"]),
            }
        )
    return rows


def _software_manifest() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "cpu_count": os.cpu_count(),
    }


def _artifact_manifest(output: Path) -> Path:
    excluded = {"artifact_manifest.json", "SHA256SUMS", "validation_report.json"}
    files = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name not in excluded
    )
    payload = {
        "schema": "egms-drive-study1r2-artifact-manifest-1.0",
        "hash_algorithm": "SHA-256",
        "files": {
            str(path.relative_to(output)): {
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in files
        },
    }
    return _json(output / "artifact_manifest.json", payload)


def _write_sha256sums(output: Path) -> Path:
    destination = output / "SHA256SUMS"
    files = sorted(path for path in output.rglob("*") if path.is_file() and path != destination)
    destination.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n" for path in files),
        encoding="utf-8",
        newline="\n",
    )
    return destination


def run_final(
    config_request: Path,
    freeze_manifest_request: Path,
    output_request: Path,
) -> dict[str, Any]:
    root = project_root()
    config_path = config_request if config_request.is_absolute() else root / config_request
    config_path = config_path.resolve()
    protocol = load_protocol(config_path)
    if protocol["project"]["protocol_status"] != "FROZEN_AFTER_EXPLORATORY_DEVELOPMENT_BEFORE_FINAL_EVALUATION":
        raise ValueError("run-final requires the frozen Study 1-R2 evaluation protocol")
    freeze_path = freeze_manifest_request if freeze_manifest_request.is_absolute() else root / freeze_manifest_request
    freeze_path = freeze_path.resolve()
    freeze, freeze_sha256 = _load_and_verify_freeze(root, config_path, freeze_path)
    seed_firewall = _seed_firewall(protocol)
    access_root = root / "outputs" / "study1r2_final_access" / freeze_sha256
    access_root.mkdir(parents=True, exist_ok=False)
    output = _prepare_output(root, output_request)
    started = datetime.now(timezone.utc)
    initial_source_hashes = _source_hashes(root)
    access_record = _json(
        access_root / "evaluation_access_record.json",
        {
            "freeze_manifest_sha256": freeze_sha256,
            "protocol_sha256": sha256_file(config_path),
            "requested_output": str(output),
            "accessed_utc": started.isoformat(),
            "source_sha256": initial_source_hashes,
            "final_seeds": seed_firewall,
            "results_not_yet_computed": True,
        },
    )
    resolved_config = output / "config_resolved.yaml"
    shutil.copy2(config_path, resolved_config)
    provenance = output / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    shutil.copy2(freeze_path, provenance / "freeze_manifest.json")
    for source_key, target_name in (
        ("development_manifest_path", "development_manifest.json"),
        ("selection_path", "selection.json"),
        ("candidate_ledger_path", "candidate_ledger.csv"),
    ):
        shutil.copy2(Path(freeze[source_key]), provenance / target_name)
    shutil.copy2(access_record, provenance / "evaluation_access_record.json")
    protocol_sha256 = sha256_file(resolved_config)

    print("[1/7] Generating fixed validation/test data and rollout tape", flush=True)
    validation = generate_offline_split(protocol, "validation")
    test = generate_offline_split(protocol, "test")
    validation_path = _write_csv(
        validation, output / "generated" / "validation.csv.gz", compressed=True
    )
    test_path = _write_csv(test, output / "generated" / "test.csv.gz", compressed=True)
    split_rows = []
    split_rows.extend(
        _split_rows(
            validation,
            relative_path=str(validation_path.relative_to(output)),
            file_path=validation_path,
            replicate_id=-1,
        )
    )
    split_rows.extend(
        _split_rows(
            test,
            relative_path=str(test_path.relative_to(output)),
            file_path=test_path,
            replicate_id=-1,
        )
    )
    rollout_tape = make_scenario_tape(protocol, "rollout")
    _, rollout_manifest = _save_rollout_tape(rollout_tape, output)
    split_rows.extend(rollout_manifest.to_dict("records"))

    model_seeds = [int(value) for value in protocol["models"]["model_training_seeds"]]
    training_data_seeds = [int(value) for value in protocol["data"]["training_data_seeds"]]
    all_offline: list[pd.DataFrame] = []
    all_actions: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []
    all_histories: list[pd.DataFrame] = []
    all_architecture: list[pd.DataFrame] = []
    model_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []

    print("[2/7] Training 10 paired Baseline B/Structured-R2 replicates", flush=True)
    for replicate_id, (data_seed, training_seed) in enumerate(
        zip(training_data_seeds, model_seeds)
    ):
        print(f"  replicate {replicate_id + 1:02d}/10", flush=True)
        train = generate_offline_split(protocol, "train", training_replicate=replicate_id)
        train_path = _write_csv(
            train,
            output / "generated" / f"train_replicate_{replicate_id:02d}.csv.gz",
            compressed=True,
        )
        train_hash = sha256_file(train_path)
        split_rows.extend(
            _split_rows(
                train,
                relative_path=str(train_path.relative_to(output)),
                file_path=train_path,
                replicate_id=replicate_id,
            )
        )
        fitted = train_paired_models(
            train,
            validation,
            replicate_id=replicate_id,
            training_seed=training_seed,
            config=protocol,
            output_root=output / "checkpoints",
        )
        history = fitted.history.copy()
        history["training_data_seed"] = data_seed
        all_histories.append(history)
        architecture = fitted.architecture.copy()
        architecture["replicate_id"] = replicate_id
        all_architecture.append(architecture)
        rows = _model_manifest_rows(
            fitted,
            replicate_id=replicate_id,
            training_seed=training_seed,
            training_data_path=train_path,
            training_data_sha256=train_hash,
            config_path=resolved_config,
            output=output,
        )
        for row in rows:
            row["validation_data_sha256"] = sha256_file(validation_path)
        model_rows.extend(rows)
        seed_rows.append(
            {
                "replicate_id": replicate_id,
                "training_data_seed": data_seed,
                "model_training_seed": training_seed,
                "baseline_initialization_seed": training_seed,
                "structured_initialization_seed": training_seed,
                "validation_data_seed": int(protocol["data"]["fixed_validation_seed"]),
                "offline_test_data_seed": int(protocol["data"]["fixed_test_seed"]),
                "rollout_tape_seed": int(protocol["data"]["fixed_rollout_seed"]),
                "bootstrap_seed": int(protocol["statistics"]["bootstrap_seed"]),
            }
        )
        all_offline.append(
            _offline_predictions(
                test,
                fitted,
                replicate_id=replicate_id,
                training_seed=training_seed,
                protocol_sha256=protocol_sha256,
            )
        )
        actions, events = run_paired_rollouts(
            protocol,
            rollout_tape,
            fitted.models,
            replicate_id=replicate_id,
            training_seed=training_seed,
            protocol_sha256=protocol_sha256,
        )
        all_actions.append(actions)
        all_events.append(events)

    print("[3/7] Writing frame-, action-, event-, and episode-level raw outputs", flush=True)
    offline = pd.concat(all_offline, ignore_index=True)
    actions = pd.concat(all_actions, ignore_index=True)
    events = pd.concat(all_events, ignore_index=True) if any(len(frame) for frame in all_events) else pd.DataFrame()
    episodes = derive_episode_metrics(actions, protocol)
    offline_path = _write_csv(
        offline,
        output / "raw" / "offline_frame_predictions.csv.gz",
        compressed=True,
    )
    action_path = _write_csv(
        actions, output / "raw" / "action_records.csv.gz", compressed=True
    )
    event_path = _write_csv(
        events, output / "raw" / "event_records.csv.gz", compressed=True
    )
    episode_path = _write_csv(
        episodes, output / "raw" / "episode_metrics.csv.gz", compressed=True
    )
    del all_offline, all_actions, all_events

    print("[4/7] Recomputing all metrics and paired 95% intervals from raw outputs", flush=True)
    report_paths = write_study1r2_reports(
        offline,
        episodes,
        output / "analysis",
        repetitions=int(protocol["statistics"]["bootstrap_repetitions"]),
        confidence=float(protocol["statistics"]["confidence_level"]),
        bootstrap_seed=int(protocol["statistics"]["bootstrap_seed"]),
    )

    split_manifest = pd.DataFrame(split_rows).sort_values(
        ["split", "replicate_id", "scenario_cell", "episode_id"], kind="stable"
    )
    _write_csv(split_manifest, output / "manifests" / "split_manifest.csv")
    _write_csv(pd.DataFrame(seed_rows), output / "manifests" / "seed_registry.csv")
    _write_csv(pd.DataFrame(model_rows), output / "manifests" / "model_manifest.csv")
    _write_csv(
        pd.concat(all_histories, ignore_index=True), output / "logs" / "training_history.csv"
    )
    _write_csv(
        pd.concat(all_architecture, ignore_index=True), output / "logs" / "model_architecture.csv"
    )

    finished = datetime.now(timezone.utc)
    final_source_hashes = _source_hashes(root)
    if final_source_hashes != initial_source_hashes:
        raise RuntimeError("Study 1-R2 source changed during the final run")
    run_manifest = {
        "schema": "egms-drive-study1r2-run-manifest-1.0",
        "project_id": protocol["project"]["id"],
        "evidence_label": protocol["project"]["evidence_label"],
        "protocol_status": protocol["project"]["protocol_status"],
        "protocol_sha256": protocol_sha256,
        "analysis_status": "post_hoc_exploratory_final_evaluation",
        "final_training_replicates": 10,
        "methods": [MODEL_BASELINE_B, MODEL_STRUCTURED],
        "training_data_seeds": training_data_seeds,
        "model_training_seeds": model_seeds,
        "fixed_test_episodes": int(test["episode_id"].nunique()),
        "fixed_rollout_episodes": int(rollout_tape.n_episodes),
        "rollout_tape_content_sha256": rollout_tape.digest(),
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "source_sha256": final_source_hashes,
        "freeze_manifest_path": "provenance/freeze_manifest.json",
        "freeze_manifest_sha256": freeze_sha256,
        "selected_candidate_id": freeze["selected_candidate_id"],
        "selected_candidate_spec": freeze["selected_candidate_spec"],
        "development_manifest_sha256": freeze["development_manifest_sha256"],
        "selection_sha256": freeze["selection_sha256"],
        "candidate_ledger_sha256": freeze["candidate_ledger_sha256"],
        "seed_firewall": seed_firewall,
        "post_hoc_after_study1r": True,
        "not_preregistered": True,
        "original_study1r_replaced": False,
        "capacity_asymmetry_disclosure": "Baseline B has the original head; Structured-R2 has the frozen development-selected larger dual-path head.",
        "software": _software_manifest(),
        "raw_files": {
            "offline": str(offline_path.relative_to(output)),
            "actions": str(action_path.relative_to(output)),
            "events": str(event_path.relative_to(output)),
            "episodes": str(episode_path.relative_to(output)),
        },
        "selection_rule": "One development-selected candidate was frozen before final evaluation. No final replicate, endpoint, seed, or candidate was selected, removed, or substituted based on final results.",
        "validation_rule_note": "No integrity rule requires Structured-R2 to outperform Baseline B.",
        "interpretation_boundary": (
            "Post-hoc exploratory controlled-synthetic Study 1-R2 only; not CARLA, public-dataset, real-vehicle, "
            "or empirical safety evidence."
        ),
    }
    _json(output / "run_manifest.json", run_manifest)
    _artifact_manifest(output)

    print("[5/7] Running direction-neutral integrity and raw-recomputation validation", flush=True)
    validation_report = validate_output(output)
    _write_sha256sums(output)
    print("[6/7] PASS: all formal artifacts validated", flush=True)
    print("[7/7] Figure 2 and manuscript source rows are ready", flush=True)
    return {
        "status": "completed",
        "output": str(output),
        "validation": validation_report,
        "figure_png": str(report_paths["figure2"]["png"]),
        "table2_csv": str(report_paths["table2"]["csv"]),
    }


run = run_final


def analyze_existing(output_request: Path) -> dict[str, Any]:
    root = project_root()
    output = output_request if output_request.is_absolute() else root / output_request
    output = output.resolve()
    protocol = load_protocol(output / "config_resolved.yaml")
    offline = pd.read_csv(output / "raw" / "offline_frame_predictions.csv.gz")
    episodes = pd.read_csv(output / "raw" / "episode_metrics.csv.gz")
    paths = write_study1r2_reports(
        offline,
        episodes,
        output / "analysis",
        repetitions=int(protocol["statistics"]["bootstrap_repetitions"]),
        confidence=float(protocol["statistics"]["confidence_level"]),
        bootstrap_seed=int(protocol["statistics"]["bootstrap_seed"]),
    )
    _artifact_manifest(output)
    report = validate_output(output)
    _write_sha256sums(output)
    return {"status": "reanalyzed", "validation": report, "figure2": str(paths["figure2"]["png"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    develop_parser = subparsers.add_parser("develop", help="Run validation-only R2 candidate development")
    develop_parser.add_argument("--config", type=Path, required=True)
    develop_parser.add_argument("--output", type=Path, required=True)
    develop_parser.add_argument("--overwrite", action="store_true")
    freeze_parser = subparsers.add_parser("freeze", help="Freeze one development-selected candidate")
    freeze_parser.add_argument("--config", type=Path, required=True)
    freeze_parser.add_argument("--development-output", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    run_parser = subparsers.add_parser("run-final", help="Execute the once-only final Study 1-R2 evaluation")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--freeze-manifest", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate", help="Validate a completed Study 1-R2")
    validate_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = project_root()
    if args.command == "develop":
        from .development import run_development
        result = run_development(root, args.config, args.output, overwrite=args.overwrite)
    elif args.command == "freeze":
        config = args.config if args.config.is_absolute() else root / args.config
        development = args.development_output if args.development_output.is_absolute() else root / args.development_output
        destination = args.output if args.output.is_absolute() else root / args.output
        result = {"status": "frozen", "freeze_manifest": str(build_freeze_manifest(root, config.resolve(), development.resolve(), destination.resolve()))}
    elif args.command == "run-final":
        result = run_final(args.config, args.freeze_manifest, args.output)
    else:
        output = args.output if args.output.is_absolute() else root / args.output
        result = validate_output(output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
