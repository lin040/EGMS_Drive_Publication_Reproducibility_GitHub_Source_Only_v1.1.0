from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import yaml

from .common import (
    expected_calibration_error,
    macro_f1,
    multiclass_brier,
    negative_log_likelihood,
    sha256,
    write_json,
)


EXPECTED_EVIDENCE = "CONTROLLED_SYNTHETIC_MECHANISM_VALIDATION_NOT_EMPIRICAL_DATASET"


def _valid_png(path: Path) -> bool:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return False
    position = 8
    while position + 12 <= len(data):
        length = int.from_bytes(data[position : position + 4], "big")
        chunk_type = data[position + 4 : position + 8]
        payload_end = position + 8 + length
        chunk_end = payload_end + 4
        if chunk_end > len(data):
            return False
        expected = int.from_bytes(data[payload_end:chunk_end], "big")
        observed = zlib.crc32(chunk_type + data[position + 8 : payload_end])
        if expected != observed:
            return False
        position = chunk_end
        if chunk_type == b"IEND":
            return position == len(data)
    return False


def _append_if(errors: list[str], condition: bool, message: str) -> None:
    if condition:
        errors.append(message)


def _allclose(frame: pd.DataFrame, columns: list[str], expected: np.ndarray, atol: float = 1e-9) -> bool:
    return np.allclose(frame[columns].to_numpy(float), expected, atol=atol, rtol=0.0, equal_nan=True)


def _validate_study2(output: Path, cfg: dict, errors: list[str]) -> dict[str, int]:
    seeds = list(cfg["common"]["training_seeds"])
    methods = list(cfg["study2"]["methods"])
    patterns = list(cfg["study2"]["missing_patterns"])
    sequence_length = int(cfg["study2"]["sequence_length"])
    test_scenes = int(cfg["study2"]["test_sequences"])
    stress_scenes = int(cfg["study2"]["stress_sequences"])
    bins = int(cfg["common"]["calibration_bins"])

    seed_metrics = pd.read_csv(output / "data" / "study2_seed_metrics.csv")
    _append_if(errors, len(seed_metrics) != len(seeds) * len(methods) * 2, "Study 2 seed-metric grid is incomplete")
    _append_if(errors, set(seed_metrics["training_seed"]) != set(seeds), "Study 2 training seeds differ from config")
    _append_if(errors, set(seed_metrics["method"]) != set(methods), "Study 2 methods differ from config")
    for column in ("macro_f1", "ece", "recall_at_1", "recall_at_5", "recall_at_10"):
        _append_if(errors, not seed_metrics[column].between(0, 1).all(), f"Study 2 {column} outside [0,1]")
    _append_if(
        errors,
        not np.isfinite(seed_metrics[["nll", "brier", "feature_drift"]].to_numpy(float)).all()
        or (seed_metrics[["nll", "brier", "feature_drift"]].to_numpy(float) < 0).any(),
        "Study 2 loss/prediction error is non-finite or negative",
    )

    predictions = pd.read_csv(output / "data" / "study2_predictions.csv.gz")
    key = ["training_seed", "method", "domain", "missing_pattern", "scene_id", "frame"]
    _append_if(errors, predictions.duplicated(key).any(), "Study 2 prediction keys are duplicated")
    expected_rows = len(seeds) * len(methods) * sequence_length * (
        test_scenes * len(patterns) + stress_scenes
    )
    _append_if(errors, len(predictions) != expected_rows, f"Study 2 predictions have {len(predictions)} rows, expected {expected_rows}")
    p_cols = ["p_KEEP", "p_SLOW", "p_YIELD", "p_STOP"]
    probabilities = predictions[p_cols].to_numpy(float)
    _append_if(errors, not np.isfinite(probabilities).all(), "Study 2 probabilities contain non-finite values")
    _append_if(errors, (probabilities < 0).any() or (probabilities > 1).any(), "Study 2 probability outside [0,1]")
    _append_if(errors, not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8), "Study 2 probability rows do not sum to one")
    _append_if(errors, not np.array_equal(predictions["y_pred"].to_numpy(int), probabilities.argmax(axis=1)), "Study 2 y_pred differs from argmax")
    _append_if(errors, set(predictions["y_true"].unique()) != set(range(4)), "Study 2 predictions lack an action class")

    group_counts = predictions.groupby(["training_seed", "method", "domain", "missing_pattern"]).agg(
        rows=("scene_id", "size"), scenes=("scene_id", "nunique"), frames=("frame", "nunique")
    )
    for seed in seeds:
        for method in methods:
            for pattern in patterns:
                expected = (test_scenes * sequence_length, test_scenes, sequence_length)
                observed = tuple(group_counts.loc[(seed, method, "clean_controlled", pattern)])
                _append_if(errors, observed != expected, f"Study 2 clean grid mismatch: {seed}/{method}/{pattern}")
            expected = (stress_scenes * sequence_length, stress_scenes, sequence_length)
            observed = tuple(group_counts.loc[(seed, method, "adverse_controlled_injected", "none")])
            _append_if(errors, observed != expected, f"Study 2 stress grid mismatch: {seed}/{method}")

    # Independently recompute predictive seed metrics from raw frame probabilities.
    metric_index = seed_metrics.set_index(["training_seed", "method", "domain"])
    for (seed, method, domain), group in predictions[predictions["missing_pattern"] == "none"].groupby(
        ["training_seed", "method", "domain"], sort=False
    ):
        truth = group["y_true"].to_numpy(int)
        p = group[p_cols].to_numpy(float)
        expected = np.array(
            [
                macro_f1(truth, p.argmax(axis=1), 4),
                negative_log_likelihood(truth, p),
                multiclass_brier(truth, p),
                expected_calibration_error(truth, p, bins),
            ]
        )
        observed = metric_index.loc[(seed, method, domain), ["macro_f1", "nll", "brier", "ece"]].to_numpy(float)
        _append_if(errors, not np.allclose(observed, expected, atol=1e-10), f"Study 2 seed metrics fail raw recomputation: {seed}/{method}/{domain}")

    missing = pd.read_csv(output / "data" / "study2_missing_modality.csv")
    _append_if(errors, len(missing) != len(seeds) * len(methods) * len(patterns), "Study 2 missing-modality grid incomplete")
    missing_index = missing.set_index(["training_seed", "method", "missing_pattern"])
    for (seed, method, pattern), group in predictions[predictions["domain"] == "clean_controlled"].groupby(
        ["training_seed", "method", "missing_pattern"], sort=False
    ):
        truth = group["y_true"].to_numpy(int)
        p = group[p_cols].to_numpy(float)
        expected = np.array(
            [
                macro_f1(truth, p.argmax(axis=1), 4),
                negative_log_likelihood(truth, p),
                multiclass_brier(truth, p),
                expected_calibration_error(truth, p, bins),
            ]
        )
        observed = missing_index.loc[(seed, method, pattern), ["macro_f1", "nll", "brier", "ece"]].to_numpy(float)
        _append_if(errors, not np.allclose(observed, expected, atol=1e-10), f"Study 2 missing metrics fail raw recomputation: {seed}/{method}/{pattern}")

    scene_metrics = pd.read_csv(output / "data" / "study2_scene_metrics.csv")
    missing_scene = pd.read_csv(output / "data" / "study2_missing_scene_metrics.csv")
    _append_if(errors, len(scene_metrics) != len(seeds) * len(methods) * (test_scenes + stress_scenes), "Study 2 scene-metric grid incomplete")
    _append_if(errors, len(missing_scene) != len(seeds) * len(methods) * len(patterns) * test_scenes, "Study 2 missing scene-metric grid incomplete")
    _append_if(errors, scene_metrics.duplicated(["training_seed", "method", "domain", "scene_id"]).any(), "Study 2 scene metrics contain duplicate keys")
    _append_if(errors, missing_scene.duplicated(["training_seed", "method", "missing_pattern", "scene_id"]).any(), "Study 2 missing scene metrics contain duplicate keys")
    scene_counts = scene_metrics.groupby(["training_seed", "method", "domain"])["scene_id"].agg(["size", "nunique"])
    for seed in seeds:
        for method in methods:
            for domain, expected_scenes in (("clean_controlled", test_scenes), ("adverse_controlled_injected", stress_scenes)):
                observed = tuple(scene_counts.loc[(seed, method, domain)])
                _append_if(errors, observed != (expected_scenes, expected_scenes), f"Study 2 scene grid mismatch: {seed}/{method}/{domain}")
    missing_counts = missing_scene.groupby(["training_seed", "method", "missing_pattern"])["scene_id"].agg(["size", "nunique"])
    _append_if(errors, not ((missing_counts["size"] == test_scenes) & (missing_counts["nunique"] == test_scenes)).all(), "Study 2 missing-scene Cartesian grid incomplete")
    drift_from_scene = scene_metrics.groupby(["training_seed", "method", "domain"])["feature_drift"].mean().sort_index()
    drift_reported = seed_metrics.set_index(["training_seed", "method", "domain"])["feature_drift"].sort_index()
    _append_if(errors, not np.allclose(drift_from_scene, drift_reported, atol=1e-10), "Study 2 feature prediction error fails scene aggregation")

    pair = pd.read_csv(output / "data" / "study2_pair_integrity.csv")
    passed = str(pair.loc[0, "pass_exact_latent_pairing"]).lower() == "true"
    _append_if(errors, not passed or float(pair.loc[0, "max_absolute_latent_difference"]) != 0.0, "Study 2 exact latent counterfactual pairing failed")

    split = pd.read_csv(output / "data" / "study2_split_manifest.csv")
    train = split[split["split"] == "train"]
    _append_if(errors, len(train) != len(seeds) or train["data_sha256"].nunique() != len(seeds), "Study 2 training replicates are not independently hashed")
    models = pd.read_csv(output / "data" / "study2_model_manifest.csv")
    _append_if(errors, len(models) != len(seeds) * len(methods), "Study 2 model manifest incomplete")
    _append_if(errors, models["model_sha256"].str.fullmatch(r"[0-9a-f]{64}").eq(False).any(), "Study 2 checkpoint hash malformed")
    _append_if(errors, models.groupby("training_seed")["training_data_sha256"].nunique().max() != 1, "Study 2 methods within replicate used different training data")
    _append_if(errors, models.drop_duplicates("training_seed")["training_data_sha256"].nunique() != len(seeds), "Study 2 training replicate hashes are not unique")
    return {"study2_prediction_rows": len(predictions), "study2_seed_metric_rows": len(seed_metrics), "study2_model_records": len(models)}


def _validate_study3(output: Path, cfg: dict, errors: list[str]) -> dict[str, int]:
    seeds = list(cfg["common"]["training_seeds"])
    methods = list(cfg["study3"]["methods"])
    scenes = int(cfg["study3"]["test_scenes"])
    max_modes = int(cfg["study3"]["modes"])
    miss_threshold = float(cfg["study3"]["miss_threshold_m"])

    seed_metrics = pd.read_csv(output / "data" / "study3_seed_metrics.csv")
    _append_if(errors, len(seed_metrics) != len(seeds) * len(methods), "Study 3 seed-metric grid incomplete")
    _append_if(errors, set(seed_metrics["training_seed"]) != set(seeds), "Study 3 training seeds differ from config")
    _append_if(errors, set(seed_metrics["method"]) != set(methods), "Study 3 methods differ from config")
    predictions = pd.read_csv(output / "data" / "study3_predictions.csv.gz")
    key = ["training_seed", "method", "scene_id"]
    _append_if(errors, predictions.duplicated(key).any(), "Study 3 prediction keys are duplicated")
    expected_rows = len(seeds) * len(methods) * scenes
    _append_if(errors, len(predictions) != expected_rows, f"Study 3 predictions have {len(predictions)} rows, expected {expected_rows}")
    counts = predictions.groupby(["training_seed", "method"])["scene_id"].agg(["size", "nunique"])
    _append_if(errors, not ((counts["size"] == scenes) & (counts["nunique"] == scenes)).all(), "Study 3 per-condition scene grid incomplete")

    intent_cols = [f"intent_p_{index}" for index in range(8)]
    intent_p = predictions[intent_cols].to_numpy(float)
    _append_if(errors, not np.isfinite(intent_p).all() or (intent_p < 0).any() or (intent_p > 1).any(), "Study 3 intent probabilities invalid")
    _append_if(errors, not np.allclose(intent_p.sum(axis=1), 1.0, atol=1e-8), "Study 3 intent probabilities do not sum to one")
    _append_if(errors, not np.array_equal(predictions["intent_pred"].to_numpy(int), intent_p.argmax(axis=1)), "Study 3 intent_pred differs from argmax")

    p_cols = [f"p_mode_{index}" for index in range(1, max_modes + 1)]
    ade_cols = [f"ade_mode_{index}" for index in range(1, max_modes + 1)]
    fde_cols = [f"fde_mode_{index}" for index in range(1, max_modes + 1)]
    p_all = predictions[p_cols].to_numpy(float)
    ade_all = predictions[ade_cols].to_numpy(float)
    fde_all = predictions[fde_cols].to_numpy(float)
    recomputed = {name: np.empty(len(predictions), dtype=float) for name in ["minADE_K", "minFDE_K", "MR_K", "Brier-minFDE_K", "ADE_1", "FDE_1", "MR_1"]}
    for k in (1, max_modes):
        mask = predictions["K"].to_numpy(int) == k
        p = p_all[mask, :k]
        ade = ade_all[mask, :k]
        fde = fde_all[mask, :k]
        _append_if(errors, not np.isfinite(p).all() or not np.isfinite(ade).all() or not np.isfinite(fde).all(), f"Study 3 K={k} mode outputs non-finite")
        _append_if(errors, (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-8), f"Study 3 K={k} mode probabilities invalid")
        if k == 1:
            _append_if(errors, not np.isnan(p_all[mask, 1:]).all() or not np.isnan(ade_all[mask, 1:]).all() or not np.isnan(fde_all[mask, 1:]).all(), "Study 3 K=1 trailing mode fields must be N/A")
        best = fde.argmin(axis=1)
        row = np.arange(len(best))
        top = p.argmax(axis=1)
        recomputed["minADE_K"][mask] = ade[row, best]
        recomputed["minFDE_K"][mask] = fde[row, best]
        recomputed["MR_K"][mask] = (fde[row, best] > miss_threshold).astype(float)
        recomputed["Brier-minFDE_K"][mask] = fde[row, best] + (1.0 - p[row, best]) ** 2
        recomputed["ADE_1"][mask] = ade[row, top]
        recomputed["FDE_1"][mask] = fde[row, top]
        recomputed["MR_1"][mask] = (fde[row, top] > miss_threshold).astype(float)
        _append_if(errors, not np.array_equal(predictions.loc[mask, "best_fde_mode"].to_numpy(int), best + 1), f"Study 3 K={k} best-FDE mode index incorrect")
    for name, expected in recomputed.items():
        _append_if(errors, not np.allclose(predictions[name].to_numpy(float), expected, atol=1e-10), f"Study 3 {name} fails raw mode-level recomputation")

    # Recompute all seed-level endpoints from per-scene raw predictions.
    grouped = predictions.groupby(["training_seed", "method"], as_index=False)
    expected_rows_list = []
    for (seed, method), group in grouped:
        row = {"training_seed": seed, "method": method}
        row["intent_macro_f1"] = macro_f1(group["intent_true"], group["intent_pred"], 8)
        for name in recomputed:
            row[name] = float(group[name].mean())
        expected_rows_list.append(row)
    expected_frame = pd.DataFrame(expected_rows_list).set_index(["training_seed", "method"]).sort_index()
    observed_frame = seed_metrics.set_index(["training_seed", "method"]).sort_index()
    compare = ["intent_macro_f1", *recomputed.keys()]
    _append_if(errors, not np.allclose(observed_frame[compare], expected_frame[compare], atol=1e-10), "Study 3 seed metrics fail raw recomputation")

    single = seed_metrics[seed_metrics["method"] == "Single mode K=1"]
    reportable = ["minADE_6_reportable", "minFDE_6_reportable", "MR_6_reportable", "Brier-minFDE_6_reportable"]
    _append_if(errors, not single[reportable].isna().all().all(), "K=1 condition is mislabeled with K=6 metrics")
    multi = seed_metrics[seed_metrics["method"] != "Single mode K=1"]
    _append_if(errors, multi[reportable].isna().any().any(), "A K=6 condition lacks reportable K=6 metrics")

    split = pd.read_csv(output / "data" / "study3_split_manifest.csv")
    train = split[split["split"] == "train"]
    _append_if(errors, len(train) != len(seeds) or train["data_sha256"].nunique() != len(seeds), "Study 3 training replicates are not independently hashed")
    models = pd.read_csv(output / "data" / "study3_model_manifest.csv")
    _append_if(errors, len(models) != len(seeds) * len(methods), "Study 3 model manifest incomplete")
    _append_if(errors, models["model_sha256"].str.fullmatch(r"[0-9a-f]{64}").eq(False).any(), "Study 3 checkpoint hash malformed")
    _append_if(errors, models.groupby("training_seed")["training_data_sha256"].nunique().max() != 1, "Study 3 methods within replicate used different training data")
    _append_if(errors, models.drop_duplicates("training_seed")["training_data_sha256"].nunique() != len(seeds), "Study 3 training replicate hashes are not unique")
    parameter_pivot = models[models["method"].isin(["Full temporal graph", "No predicted-intent gating"])].pivot(index="training_seed", columns="method", values="parameter_sha256")
    _append_if(errors, not (parameter_pivot["Full temporal graph"] == parameter_pivot["No predicted-intent gating"]).all(), "No-gating intervention does not share Full fitted parameters")
    inference_pivot = models[models["method"].isin(["Full temporal graph", "No predicted-intent gating"])].pivot(index="training_seed", columns="method", values="inference_rule_sha256")
    _append_if(errors, (inference_pivot["Full temporal graph"] == inference_pivot["No predicted-intent gating"]).any(), "No-gating inference rule hash equals Full")
    return {"study3_prediction_rows": len(predictions), "study3_seed_metric_rows": len(seed_metrics), "study3_model_records": len(models)}


def validate_output(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    required = [
        output / "config_resolved.yaml",
        output / "run_manifest.json",
        output / "summary.json",
        output / "MANUSCRIPT_RESULTS_CONTROLLED_SYNTHETIC.md",
        output / "data" / "study2_seed_metrics.csv",
        output / "data" / "study2_predictions.csv.gz",
        output / "data" / "study2_scene_metrics.csv",
        output / "data" / "study2_missing_modality.csv",
        output / "data" / "study2_missing_scene_metrics.csv",
        output / "data" / "study2_model_manifest.csv",
        output / "data" / "study3_seed_metrics.csv",
        output / "data" / "study3_predictions.csv.gz",
        output / "data" / "study3_model_manifest.csv",
        output / "tables" / "table_execution_status.csv",
        output / "tables" / "table_s2_primary_contrasts.csv",
        output / "tables" / "table_s3_primary_contrasts.csv",
    ]
    errors = [f"Missing required file: {path}" for path in required if not path.is_file()]
    if errors:
        raise RuntimeError("; ".join(errors))
    with (output / "config_resolved.yaml").open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    _append_if(errors, manifest.get("evidence_label") != EXPECTED_EVIDENCE, "Incorrect controlled-synthetic evidence label")
    _append_if(errors, manifest.get("carla_executed") is not False, "CARLA executed flag must be false")
    _append_if(errors, manifest.get("real_datasets_executed") is not False, "Real-dataset executed flag must be false")
    formal_seeds = manifest.get("training_replicate_seeds", [])
    _append_if(errors, len(formal_seeds) != 10 or len(set(formal_seeds)) != 10, "Exactly 10 unique formal training replicate seeds required")
    _append_if(errors, bool(set(formal_seeds) & set(manifest.get("pilot_training_seeds", []))), "Pilot and final full-run seeds overlap")
    _append_if(errors, manifest.get("preregistered") is not False, "This run must not claim preregistration")

    status = pd.read_csv(output / "tables" / "table_execution_status.csv")
    named = status[status["source"].isin(["CARLA_0.9.16", "nuScenes", "RADIATE", "Argoverse_2_Motion"])]
    _append_if(errors, not named["status"].str.startswith("N/A").all(), "One or more named sources are not N/A")
    _append_if(errors, (named["scenes_or_logs"].astype(str) != "0").any(), "Named sources must have zero executed scenes/logs")

    counts = {}
    counts.update(_validate_study2(output, cfg, errors))
    counts.update(_validate_study3(output, cfg, errors))

    for table_name in ("table_s2_primary_contrasts.csv", "table_s3_primary_contrasts.csv"):
        contrast = pd.read_csv(output / "tables" / table_name)
        expected_support = contrast["direction_ci_consistency_met"].astype(bool) & (contrast["holm_adjusted_p"] < 0.05)
        _append_if(errors, not np.array_equal(contrast["mechanism_support_rule_met"].astype(bool), expected_support), f"{table_name} support flag ignores Holm correction")

    figure_files = sorted((output / "figures").glob("figure_*.*"))
    _append_if(errors, len(figure_files) != 27, f"Expected 27 figure files, found {len(figure_files)}")
    _append_if(errors, any(path.stat().st_size == 0 for path in figure_files), "At least one figure is zero bytes")
    for path in figure_files:
        if path.suffix == ".png" and not _valid_png(path):
            errors.append(f"Invalid PNG: {path.name}")
        elif path.suffix == ".pdf":
            data = path.read_bytes()
            if not (data.startswith(b"%PDF-") and b"%%EOF" in data[-1024:]):
                errors.append(f"Invalid PDF: {path.name}")
        elif path.suffix == ".svg":
            try:
                ElementTree.parse(path)
            except ElementTree.ParseError:
                errors.append(f"Invalid SVG: {path.name}")

    file_hashes = {
        str(path.relative_to(output)): sha256(path)
        for path in [*required, *figure_files]
        if path.is_file()
    }
    report = {
        "passed": not errors,
        "errors": errors,
        "evidence_label": manifest.get("evidence_label"),
        "formal_training_replicate_seeds": formal_seeds,
        **counts,
        "fitted_model_records": counts["study2_model_records"] + counts["study3_model_records"],
        "figure_files": len(figure_files),
        "file_sha256": file_hashes,
        "validation_rule_note": "No rule requires the Full condition to win.",
        "raw_recomputation": "Study 2 predictive metrics and all Study 3 trajectory metrics recomputed from raw saved probabilities/errors.",
    }
    write_json(output / "validation_report.json", report)
    if errors:
        raise RuntimeError("Validation failed: " + "; ".join(errors))
    return report
