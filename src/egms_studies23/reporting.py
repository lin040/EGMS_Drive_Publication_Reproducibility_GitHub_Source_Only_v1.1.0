from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from .common import holm_adjust, seed_summary, stable_seed


EVIDENCE_NOTE = (
    "Controlled synthetic mechanism validation only; not CARLA, nuScenes, "
    "RADIATE, Argoverse 2, real-weather, or real-world evidence."
)


def markdown_table(frame: pd.DataFrame) -> str:
    values = frame.copy()
    for column in values.columns:
        values[column] = values[column].map(_format_markdown_value)
    header = "| " + " | ".join(map(str, values.columns)) + " |"
    separator = "| " + " | ".join("---" for _ in values.columns) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in values.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _format_markdown_value(value: object) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def write_table(frame: pd.DataFrame, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{stem}.csv", index=False)
    (output_dir / f"{stem}.md").write_text(markdown_table(frame), encoding="utf-8")
    (output_dir / f"{stem}.tex").write_text(_latex_table(frame), encoding="utf-8")


def _latex_escape(value: object) -> str:
    text = _format_markdown_value(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _latex_table(frame: pd.DataFrame) -> str:
    alignment = "l" * len(frame.columns)
    lines = [f"\\begin{{tabular}}{{{alignment}}}", r"\hline"]
    lines.append(" & ".join(_latex_escape(column) for column in frame.columns) + r" \\")
    lines.append(r"\hline")
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(_latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def _summarize_long(
    frame: pd.DataFrame,
    group_columns: list[str],
    metrics: list[tuple[str, str, str]],
    confidence: float = 0.95,
) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(group_columns, keys))
        for column, label, better in metrics:
            estimate, low, high = seed_summary(group[column].to_numpy(float), confidence)
            rows.append(
                {
                    **identity,
                    "metric": label,
                    "better": better,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "training_replicates": group["training_seed"].nunique(),
                    "ci_definition": "95% t CI across independent training-sample replicates; fixed test scenes",
                    "evidence": EVIDENCE_NOTE,
                }
            )
    return pd.DataFrame(rows)


def _summarize_replicate_bootstrap_long(
    frame: pd.DataFrame,
    group_columns: list[str],
    metrics: list[tuple[str, str, str]],
    confidence: float,
    repetitions: int,
) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(group_columns, keys))
        for column, label, better in metrics:
            values = group[column].to_numpy(float)
            values = values[np.isfinite(values)]
            rng = np.random.default_rng(stable_seed("replicate_bootstrap", *keys, column))
            draws = np.median(
                rng.choice(values, size=(int(repetitions), len(values)), replace=True),
                axis=1,
            )
            alpha = 1.0 - confidence
            low, high = np.quantile(draws, [alpha / 2, 1.0 - alpha / 2])
            rows.append(
                {
                    **identity,
                    "metric": label,
                    "better": better,
                    "estimate": float(np.median(values)),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "training_replicates": group["training_seed"].nunique(),
                    "ci_definition": f"Median across training-replicate latency summaries with {int(confidence * 100)}% percentile bootstrap CI ({repetitions} draws)",
                    "evidence": EVIDENCE_NOTE,
                }
            )
    return pd.DataFrame(rows)


def study2_tables(results: dict[str, pd.DataFrame], cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    confidence = float(cfg["common"]["confidence_level"]) if cfg else 0.95
    repetitions = int(cfg["common"]["bootstrap_repetitions"]) if cfg else 10000
    seed_metrics = results["study2_seed_metrics"]
    main = _summarize_long(
        seed_metrics,
        ["method", "domain"],
        [
            ("macro_f1", "Action macro-F1", "higher"),
            ("nll", "NLL", "lower"),
            ("brier", "Multiclass Brier (0-2)", "lower"),
            ("ece", "ECE (15 equal-mass bins)", "lower"),
            ("sas", "SAS (Eq. 40)", "higher"),
            ("recall_at_1", "Bidirectional Recall@1", "higher"),
            ("recall_at_5", "Bidirectional Recall@5", "higher"),
            ("recall_at_10", "Bidirectional Recall@10", "higher"),
            ("feature_drift", "Held-out one-step feature prediction error", "lower"),
            ("representation_variance", "Representation variance", "diagnostic"),
        ], confidence,
    )
    missing = _summarize_long(
        results["study2_missing_modality"],
        ["method", "missing_pattern"],
        [
            ("macro_f1", "Action macro-F1", "higher"),
            ("macro_f1_absolute_drop", "Absolute macro-F1 drop", "lower"),
            ("macro_f1_relative_drop", "Relative macro-F1 drop", "lower"),
            ("nll", "NLL", "lower"),
            ("brier", "Multiclass Brier (0-2)", "lower"),
            ("ece", "ECE", "lower"),
        ], confidence,
    )
    latency = _summarize_replicate_bootstrap_long(
        results["study2_latency"],
        ["method"],
        [("p50_ms", "Batch-1 CPU latency P50 (ms)", "lower"), ("p95_ms", "Batch-1 CPU latency P95 (ms)", "lower")], confidence, repetitions,
    )
    contrasts = _study2_contrasts(results, cfg)
    return {
        "table_s2_main": main,
        "table_s2_missing_modality": missing,
        "table_s2_primary_contrasts": contrasts,
        "table_s2_latency": latency,
    }


def _paired_contrast(
    full: pd.Series,
    ablation: pd.Series,
    comparison: str,
    endpoint: str,
    favorable_sign: str,
    confidence: float = 0.95,
) -> dict[str, object]:
    joined = pd.concat([full.rename("full"), ablation.rename("ablation")], axis=1).dropna()
    difference = joined["full"] - joined["ablation"]
    difference_values = difference.to_numpy(float)
    estimate, low, high = seed_summary(difference_values, confidence)
    p_value = _exact_sign_flip_p(difference_values)
    if favorable_sign == "negative":
        favorable = difference < 0
        direction_met = bool(high < 0 and favorable.sum() >= math.ceil(0.8 * len(difference)))
    else:
        favorable = difference > 0
        direction_met = bool(low > 0 and favorable.sum() >= math.ceil(0.8 * len(difference)))
    return {
        "comparison": comparison,
        "primary_endpoint": endpoint,
        "difference_full_minus_ablation": estimate,
        "ci_low": low,
        "ci_high": high,
        "favorable_sign": favorable_sign,
        "favorable_training_replicates": int(favorable.sum()),
        "total_training_replicates": len(difference),
        "paired_exact_sign_flip_p": p_value,
        "ci_method": "95% t CI across paired training replicates; fixed test scenes",
        "direction_ci_consistency_met": direction_met,
        "evidence": EVIDENCE_NOTE,
    }


def _exact_sign_flip_p(difference: np.ndarray) -> float:
    values = np.asarray(difference, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    combinations = np.arange(2 ** len(values), dtype=np.uint64)[:, None]
    bits = (combinations >> np.arange(len(values), dtype=np.uint64)) & 1
    signs = 1.0 - 2.0 * bits
    distribution = (signs * values[None, :]).mean(axis=1)
    return float(np.mean(np.abs(distribution) >= abs(values.mean()) - 1.0e-15))


def _paired_crossed_contrast(
    frame: pd.DataFrame,
    full_method: str,
    ablation_method: str,
    metric: str,
    comparison: str,
    endpoint: str,
    favorable_sign: str,
    repetitions: int,
    random_seed: int,
    confidence: float,
) -> dict[str, object]:
    grouped = frame.groupby(["method", "training_seed", "scene_id"], as_index=False)[metric].mean()
    full = grouped[grouped["method"] == full_method].set_index(["training_seed", "scene_id"])[metric]
    ablation = grouped[grouped["method"] == ablation_method].set_index(["training_seed", "scene_id"])[metric]
    full = full.sort_index()
    ablation = ablation.sort_index()
    if not full.index.equals(ablation.index):
        raise ValueError(f"Full and ablation replicate-scene indices differ for {comparison}")
    joined = pd.concat([full.rename("full"), ablation.rename("ablation")], axis=1)
    if joined.isna().any().any():
        raise ValueError(f"Missing paired replicate-scene value for {comparison}")
    matrix = (joined["full"] - joined["ablation"]).unstack("scene_id").sort_index()
    if matrix.isna().any().any():
        raise ValueError(f"Incomplete replicate-scene grid for {comparison}")
    values = matrix.to_numpy(float)
    estimate = float(values.mean())
    rng = np.random.default_rng(random_seed)
    bootstrap = np.empty(int(repetitions), dtype=float)
    batch_size = 200
    for start in range(0, int(repetitions), batch_size):
        stop = min(start + batch_size, int(repetitions))
        batch = stop - start
        sampled_replicates = rng.integers(0, values.shape[0], size=(batch, values.shape[0]))
        sampled_scenes = rng.integers(0, values.shape[1], size=(batch, values.shape[1]))
        sampled = values[sampled_replicates[:, :, None], sampled_scenes[:, None, :]]
        bootstrap[start:stop] = sampled.mean(axis=(1, 2))
    alpha = 1.0 - confidence
    low, high = np.quantile(bootstrap, [alpha / 2, 1.0 - alpha / 2])
    replicate_difference = values.mean(axis=1)
    favorable = replicate_difference < 0 if favorable_sign == "negative" else replicate_difference > 0
    if favorable_sign == "negative":
        direction_met = bool(high < 0 and favorable.sum() >= math.ceil(0.8 * len(favorable)))
    else:
        direction_met = bool(low > 0 and favorable.sum() >= math.ceil(0.8 * len(favorable)))
    return {
        "comparison": comparison,
        "primary_endpoint": endpoint,
        "difference_full_minus_ablation": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "favorable_sign": favorable_sign,
        "favorable_training_replicates": int(favorable.sum()),
        "total_training_replicates": len(favorable),
        "paired_exact_sign_flip_p": _exact_sign_flip_p(replicate_difference),
        "ci_method": f"{int(confidence * 100)}% paired crossed bootstrap over training replicate and scene ({repetitions} draws)",
        "direction_ci_consistency_met": direction_met,
        "evidence": EVIDENCE_NOTE,
    }


def _finalize_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["holm_adjusted_p"] = holm_adjust(frame["paired_exact_sign_flip_p"].to_numpy(float))
    frame["mechanism_support_rule_met"] = (
        frame["direction_ci_consistency_met"] & (frame["holm_adjusted_p"] < 0.05)
    )
    return frame


def _study2_contrasts(results: dict[str, pd.DataFrame], cfg: dict | None) -> pd.DataFrame:
    metrics = results["study2_seed_metrics"]
    scene = results["study2_scene_metrics"]
    missing_scene = results["study2_missing_scene_metrics"]
    repetitions = int(cfg["common"]["bootstrap_repetitions"]) if cfg else 10000
    confidence = float(cfg["common"]["confidence_level"]) if cfg else 0.95
    clean = metrics[metrics["domain"] == "clean_controlled"].set_index(["method", "training_seed"])
    stress = metrics[metrics["domain"] == "adverse_controlled_injected"].set_index(["method", "training_seed"])
    rows = []
    rows.append(
        _paired_crossed_contrast(
            scene[scene["domain"] == "adverse_controlled_injected"],
            "Full",
            "Equal weighting",
            "nll",
            "Full vs Equal weighting",
            "Adverse controlled-injection NLL",
            "negative",
            repetitions,
            stable_seed("study2", "equal_weighting", "bootstrap"),
            confidence,
        )
    )
    rows.append(
        _paired_contrast(
            clean.loc["Full", "sas"],
            clean.loc["No alignment", "sas"],
            "Full vs No alignment",
            "Clean controlled SAS",
            "positive",
            confidence,
        )
    )
    rows.append(
        _paired_crossed_contrast(
            scene[scene["domain"] == "adverse_controlled_injected"],
            "Full",
            "No temporal consistency",
            "feature_drift",
            "Full vs No temporal consistency",
            "Adverse held-out one-step feature prediction error",
            "negative",
            repetitions,
            stable_seed("study2", "temporal", "bootstrap"),
            confidence,
        )
    )
    active_missing = (
        missing_scene[missing_scene["missing_pattern"] != "none"]
        .groupby(["method", "training_seed", "scene_id"], as_index=False)["macro_f1_absolute_drop"]
        .mean()
    )
    rows.append(
        _paired_crossed_contrast(
            active_missing,
            "Full",
            "No modality dropout-distillation",
            "macro_f1_absolute_drop",
            "Full vs No modality dropout-distillation",
            "Mean scene-level single/dual missing-modality macro-F1 drop",
            "negative",
            repetitions,
            stable_seed("study2", "missing", "bootstrap"),
            confidence,
        )
    )
    return _finalize_contrasts(pd.DataFrame(rows))


def study3_tables(results: dict[str, pd.DataFrame], cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    confidence = float(cfg["common"]["confidence_level"]) if cfg else 0.95
    repetitions = int(cfg["common"]["bootstrap_repetitions"]) if cfg else 10000
    seed_metrics = results["study3_seed_metrics"]
    main = _summarize_long(
        seed_metrics,
        ["method", "K"],
        [
            ("intent_macro_f1", "Synthetic dominant-maneuver macro-F1", "higher"),
            ("ADE_1", "ADE_1 (m)", "lower"),
            ("FDE_1", "FDE_1 (m)", "lower"),
            ("MR_1", "MR_1 at 2 m", "lower"),
            ("minADE_6_reportable", "minADE_6 (m)", "lower"),
            ("minFDE_6_reportable", "minFDE_6 (m)", "lower"),
            ("MR_6_reportable", "MR_6 at 2 m", "lower"),
            ("Brier-minFDE_6_reportable", "Brier-minFDE_6", "lower"),
        ], confidence,
    )
    predictions = results["study3_predictions"]
    per_seed_regime = (
        predictions.groupby(["method", "training_seed", "regime"], as_index=False)[
            ["minADE_K", "minFDE_K", "MR_K", "Brier-minFDE_K", "ADE_1", "FDE_1", "MR_1"]
        ]
        .mean()
    )
    regime = _summarize_long(
        per_seed_regime,
        ["method", "regime"],
        [
            ("minADE_K", "minADE_K (m)", "lower"),
            ("minFDE_K", "minFDE_K (m)", "lower"),
            ("MR_K", "MR_K at 2 m", "lower"),
            ("Brier-minFDE_K", "Brier-minFDE_K", "lower"),
        ], confidence,
    )
    latency = _summarize_replicate_bootstrap_long(
        results["study3_latency"],
        ["method"],
        [("p50_ms", "End-to-end batch-1 CPU latency P50 (ms)", "lower"), ("p95_ms", "End-to-end batch-1 CPU latency P95 (ms)", "lower")], confidence, repetitions,
    )
    contrasts, null_checks = _study3_contrasts(results, cfg)
    return {
        "table_s3_main": main,
        "table_s3_by_regime": regime,
        "table_s3_primary_contrasts": contrasts,
        "table_s3_negative_controls": null_checks,
        "table_s3_latency": latency,
    }


def _study3_contrasts(
    results: dict[str, pd.DataFrame], cfg: dict | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = results["study3_predictions"]
    repetitions = int(cfg["common"]["bootstrap_repetitions"]) if cfg else 10000
    confidence = float(cfg["common"]["confidence_level"]) if cfg else 0.95
    rows = []
    for ablation in ("No graph", "Spatial-only graph"):
        rows.append(
            _paired_crossed_contrast(
                predictions[predictions["regime"] == "history_dependent"],
                "Full temporal graph",
                ablation,
                "minFDE_K",
                f"Full temporal graph vs {ablation}",
                "History-dependent minFDE_6",
                "negative",
                repetitions,
                stable_seed("study3", ablation, "history", "bootstrap"),
                confidence,
            )
        )
    rows.append(
        _paired_crossed_contrast(
            predictions,
            "Full temporal graph",
            "No predicted-intent gating",
            "Brier-minFDE_K",
            "Full temporal graph vs No predicted-intent gating",
            "Brier-minFDE_6",
            "negative",
            repetitions,
            stable_seed("study3", "intent_gating", "bootstrap"),
            confidence,
        )
    )
    rows.append(
        _paired_crossed_contrast(
            predictions,
            "Full temporal graph",
            "Single mode K=1",
            "MR_1",
            "Full top-1 vs independently trained Single mode K=1",
            "MR_1 at 2 m",
            "negative",
            repetitions,
            stable_seed("study3", "single_mode", "bootstrap"),
            confidence,
        )
    )
    contrasts = _finalize_contrasts(pd.DataFrame(rows))

    null_rows = []
    for regime_name, ablation, margin in (
        ("independent", "No graph", 0.25),
        ("spatial", "Spatial-only graph", 0.25),
    ):
        contrast = _paired_crossed_contrast(
            predictions[predictions["regime"] == regime_name],
            "Full temporal graph",
            ablation,
            "minFDE_K",
            f"{regime_name}: Full vs {ablation}",
            "minFDE_K (m)",
            "negative",
            repetitions,
            stable_seed("study3", regime_name, "null", "bootstrap"),
            confidence,
        )
        null_rows.append(
            {
                "negative_control": f"{regime_name}: Full vs {ablation}",
                "endpoint": "minFDE_K (m)",
                "equivalence_margin": margin,
                "difference_full_minus_ablation": contrast["difference_full_minus_ablation"],
                "ci_low": contrast["ci_low"],
                "ci_high": contrast["ci_high"],
                "descriptive_equivalence_met": bool(
                    contrast["ci_low"] >= -margin and contrast["ci_high"] <= margin
                ),
                "ci_method": contrast["ci_method"],
                "note": "Prespecified descriptive margin; entire crossed-bootstrap CI must lie within it; not TOST.",
            }
        )
    return contrasts, pd.DataFrame(null_rows)


def execution_status_table(cfg: dict) -> pd.DataFrame:
    urls = {
        "CARLA_0.9.16": "https://carla.readthedocs.io/en/0.9.16/ref_sensors/",
        "nuScenes": "https://www.nuscenes.org/",
        "RADIATE": "https://pro.hw.ac.uk/radiate/",
        "Argoverse_2_Motion": "https://argoverse.github.io/user-guide/tasks/motion_forecasting.html",
    }
    rows = [
        {
            "source": source,
            "status": status,
            "raw_files": 0,
            "scenes_or_logs": 0,
            "checkpoints": 0,
            "official_evaluator_outputs": 0,
            "result_claim_allowed": False,
            "official_source": urls[source],
        }
        for source, status in cfg["execution_status"].items()
    ]
    rows.insert(
        0,
        {
            "source": "Controlled_synthetic_generator",
            "status": "EXECUTED",
            "raw_files": "generated from frozen code/seeds",
            "scenes_or_logs": "see split manifests",
            "checkpoints": "lightweight fitted models; hashes in manifest",
            "official_evaluator_outputs": 0,
            "result_claim_allowed": True,
            "official_source": "N/A; controlled data-generating mechanism",
        },
    )
    return pd.DataFrame(rows)


def parameter_table(cfg: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def visit(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(f"{prefix}.{key}" if prefix else key, item)
        elif isinstance(value, list):
            rows.append({"parameter": prefix, "value": ", ".join(map(str, value))})
        else:
            rows.append({"parameter": prefix, "value": value})

    visit("", cfg)
    return pd.DataFrame(rows)


def manuscript_results(
    s2_tables: dict[str, pd.DataFrame],
    s3_tables: dict[str, pd.DataFrame],
    evidence_label: str,
) -> str:
    s2_contrasts = s2_tables["table_s2_primary_contrasts"].copy()
    s3_contrasts = s3_tables["table_s3_primary_contrasts"].copy()
    return f"""# Controlled synthetic component-validation results

**Evidence label:** `{evidence_label}`

The following text is suitable only for a clearly titled controlled synthetic
mechanism-validation subsection or supplement. It must not be described as a
CARLA, nuScenes, RADIATE, Argoverse 2, real-weather, or real-world result.

## Study 2: reliability, alignment, temporal consistency, and missing modalities

Ten independent, equal-sized training-sample replicates were evaluated on
identical frozen synthetic test scenes and corruption realizations. Conditions
within each replicate shared the same training data. Scene-decomposable primary
contrast intervals use a paired crossed bootstrap over training replicate and
test scene; SAS uses a paired replicate-only interval conditional on its fixed
retrieval gallery. The long summary tables additionally show replicate-only
intervals conditional on the fixed test scenes. The mechanism-specific paired
contrasts were:

{markdown_table(s2_contrasts)}

These contrasts should be interpreted separately: SAS diagnoses cross-modal
alignment; held-out one-step feature prediction error diagnoses temporal consistency; adverse NLL
diagnoses probabilistic prediction under controlled injected degradation; and
missing-modality macro-F1 loss diagnoses the combined dropout-distillation
package. They do not establish real-weather robustness.

## Study 3: temporal graph, intent conditioning, and multimodal trajectories

The controlled trajectory task used 5 s observed histories and 6 s futures at
10 Hz. Multi-hypothesis conditions emitted K=6 trajectories; the single-mode
condition emitted K=1 and was not mislabeled with K=6 metrics. Dominant
maneuver labels are synthetic ground truth and are not Argoverse 2 intent
annotations. The prespecified paired contrasts were:

{markdown_table(s3_contrasts)}

The graph comparisons use the history-dependent regime as their primary
mechanism endpoint. The no-gating intervention keeps the same intent head and
24-prototype bank, replacing only the scene-specific predicted-intent posterior
with a training-only fixed prior. Independent-agent and spatial-only regimes
are retained as negative controls. Mixed or unsupported contrasts must remain
reported and must not be removed by post hoc replicate or parameter selection.
"""
