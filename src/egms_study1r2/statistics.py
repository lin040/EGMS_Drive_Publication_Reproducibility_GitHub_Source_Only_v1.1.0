"""Raw-output statistics for the frozen Study 1-R comparison.

This module deliberately has no expected-performance constants.  It accepts a
complete paired grid of Baseline B and Structured predictions, recomputes every
endpoint from those rows, and retains favorable, null, and adverse effects.

Study 1-R freezes a 15-bin *equal-width* ECE on [0, 1].  This differs from the
equal-mass diagnostic used elsewhere in the repository and is intentional: the
fixed bins yield exact additive sufficient statistics for the paired crossed
bootstrap.  The right endpoint (confidence == 1) belongs to bin 15.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


METHODS = ("Baseline B", "Structured-R2")
ACTION_NAMES = ("KEEP", "SLOW", "YIELD", "STOP")
PROBABILITY_COLUMNS = tuple(f"p_{name}" for name in ACTION_NAMES)
LOGIT_COLUMNS = tuple(f"logit_{name}" for name in ACTION_NAMES)
EPS = 1.0e-12
DIRECTION_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class MetricSpec:
    endpoint: str
    label: str
    orientation: str
    unit: str
    display_scale: float
    panel: str

    @property
    def orientation_sign(self) -> float:
        if self.orientation == "higher_is_better":
            return 1.0
        if self.orientation == "lower_is_better":
            return -1.0
        raise ValueError(f"Unknown metric orientation: {self.orientation}")


METRIC_SPECS = (
    MetricSpec("macro_f1", "Macro-F1", "higher_is_better", "raw difference", 1.0, "action"),
    MetricSpec("nll", "NLL", "lower_is_better", "raw difference", 1.0, "action"),
    MetricSpec("brier", "Brier", "lower_is_better", "raw difference", 1.0, "action"),
    MetricSpec("ece", "ECE", "lower_is_better", "raw difference", 1.0, "action"),
    MetricSpec("collision", "Collision", "lower_is_better", "percentage points", 100.0, "event"),
    MetricSpec("near_miss", "Near miss", "lower_is_better", "percentage points", 100.0, "event"),
    MetricSpec(
        "critical_event",
        "Critical event",
        "lower_is_better",
        "percentage points",
        100.0,
        "event",
    ),
    MetricSpec(
        "route_completion",
        "Route completion",
        "higher_is_better",
        "percentage points",
        100.0,
        "event",
    ),
    MetricSpec("ttc_p5", "TTC-P5", "higher_is_better", "seconds", 1.0, "ttc"),
    MetricSpec("jerk_p95", "Jerk-P95", "lower_is_better", "m/s^3", 1.0, "jerk"),
)
METRIC_BY_ENDPOINT = {spec.endpoint: spec for spec in METRIC_SPECS}


OFFLINE_REQUIRED_COLUMNS = {
    "replicate_id",
    "training_seed",
    "method",
    "episode_id",
    "scenario_cell",
    "frame_id",
    "y_true",
    "y_pred",
    *LOGIT_COLUMNS,
    *PROBABILITY_COLUMNS,
}

EPISODE_REQUIRED_COLUMNS = {
    "replicate_id",
    "training_seed",
    "method",
    "episode_id",
    "scenario_cell",
    "collision",
    "near_miss",
    "critical_event",
    "route_completed",
    "ttc_p5",
    "jerk_p95",
}


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _as_binary(series: pd.Series, name: str) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=np.int8)
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0]).all():
        raise ValueError(f"{name} must contain only boolean/0/1 values")
    return values.astype(np.int8)


def _method_and_grid(
    offline: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    expected_replicates: int,
    expected_episodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offline_methods = set(offline["method"].astype(str).unique())
    episode_methods = set(episodes["method"].astype(str).unique())
    if offline_methods != set(METHODS) or episode_methods != set(METHODS):
        raise ValueError(
            f"Both inputs must contain exactly {METHODS}; found "
            f"offline={sorted(offline_methods)}, episodes={sorted(episode_methods)}"
        )

    offline_replicates = np.sort(offline["replicate_id"].astype(int).unique())
    episode_replicates = np.sort(episodes["replicate_id"].astype(int).unique())
    if not np.array_equal(offline_replicates, episode_replicates):
        raise ValueError("Offline and episode replicate IDs differ")
    if len(offline_replicates) != expected_replicates:
        raise ValueError(
            f"Expected {expected_replicates} training replicates; "
            f"found {len(offline_replicates)}"
        )

    offline_episode_ids = np.sort(offline["episode_id"].astype(str).unique())
    rollout_episode_ids = np.sort(episodes["episode_id"].astype(str).unique())
    if len(offline_episode_ids) != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} fixed offline test episodes; "
            f"found {len(offline_episode_ids)}"
        )
    if len(rollout_episode_ids) != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} fixed closed-loop episodes; "
            f"found {len(rollout_episode_ids)}"
        )

    return offline_replicates, offline_episode_ids, rollout_episode_ids


def _validate_offline_grid(
    frame: pd.DataFrame,
    replicates: np.ndarray,
    episode_ids: np.ndarray,
) -> pd.DataFrame:
    offline = frame.copy()
    offline["replicate_id"] = pd.to_numeric(offline["replicate_id"], errors="raise").astype(int)
    offline["frame_id"] = pd.to_numeric(offline["frame_id"], errors="raise").astype(int)
    offline["episode_id"] = offline["episode_id"].astype(str)
    offline["method"] = offline["method"].astype(str)

    key = ["method", "replicate_id", "episode_id", "frame_id"]
    if offline.duplicated(key).any():
        examples = offline.loc[offline.duplicated(key, keep=False), key].head().to_dict("records")
        raise ValueError(f"Duplicate offline prediction keys: {examples}")

    probability = offline.loc[:, PROBABILITY_COLUMNS].apply(pd.to_numeric, errors="raise").to_numpy(float)
    logits = offline.loc[:, LOGIT_COLUMNS].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(probability).all() or not np.isfinite(logits).all():
        raise ValueError("Offline logits/probabilities contain non-finite values")
    if np.any(probability < -1.0e-12) or np.any(probability > 1.0 + 1.0e-12):
        raise ValueError("Offline probability lies outside [0,1]")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1.0e-8, rtol=0.0):
        raise ValueError("Offline probability rows do not sum to one")

    truth = pd.to_numeric(offline["y_true"], errors="raise").to_numpy(dtype=int)
    prediction = pd.to_numeric(offline["y_pred"], errors="raise").to_numpy(dtype=int)
    if np.any(truth < 0) or np.any(truth >= len(ACTION_NAMES)):
        raise ValueError("y_true is outside the frozen four-class action range")
    if np.any(prediction < 0) or np.any(prediction >= len(ACTION_NAMES)):
        raise ValueError("y_pred is outside the frozen four-class action range")
    if not np.array_equal(prediction, probability.argmax(axis=1)):
        raise ValueError("y_pred does not equal deterministic argmax of calibrated probabilities")

    # The held-out frame grid and labels must be identical for both methods and
    # every training replicate.  This is the offline paired comparison.
    canonical_key = ["episode_id", "frame_id"]
    canonical = offline.groupby(canonical_key, sort=False).agg(
        rows=("y_true", "size"),
        labels=("y_true", "nunique"),
        cells=("scenario_cell", "nunique"),
    )
    expected_rows = len(METHODS) * len(replicates)
    if not canonical["rows"].eq(expected_rows).all():
        raise ValueError("The fixed offline frame grid is incomplete or contains extra rows")
    if not canonical["labels"].eq(1).all() or not canonical["cells"].eq(1).all():
        raise ValueError("A fixed test frame changes label or scenario cell across runs")

    observed_pairs = offline[["replicate_id", "episode_id"]].drop_duplicates()
    if len(observed_pairs) != len(replicates) * len(episode_ids):
        raise ValueError("At least one replicate/test-episode pair is missing offline predictions")

    offline.loc[:, PROBABILITY_COLUMNS] = probability
    offline.loc[:, LOGIT_COLUMNS] = logits
    offline["y_true"] = truth
    offline["y_pred"] = prediction
    return offline


def _validate_episode_grid(
    frame: pd.DataFrame,
    offline: pd.DataFrame,
    replicates: np.ndarray,
    episode_ids: np.ndarray,
) -> pd.DataFrame:
    episodes = frame.copy()
    episodes["replicate_id"] = pd.to_numeric(episodes["replicate_id"], errors="raise").astype(int)
    episodes["episode_id"] = episodes["episode_id"].astype(str)
    episodes["method"] = episodes["method"].astype(str)
    key = ["method", "replicate_id", "episode_id"]
    if episodes.duplicated(key).any():
        examples = episodes.loc[episodes.duplicated(key, keep=False), key].head().to_dict("records")
        raise ValueError(f"Duplicate episode metric keys: {examples}")
    expected_rows = len(METHODS) * len(replicates) * len(episode_ids)
    if len(episodes) != expected_rows:
        raise ValueError(f"Expected {expected_rows} paired episode rows; found {len(episodes)}")

    collision = _as_binary(episodes["collision"], "collision")
    near_miss = _as_binary(episodes["near_miss"], "near_miss")
    critical = _as_binary(episodes["critical_event"], "critical_event")
    if np.any((collision == 1) & (near_miss == 1)):
        raise ValueError("Near miss must exclude collision episodes in the frozen protocol")
    if np.any(critical < np.maximum(collision, near_miss)):
        raise ValueError("Every collision or near miss must also be a critical event")

    for column in ("route_completed", "ttc_p5", "jerk_p95"):
        episodes[column] = pd.to_numeric(episodes[column], errors="raise")
        if not np.isfinite(episodes[column].to_numpy(float)).all():
            raise ValueError(f"{column} contains non-finite values")
    if not episodes["route_completed"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("route_completed must be a flag or fraction in [0,1]")
    if (episodes[["ttc_p5", "jerk_p95"]] < 0.0).any().any():
        raise ValueError("TTC-P5 and jerk-P95 must be nonnegative")

    episodes["collision"] = collision
    episodes["near_miss"] = near_miss
    episodes["critical_event"] = critical

    # Training data seeds and scenario-cell identities are pairing metadata.
    combined = pd.concat(
        [
            offline[["replicate_id", "method", "training_seed"]].drop_duplicates(),
            episodes[["replicate_id", "method", "training_seed"]].drop_duplicates(),
        ],
        ignore_index=True,
    )
    if combined.groupby(["replicate_id", "method"])["training_seed"].nunique().gt(1).any():
        raise ValueError("Training seed metadata disagrees between raw output tables")
    if combined.groupby("replicate_id")["training_seed"].nunique().gt(1).any():
        raise ValueError("Paired methods must use the same training-sample seed per replicate")

    episode_cells = pd.concat(
        [
            offline[["episode_id", "scenario_cell"]].drop_duplicates(),
            episodes[["episode_id", "scenario_cell"]].drop_duplicates(),
        ],
        ignore_index=True,
    )
    if episode_cells.groupby("episode_id")["scenario_cell"].nunique().gt(1).any():
        raise ValueError("An episode_id maps to more than one scenario cell")
    return episodes


def _macro_f1(confusion: np.ndarray) -> np.ndarray:
    """Vectorized macro-F1 for arrays ending in [true, predicted]."""

    matrix = np.asarray(confusion, dtype=float)
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    fp = matrix.sum(axis=-2) - tp
    fn = matrix.sum(axis=-1) - tp
    denominator = 2.0 * tp + fp + fn
    scores = np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0.0,
    )
    return scores.mean(axis=-1)


def expected_calibration_error(
    y_true: Iterable[int],
    probabilities: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Return frozen Study 1-R top-label ECE with equal-width bins."""

    truth = np.asarray(list(y_true), dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    if probability.ndim != 2 or probability.shape[0] != len(truth):
        raise ValueError("probabilities must have one row per y_true value")
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    count = np.bincount(bins, minlength=n_bins).astype(float)
    confidence_sum = np.bincount(bins, weights=confidence, minlength=n_bins)
    correct_sum = np.bincount(
        bins,
        weights=(prediction == truth).astype(float),
        minlength=n_bins,
    )
    return float(np.abs(correct_sum - confidence_sum).sum() / max(1.0, count.sum()))


def _offline_sufficient_statistics(
    offline: pd.DataFrame,
    replicates: np.ndarray,
    episode_ids: np.ndarray,
    *,
    n_bins: int,
) -> dict[str, np.ndarray]:
    method_index = {name: index for index, name in enumerate(METHODS)}
    replicate_index = {int(value): index for index, value in enumerate(replicates)}
    episode_index = {str(value): index for index, value in enumerate(episode_ids)}
    shape = (len(METHODS), len(replicates), len(episode_ids))
    counts = np.zeros(shape, dtype=np.int32)
    confusion = np.zeros(shape + (len(ACTION_NAMES), len(ACTION_NAMES)), dtype=np.int32)
    nll_sum = np.zeros(shape, dtype=float)
    brier_sum = np.zeros(shape, dtype=float)
    bin_count = np.zeros(shape + (n_bins,), dtype=np.int32)
    bin_confidence = np.zeros(shape + (n_bins,), dtype=float)
    bin_correct = np.zeros(shape + (n_bins,), dtype=float)

    for keys, group in offline.groupby(["method", "replicate_id", "episode_id"], sort=False):
        method, replicate, episode = keys
        index = (
            method_index[str(method)],
            replicate_index[int(replicate)],
            episode_index[str(episode)],
        )
        truth = group["y_true"].to_numpy(dtype=int)
        prediction = group["y_pred"].to_numpy(dtype=int)
        probability = group.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
        counts[index] = len(group)
        np.add.at(confusion[index], (truth, prediction), 1)
        nll_sum[index] = float(-np.log(np.maximum(probability[np.arange(len(truth)), truth], EPS)).sum())
        one_hot = np.eye(len(ACTION_NAMES), dtype=float)[truth]
        brier_sum[index] = float(np.square(probability - one_hot).sum())
        confidence = probability.max(axis=1)
        bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
        np.add.at(bin_count[index], bins, 1)
        np.add.at(bin_confidence[index], bins, confidence)
        np.add.at(bin_correct[index], bins, (prediction == truth).astype(float))

    if np.any(counts <= 0):
        raise ValueError("At least one method/replicate/episode cell has no offline frames")
    return {
        "count": counts,
        "confusion": confusion,
        "nll_sum": nll_sum,
        "brier_sum": brier_sum,
        "ece_count": bin_count,
        "ece_confidence_sum": bin_confidence,
        "ece_correct_sum": bin_correct,
    }


def _episode_arrays(
    episodes: pd.DataFrame,
    replicates: np.ndarray,
    episode_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    ordered = (
        episodes.assign(
            method=pd.Categorical(episodes["method"], categories=list(METHODS), ordered=True),
            replicate_id=pd.Categorical(episodes["replicate_id"], categories=list(replicates), ordered=True),
            episode_id=pd.Categorical(episodes["episode_id"], categories=list(episode_ids), ordered=True),
        )
        .sort_values(["method", "replicate_id", "episode_id"])
        .reset_index(drop=True)
    )
    expected = len(METHODS) * len(replicates) * len(episode_ids)
    if len(ordered) != expected:
        raise ValueError("Episode grid is incomplete")
    shape = (len(METHODS), len(replicates), len(episode_ids))
    columns = {
        "collision": "collision",
        "near_miss": "near_miss",
        "critical_event": "critical_event",
        "route_completion": "route_completed",
        "ttc_p5": "ttc_p5",
        "jerk_p95": "jerk_p95",
    }
    return {
        endpoint: ordered[column].to_numpy(dtype=float).reshape(shape)
        for endpoint, column in columns.items()
    }


def _aggregate_metrics(
    offline_stats: Mapping[str, np.ndarray],
    episode_arrays: Mapping[str, np.ndarray],
    replicate_counts: np.ndarray,
    episode_counts: np.ndarray,
) -> dict[str, np.ndarray]:
    """Aggregate metrics for shared crossed-bootstrap count vectors.

    Inputs have shapes [draw, replicate] and [draw, episode].  The same two
    count matrices are contracted against both methods for every endpoint.
    """

    rc = np.asarray(replicate_counts, dtype=float)
    ec = np.asarray(episode_counts, dtype=float)
    if rc.ndim != 2 or ec.ndim != 2 or rc.shape[0] != ec.shape[0]:
        raise ValueError("Bootstrap count matrices must share a draw dimension")

    frame_count = np.einsum("br,be,mre->bm", rc, ec, offline_stats["count"], optimize=True)
    confusion = np.einsum(
        "br,be,mreij->bmij",
        rc,
        ec,
        offline_stats["confusion"],
        optimize=True,
    )
    nll_sum = np.einsum("br,be,mre->bm", rc, ec, offline_stats["nll_sum"], optimize=True)
    brier_sum = np.einsum("br,be,mre->bm", rc, ec, offline_stats["brier_sum"], optimize=True)
    ece_confidence = np.einsum(
        "br,be,mrek->bmk",
        rc,
        ec,
        offline_stats["ece_confidence_sum"],
        optimize=True,
    )
    ece_correct = np.einsum(
        "br,be,mrek->bmk",
        rc,
        ec,
        offline_stats["ece_correct_sum"],
        optimize=True,
    )

    metrics: dict[str, np.ndarray] = {
        "macro_f1": _macro_f1(confusion),
        "nll": nll_sum / frame_count,
        "brier": brier_sum / frame_count,
        # count/N * |accuracy-confidence| simplifies exactly to
        # |correct_sum-confidence_sum|/N for fixed equal-width bins.
        "ece": np.abs(ece_correct - ece_confidence).sum(axis=-1) / frame_count,
    }

    episode_denominator = rc.sum(axis=1) * ec.sum(axis=1)
    for endpoint in ("collision", "near_miss", "critical_event", "route_completion", "jerk_p95"):
        numerator = np.einsum(
            "br,be,mre->bm",
            rc,
            ec,
            episode_arrays[endpoint],
            optimize=True,
        )
        metrics[endpoint] = numerator / episode_denominator[:, None]

    baseline_index = METHODS.index("Baseline B")
    structured_index = METHODS.index("Structured-R2")
    jointly_collision_free = (
        (episode_arrays["collision"][baseline_index] == 0.0)
        & (episode_arrays["collision"][structured_index] == 0.0)
    ).astype(float)
    eligible_count = np.einsum(
        "br,be,re->b",
        rc,
        ec,
        jointly_collision_free,
        optimize=True,
    )
    if np.any(eligible_count <= 0.0):
        raise ValueError("A bootstrap draw has no jointly collision-free TTC episode pair")
    ttc_numerator = np.einsum(
        "br,be,mre,re->bm",
        rc,
        ec,
        episode_arrays["ttc_p5"],
        jointly_collision_free,
        optimize=True,
    )
    metrics["ttc_p5"] = ttc_numerator / eligible_count[:, None]
    return metrics


def exact_sign_flip_p(values: Iterable[float]) -> float:
    """Two-sided exact sign-flip p value from all 2^R assignments."""

    differences = np.asarray(list(values), dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0:
        return float("nan")
    combinations = np.arange(2 ** len(differences), dtype=np.uint64)[:, None]
    bits = (combinations >> np.arange(len(differences), dtype=np.uint64)) & 1
    signs = 1.0 - 2.0 * bits
    distribution = (signs * differences[None, :]).mean(axis=1)
    observed = abs(float(differences.mean()))
    return float(np.mean(np.abs(distribution) >= observed - 1.0e-15))


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm step-down adjusted p values, restored to input order."""

    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Holm p values must be a finite one-dimensional array")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _quantile_interval(
    values: np.ndarray,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1.0 - confidence
    quantiles = np.quantile(
        np.asarray(values, dtype=float),
        [alpha / 2.0, 1.0 - alpha / 2.0],
        axis=0,
        method="linear",
    )
    return quantiles[0], quantiles[1]


def summarize_study1r(
    offline_predictions: pd.DataFrame,
    episode_metrics: pd.DataFrame,
    *,
    repetitions: int = 10_000,
    confidence: float = 0.95,
    bootstrap_seed: int = 71031,
    calibration_bins: int = 15,
    expected_replicates: int = 10,
    expected_episodes: int = 360,
) -> dict[str, pd.DataFrame]:
    """Recompute and summarize all Study 1-R endpoints from raw outputs.

    The bootstrap samples the training-replicate axis and the fixed test-episode
    axis independently with replacement.  Both methods and all endpoints share
    the identical count vectors in a draw.  Point estimates use every formal
    replicate and episode exactly once.
    """

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    if calibration_bins != 15:
        raise ValueError("Study 1-R freezes exactly 15 equal-width ECE bins")
    _require_columns(offline_predictions, OFFLINE_REQUIRED_COLUMNS, "offline_predictions")
    _require_columns(episode_metrics, EPISODE_REQUIRED_COLUMNS, "episode_metrics")

    replicates, offline_episode_ids, rollout_episode_ids = _method_and_grid(
        offline_predictions,
        episode_metrics,
        expected_replicates=expected_replicates,
        expected_episodes=expected_episodes,
    )
    offline = _validate_offline_grid(
        offline_predictions,
        replicates,
        offline_episode_ids,
    )
    episodes = _validate_episode_grid(
        episode_metrics,
        offline,
        replicates,
        rollout_episode_ids,
    )
    offline_stats = _offline_sufficient_statistics(
        offline,
        replicates,
        offline_episode_ids,
        n_bins=calibration_bins,
    )
    episode_arrays = _episode_arrays(episodes, replicates, rollout_episode_ids)

    point_metrics = _aggregate_metrics(
        offline_stats,
        episode_arrays,
        np.ones((1, len(replicates)), dtype=np.int16),
        np.ones((1, len(offline_episode_ids)), dtype=np.int16),
    )

    rng = np.random.Generator(np.random.PCG64(int(bootstrap_seed)))
    replicate_counts = rng.multinomial(
        len(replicates),
        np.full(len(replicates), 1.0 / len(replicates)),
        size=int(repetitions),
    ).astype(np.int16)
    episode_counts = rng.multinomial(
        len(offline_episode_ids),
        np.full(len(offline_episode_ids), 1.0 / len(offline_episode_ids)),
        size=int(repetitions),
    ).astype(np.int16)
    bootstrap_metrics = _aggregate_metrics(
        offline_stats,
        episode_arrays,
        replicate_counts,
        episode_counts,
    )

    # Replicate-level effects use all 360 fixed episodes and feed the exact
    # sign-flip test and directionality inventory.
    replicate_metrics = _aggregate_metrics(
        offline_stats,
        episode_arrays,
        np.eye(len(replicates), dtype=np.int16),
        np.ones((len(replicates), len(offline_episode_ids)), dtype=np.int16),
    )
    seed_map = (
        episodes.groupby("replicate_id", sort=True)["training_seed"]
        .first()
        .reindex(replicates)
        .to_dict()
    )

    method_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    replicate_rows: list[dict[str, object]] = []
    bootstrap_frames: list[pd.DataFrame] = []
    baseline_index = METHODS.index("Baseline B")
    structured_index = METHODS.index("Structured-R2")

    for spec in METRIC_SPECS:
        point = point_metrics[spec.endpoint][0]
        draws = bootstrap_metrics[spec.endpoint]
        marginal_low, marginal_high = _quantile_interval(draws, confidence)
        raw_draws = draws[:, structured_index] - draws[:, baseline_index]
        raw_low, raw_high = _quantile_interval(raw_draws, confidence)
        raw_difference = float(point[structured_index] - point[baseline_index])
        benefit_draws = spec.orientation_sign * raw_draws
        benefit_low, benefit_high = _quantile_interval(benefit_draws, confidence)
        benefit_difference = spec.orientation_sign * raw_difference

        for method_index, method in enumerate(METHODS):
            method_rows.append(
                {
                    "endpoint": spec.endpoint,
                    "metric": spec.label,
                    "panel": spec.panel,
                    "method": method,
                    "estimate": float(point[method_index]),
                    "ci_low": float(marginal_low[method_index]),
                    "ci_high": float(marginal_high[method_index]),
                    "unit": spec.unit,
                    "display_scale": spec.display_scale,
                    "orientation": spec.orientation,
                    "ci_method": (
                        f"{int(round(confidence * 100))}% percentile paired crossed bootstrap; "
                        f"{repetitions} shared PCG64 draws; NumPy linear quantiles"
                    ),
                }
            )

        replicate_raw = (
            replicate_metrics[spec.endpoint][:, structured_index]
            - replicate_metrics[spec.endpoint][:, baseline_index]
        )
        replicate_benefit = spec.orientation_sign * replicate_raw
        favorable = int(np.sum(replicate_benefit > DIRECTION_TOLERANCE))
        tied = int(np.sum(np.abs(replicate_benefit) <= DIRECTION_TOLERANCE))
        adverse = int(np.sum(replicate_benefit < -DIRECTION_TOLERANCE))
        sign_flip_p = exact_sign_flip_p(replicate_raw)
        eligible_pairs = (
            int(
                np.sum(
                    (episode_arrays["collision"][baseline_index] == 0.0)
                    & (episode_arrays["collision"][structured_index] == 0.0)
                )
            )
            if spec.endpoint == "ttc_p5"
            else len(replicates) * len(rollout_episode_ids)
        )
        contrast_rows.append(
            {
                "endpoint": spec.endpoint,
                "metric": spec.label,
                "panel": spec.panel,
                "orientation": spec.orientation,
                "unit": spec.unit,
                "display_scale": spec.display_scale,
                "baseline_estimate": float(point[baseline_index]),
                "structured_estimate": float(point[structured_index]),
                "raw_difference_structured_minus_baseline": raw_difference,
                "raw_ci_low": float(raw_low),
                "raw_ci_high": float(raw_high),
                "benefit_difference": float(benefit_difference),
                "benefit_ci_low": float(benefit_low),
                "benefit_ci_high": float(benefit_high),
                "favorable_replicates": favorable,
                "tied_replicates": tied,
                "adverse_replicates": adverse,
                "total_training_replicates": len(replicates),
                "paired_exact_sign_flip_p": sign_flip_p,
                "eligible_replicate_episode_pairs": eligible_pairs,
                "bootstrap_repetitions": int(repetitions),
                "bootstrap_seed": int(bootstrap_seed),
                "ci_method": (
                    f"{int(round(confidence * 100))}% paired crossed percentile bootstrap "
                    f"over {len(replicates)} training replicates and {len(offline_episode_ids)} fixed "
                    f"test episodes; {repetitions} shared draws; NumPy linear quantiles"
                ),
                "ece_definition": "15 equal-width confidence bins on [0,1], right endpoint included",
            }
        )

        for replicate_index, replicate_id in enumerate(replicates):
            benefit = float(replicate_benefit[replicate_index])
            direction = (
                "favorable"
                if benefit > DIRECTION_TOLERANCE
                else "adverse"
                if benefit < -DIRECTION_TOLERANCE
                else "tie"
            )
            replicate_rows.append(
                {
                    "replicate_id": int(replicate_id),
                    "training_seed": int(seed_map[int(replicate_id)]),
                    "endpoint": spec.endpoint,
                    "metric": spec.label,
                    "baseline_estimate": float(replicate_metrics[spec.endpoint][replicate_index, baseline_index]),
                    "structured_estimate": float(replicate_metrics[spec.endpoint][replicate_index, structured_index]),
                    "raw_difference_structured_minus_baseline": float(replicate_raw[replicate_index]),
                    "benefit_difference": benefit,
                    "direction": direction,
                }
            )

        bootstrap_frames.append(
            pd.DataFrame(
                {
                    "draw": np.arange(int(repetitions), dtype=int),
                    "endpoint": spec.endpoint,
                    "baseline_estimate": draws[:, baseline_index],
                    "structured_estimate": draws[:, structured_index],
                    "raw_difference_structured_minus_baseline": raw_draws,
                    "benefit_difference": benefit_draws,
                    "bootstrap_seed": int(bootstrap_seed),
                }
            )
        )

    contrasts = pd.DataFrame(contrast_rows)
    contrasts["holm_adjusted_p"] = holm_adjust(
        contrasts["paired_exact_sign_flip_p"].to_numpy(dtype=float)
    )
    contrasts["direction_ci_consistency_met"] = (
        (contrasts["benefit_ci_low"] > 0.0)
        & (contrasts["favorable_replicates"] >= 8)
    )
    contrasts["support_rule_met"] = (
        contrasts["direction_ci_consistency_met"]
        & (contrasts["holm_adjusted_p"] < 0.05)
    )
    # This is a result field, never an acceptance criterion.
    contrasts["validator_requires_support"] = False

    return {
        "metric_summary": pd.DataFrame(method_rows),
        "paired_contrasts": contrasts,
        "replicate_effects": pd.DataFrame(replicate_rows),
        "bootstrap_draws": pd.concat(bootstrap_frames, ignore_index=True),
    }
