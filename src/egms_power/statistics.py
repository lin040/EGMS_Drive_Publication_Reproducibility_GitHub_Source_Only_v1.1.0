"""Statistical calculations for a prospective collision-outcome study.

The functions in this module operate on prespecified design assumptions. They
do not simulate or estimate EGMS-Drive performance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import binom, norm


@dataclass(frozen=True)
class MonteCarloEstimate:
    """A binomial Monte Carlo estimate with uncertainty."""

    estimate: float
    rejections: int
    repetitions: int
    mcse: float
    ci_low: float
    ci_high: float


def _validate_probability(value: float, name: str, *, strict: bool = False) -> None:
    lower_ok = value > 0 if strict else value >= 0
    upper_ok = value < 1 if strict else value <= 1
    if not (lower_ok and upper_ok):
        bounds = "(0, 1)" if strict else "[0, 1]"
        raise ValueError(f"{name} must be in {bounds}; received {value!r}")


def _validate_design(alpha: float, target_power: float | None = None) -> None:
    _validate_probability(alpha, "alpha", strict=True)
    if target_power is not None:
        _validate_probability(target_power, "target_power", strict=True)


def required_n_two_independent_proportions(
    p_control: float,
    p_candidate: float,
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> tuple[float, int]:
    """Return raw and ceiling sample sizes per arm for a two-sided score test.

    This uses a pooled variance under the null and an unpooled variance under
    the alternative. Equal allocation is assumed.
    """

    _validate_probability(p_control, "p_control")
    _validate_probability(p_candidate, "p_candidate")
    _validate_design(alpha, target_power)
    delta = abs(p_control - p_candidate)
    if delta == 0:
        raise ValueError("p_control and p_candidate must differ for power planning")

    pooled = (p_control + p_candidate) / 2.0
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    z_power = norm.ppf(target_power)
    null_sd = math.sqrt(2.0 * pooled * (1.0 - pooled))
    alt_sd = math.sqrt(
        p_control * (1.0 - p_control)
        + p_candidate * (1.0 - p_candidate)
    )
    raw_n = ((z_alpha * null_sd + z_power * alt_sd) ** 2) / (delta**2)
    return float(raw_n), int(math.ceil(raw_n))


def two_independent_proportions_power(
    n_per_arm: float,
    p_control: float,
    p_candidate: float,
    *,
    alpha: float = 0.05,
) -> float:
    """Approximate two-sided power for equal-sized independent arms."""

    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")
    _validate_probability(p_control, "p_control")
    _validate_probability(p_candidate, "p_candidate")
    _validate_design(alpha)
    if p_control == p_candidate:
        return float(alpha)

    pooled = (p_control + p_candidate) / 2.0
    delta = abs(p_control - p_candidate)
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    null_sd = math.sqrt(2.0 * pooled * (1.0 - pooled))
    alt_sd = math.sqrt(
        p_control * (1.0 - p_control)
        + p_candidate * (1.0 - p_candidate)
    )
    shifted_mean = math.sqrt(float(n_per_arm)) * delta
    upper = norm.cdf((shifted_mean - z_alpha * null_sd) / alt_sd)
    lower = norm.cdf((-shifted_mean - z_alpha * null_sd) / alt_sd)
    return float(upper + lower)


def pooled_two_proportion_pvalues(
    control_events: np.ndarray,
    candidate_events: np.ndarray,
    n_per_arm: int,
) -> np.ndarray:
    """Vectorized two-sided pooled z-test p-values."""

    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")
    x0 = np.asarray(control_events, dtype=float)
    x1 = np.asarray(candidate_events, dtype=float)
    pooled = (x0 + x1) / (2.0 * n_per_arm)
    variance = pooled * (1.0 - pooled) * (2.0 / n_per_arm)
    difference = x0 / n_per_arm - x1 / n_per_arm
    z = np.divide(
        difference,
        np.sqrt(variance),
        out=np.zeros_like(difference, dtype=float),
        where=variance > 0,
    )
    pvalues = 2.0 * norm.sf(np.abs(z))
    return np.where(variance > 0, pvalues, 1.0)


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between 0 and total")
    _validate_probability(confidence, "confidence", strict=True)
    z = norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p_hat = successes / total
    denominator = 1.0 + (z**2) / total
    center = (p_hat + (z**2) / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            p_hat * (1.0 - p_hat) / total
            + (z**2) / (4.0 * total**2)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def _mc_estimate(rejections: np.ndarray, confidence: float = 0.95) -> MonteCarloEstimate:
    decisions = np.asarray(rejections, dtype=bool)
    total = int(decisions.size)
    if total == 0:
        raise ValueError("at least one Monte Carlo repetition is required")
    count = int(decisions.sum())
    estimate = count / total
    mcse = math.sqrt(estimate * (1.0 - estimate) / total)
    low, high = wilson_interval(count, total, confidence)
    return MonteCarloEstimate(estimate, count, total, mcse, low, high)


def simulate_unpaired_power(
    n_per_arm: int,
    p_control: float,
    p_candidate: float,
    *,
    alpha: float = 0.05,
    repetitions: int = 10_000,
    seed: int = 20260810,
) -> MonteCarloEstimate:
    """Estimate power by simulating complete independent two-arm trials."""

    if n_per_arm <= 0 or repetitions <= 0:
        raise ValueError("n_per_arm and repetitions must be positive")
    _validate_probability(p_control, "p_control")
    _validate_probability(p_candidate, "p_candidate")
    _validate_design(alpha)
    rng = np.random.default_rng(seed)
    control_events = rng.binomial(n_per_arm, p_control, size=repetitions)
    candidate_events = rng.binomial(n_per_arm, p_candidate, size=repetitions)
    pvalues = pooled_two_proportion_pvalues(
        control_events, candidate_events, n_per_arm
    )
    return _mc_estimate(pvalues < alpha)


def simulate_wilson_coverage(
    n: int,
    true_probability: float,
    *,
    confidence: float = 0.95,
    repetitions: int = 10_000,
    seed: int = 20260810,
) -> MonteCarloEstimate:
    """Estimate coverage of a Wilson interval through complete samples."""

    if n <= 0 or repetitions <= 0:
        raise ValueError("n and repetitions must be positive")
    _validate_probability(true_probability, "true_probability")
    _validate_probability(confidence, "confidence", strict=True)
    rng = np.random.default_rng(seed)
    counts = rng.binomial(n, true_probability, size=repetitions)
    p_hat = counts / n
    z = norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    denominator = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2.0 * n)) / denominator
    half = (
        z
        * np.sqrt(p_hat * (1.0 - p_hat) / n + (z**2) / (4.0 * n**2))
        / denominator
    )
    covered = (center - half <= true_probability) & (true_probability <= center + half)
    return _mc_estimate(covered, confidence=confidence)


def feasible_correlation_bounds(p_control: float, p_candidate: float) -> tuple[float, float]:
    """Fréchet-compatible Pearson-correlation bounds for paired Bernoulli data."""

    _validate_probability(p_control, "p_control", strict=True)
    _validate_probability(p_candidate, "p_candidate", strict=True)
    scale = math.sqrt(
        p_control * (1.0 - p_control)
        * p_candidate * (1.0 - p_candidate)
    )
    lower_joint = max(0.0, p_control + p_candidate - 1.0)
    upper_joint = min(p_control, p_candidate)
    center = p_control * p_candidate
    return (lower_joint - center) / scale, (upper_joint - center) / scale


def paired_joint_probabilities(
    p_control: float,
    p_candidate: float,
    correlation: float,
) -> dict[str, float]:
    """Return the four paired collision probabilities under a correlation.

    ``control_only`` is pi_10 and ``candidate_only`` is pi_01.
    """

    lower, upper = feasible_correlation_bounds(p_control, p_candidate)
    tolerance = 1e-12
    if correlation < lower - tolerance or correlation > upper + tolerance:
        raise ValueError(
            f"correlation {correlation} is infeasible; valid range is "
            f"[{lower:.6f}, {upper:.6f}]"
        )
    scale = math.sqrt(
        p_control * (1.0 - p_control)
        * p_candidate * (1.0 - p_candidate)
    )
    both_collision = p_control * p_candidate + correlation * scale
    control_only = p_control - both_collision
    candidate_only = p_candidate - both_collision
    both_safe = 1.0 - both_collision - control_only - candidate_only
    values = {
        "both_safe": both_safe,
        "candidate_only": candidate_only,
        "control_only": control_only,
        "both_collision": both_collision,
    }
    if any(value < -tolerance for value in values.values()):
        raise ValueError("paired joint probabilities include a negative cell")
    values = {key: max(0.0, float(value)) for key, value in values.items()}
    total = sum(values.values())
    return {key: value / total for key, value in values.items()}


def exact_mcnemar_pvalue(control_only: int, candidate_only: int) -> float:
    """Two-sided exact McNemar p-value from the discordant cells."""

    if control_only < 0 or candidate_only < 0:
        raise ValueError("discordant counts cannot be negative")
    discordant = int(control_only + candidate_only)
    if discordant == 0:
        return 1.0
    smaller = int(min(control_only, candidate_only))
    return float(min(1.0, 2.0 * binom.cdf(smaller, discordant, 0.5)))


def _mcnemar_conditional_rejection_probability(
    discordant_counts: np.ndarray,
    theta_control_only: float,
    alpha: float,
) -> np.ndarray:
    """P(reject | D=d) for the exact two-sided McNemar test."""

    d = np.asarray(discordant_counts, dtype=int)
    critical = binom.ppf(alpha / 2.0, d, 0.5)
    critical = np.nan_to_num(critical, nan=-1).astype(int)
    null_cdf = binom.cdf(critical, d, 0.5)
    critical = np.where(null_cdf > alpha / 2.0, critical - 1, critical)
    critical = np.maximum(critical, -1)
    lower = np.where(
        critical >= 0,
        binom.cdf(critical, d, theta_control_only),
        0.0,
    )
    upper_threshold = d - critical
    upper = np.where(
        upper_threshold <= d,
        binom.sf(upper_threshold - 1, d, theta_control_only),
        0.0,
    )
    return lower + upper


def exact_mcnemar_power(
    n_pairs: int,
    p_control: float,
    p_candidate: float,
    correlation: float,
    *,
    alpha: float = 0.05,
) -> float:
    """Unconditional power of the exact two-sided McNemar test."""

    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    _validate_design(alpha)
    cells = paired_joint_probabilities(p_control, p_candidate, correlation)
    q10 = cells["control_only"]
    q01 = cells["candidate_only"]
    discordance = q10 + q01
    if discordance == 0:
        return float(alpha if p_control == p_candidate else 0.0)
    theta = q10 / discordance
    d = np.arange(n_pairs + 1)
    conditional = _mcnemar_conditional_rejection_probability(d, theta, alpha)
    weights = binom.pmf(d, n_pairs, discordance)
    return float(np.sum(weights * conditional))


def approximate_mcnemar_required_n(
    p_control: float,
    p_candidate: float,
    correlation: float,
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> float:
    """Asymptotic paired sample-size approximation using discordant cells."""

    _validate_design(alpha, target_power)
    cells = paired_joint_probabilities(p_control, p_candidate, correlation)
    q10 = cells["control_only"]
    q01 = cells["candidate_only"]
    qd = q10 + q01
    delta = abs(q10 - q01)
    if delta == 0:
        raise ValueError("discordant probabilities imply no paired effect")
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    z_power = norm.ppf(target_power)
    numerator = (
        z_alpha * math.sqrt(qd)
        + z_power * math.sqrt(max(0.0, qd - delta**2))
    ) ** 2
    return float(numerator / delta**2)


def find_minimum_exact_mcnemar_n(
    p_control: float,
    p_candidate: float,
    correlation: float,
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
    maximum_n: int = 20_000,
) -> tuple[int, float]:
    """Find the first pair count whose exact McNemar power reaches target."""

    _validate_design(alpha, target_power)
    if maximum_n < 2:
        raise ValueError("maximum_n must be at least 2")
    approximate = approximate_mcnemar_required_n(
        p_control,
        p_candidate,
        correlation,
        alpha=alpha,
        target_power=target_power,
    )
    start = max(2, int(math.floor(approximate)) - 300)
    for n_pairs in range(start, maximum_n + 1):
        power = exact_mcnemar_power(
            n_pairs,
            p_control,
            p_candidate,
            correlation,
            alpha=alpha,
        )
        if power >= target_power:
            return n_pairs, power
    raise RuntimeError(f"target power was not reached by maximum_n={maximum_n}")


def simulate_paired_power(
    n_pairs: int,
    p_control: float,
    p_candidate: float,
    correlation: float,
    *,
    alpha: float = 0.05,
    repetitions: int = 10_000,
    seed: int = 20260810,
) -> MonteCarloEstimate:
    """Estimate exact-McNemar power using multinomial complete trials."""

    if n_pairs <= 0 or repetitions <= 0:
        raise ValueError("n_pairs and repetitions must be positive")
    cells = paired_joint_probabilities(p_control, p_candidate, correlation)
    probabilities = np.array(
        [
            cells["both_safe"],
            cells["candidate_only"],
            cells["control_only"],
            cells["both_collision"],
        ]
    )
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_pairs, probabilities, size=repetitions)
    candidate_only = counts[:, 1]
    control_only = counts[:, 2]
    discordant = candidate_only + control_only
    smaller = np.minimum(candidate_only, control_only)
    pvalues = np.ones(repetitions, dtype=float)
    nonzero = discordant > 0
    pvalues[nonzero] = np.minimum(
        1.0,
        2.0 * binom.cdf(smaller[nonzero], discordant[nonzero], 0.5),
    )
    return _mc_estimate(pvalues < alpha)


def design_effect(
    mean_cluster_size: float,
    icc: float,
    cluster_size_cv: float = 0.0,
) -> float:
    """Approximate design effect, allowing unequal cluster sizes."""

    if mean_cluster_size <= 0:
        raise ValueError("mean_cluster_size must be positive")
    _validate_probability(icc, "icc")
    if cluster_size_cv < 0:
        raise ValueError("cluster_size_cv cannot be negative")
    effective_size = (1.0 + cluster_size_cv**2) * mean_cluster_size
    return float(1.0 + (effective_size - 1.0) * icc)


def round_up_to_block(value: float, block_size: int) -> int:
    """Round a positive planning count up to a complete allocation block."""

    if value <= 0 or block_size <= 0:
        raise ValueError("value and block_size must be positive")
    return int(math.ceil(value / block_size) * block_size)


def representative_wilson_interval(
    n: int,
    assumed_rate: float,
    confidence: float = 0.95,
) -> tuple[int, float, float, float]:
    """Wilson interval using the nearest integer to the expected event count."""

    if n <= 0:
        raise ValueError("n must be positive")
    _validate_probability(assumed_rate, "assumed_rate")
    expected_events = int(round(n * assumed_rate))
    low, high = wilson_interval(expected_events, n, confidence)
    return expected_events, low, high, high - low


def normal_risk_difference_half_width(
    n_per_arm: int,
    p_control: float,
    p_candidate: float,
    confidence: float = 0.95,
) -> float:
    """Planning half-width for an independent two-proportion risk difference."""

    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")
    _validate_probability(p_control, "p_control")
    _validate_probability(p_candidate, "p_candidate")
    _validate_probability(confidence, "confidence", strict=True)
    z = norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    variance = (
        p_control * (1.0 - p_control)
        + p_candidate * (1.0 - p_candidate)
    ) / n_per_arm
    return float(z * math.sqrt(variance))


def minimum_n_for_risk_difference_precision(
    half_width: float,
    p_control: float,
    p_candidate: float,
    confidence: float = 0.95,
) -> int:
    """Normal-approximation sample size per arm for a target half-width."""

    if half_width <= 0:
        raise ValueError("half_width must be positive")
    _validate_probability(p_control, "p_control")
    _validate_probability(p_candidate, "p_candidate")
    _validate_probability(confidence, "confidence", strict=True)
    z = norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    variance_sum = (
        p_control * (1.0 - p_control)
        + p_candidate * (1.0 - p_candidate)
    )
    return int(math.ceil((z**2) * variance_sum / (half_width**2)))


def stable_stream_seed(master_seed: int, components: Iterable[object]) -> int:
    """Create a deterministic independent stream seed from labeled components."""

    import hashlib

    payload = "|".join([str(master_seed), *(str(item) for item in components)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)
