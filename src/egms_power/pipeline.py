"""End-to-end analysis, reporting, plotting, and provenance pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

from . import __version__
from .config import DISCLAIMER, canonical_protocol_json, load_protocol
from .statistics import (
    design_effect,
    exact_mcnemar_power,
    feasible_correlation_bounds,
    find_minimum_exact_mcnemar_n,
    minimum_n_for_risk_difference_precision,
    normal_risk_difference_half_width,
    paired_joint_probabilities,
    representative_wilson_interval,
    required_n_two_independent_proportions,
    round_up_to_block,
    simulate_paired_power,
    simulate_unpaired_power,
    simulate_wilson_coverage,
    stable_stream_seed,
    two_independent_proportions_power,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _generated_at_utc() -> str:
    """Return a standard reproducible-build timestamp when one is requested."""

    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch is None:
        return datetime.now(timezone.utc).isoformat()
    try:
        timestamp = int(source_date_epoch)
    except ValueError as exc:
        raise ValueError("SOURCE_DATE_EPOCH must be an integer number of seconds") from exc
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def prepare_output_directory(root: Path, requested: str | Path) -> Path:
    """Prepare a contained output directory without recursively deleting data."""

    root = root.resolve()
    output = Path(requested)
    if not output.is_absolute():
        output = (root / output).resolve()
    else:
        output = output.resolve()
    if output == root or root not in output.parents:
        raise ValueError(
            "Refusing unsafe output path. Output must be a child of the project root."
        )
    sentinel = output / ".egms_power_output"
    existing_files = [path for path in output.rglob("*") if path.is_file()] if output.exists() else []
    if existing_files and not sentinel.exists():
        raise ValueError(
            "Refusing to overwrite a non-empty directory without the EGMS power-analysis sentinel."
        )
    # Remove only files recorded by the prior run manifest. This prevents stale
    # scientific tables while preserving untracked user files.
    prior_manifest = output / "run_manifest.json"
    if sentinel.exists() and prior_manifest.is_file():
        try:
            recorded = json.loads(prior_manifest.read_text(encoding="utf-8")).get(
                "output_files", {}
            )
        except (json.JSONDecodeError, OSError):
            recorded = {}
        for relative in recorded:
            candidate = (root / relative).resolve()
            if output in candidate.parents and candidate.is_file():
                candidate.unlink()
        prior_manifest.unlink(missing_ok=True)
    for subdirectory in ["tables", "figures", "data"]:
        (output / subdirectory).mkdir(parents=True, exist_ok=True)
    sentinel.write_text(
        "Managed EGMS-Drive protocol power-analysis output directory.\n",
        encoding="utf-8",
    )
    return output


def _save_table(frame: pd.DataFrame, stem: Path) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        text = f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)
        return text.replace("|", r"\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(column) for column in frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stem.with_suffix(".tex").write_text(_simple_latex_table(frame), encoding="utf-8")


def _frame_to_markdown(frame: pd.DataFrame, *, float_digits: int | None = None) -> str:
    """Render a DataFrame as Markdown without the optional tabulate package."""

    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        if float_digits is not None and isinstance(value, (float, np.floating)):
            text = f"{value:.{float_digits}f}"
        else:
            text = str(value)
        return text.replace("|", r"\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(column) for column in frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _latex_escape(value: object) -> str:
    text = f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)
    for source, replacement in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ]:
        text = text.replace(source, replacement)
    return text


def _simple_latex_table(frame: pd.DataFrame) -> str:
    """Create dependency-free tabular LaTeX for release tables."""

    alignment = "l" * len(frame.columns)
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\hline"]
    lines.append(" & ".join(_latex_escape(column) for column in frame.columns) + r" \\")
    lines.append("\\hline")
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(_latex_escape(value) for value in row) + r" \\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def _style() -> None:
    """Apply a deterministic IEEE-style graphics profile.

    STIXGeneral is distributed with Matplotlib, has a Times-like appearance,
    and is therefore reproducible in both the release environment and Colab.
    Plot titles and prose footnotes are deliberately omitted; the manuscript
    captions carry all descriptive and interpretive text.
    """

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 10.5,
            "axes.labelsize": 12.5,
            "axes.labelpad": 6,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "black",
            "axes.grid": False,
            "axes.axisbelow": True,
            "grid.color": "#D6D6D6",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.70,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.pad": 4,
            "ytick.major.pad": 4,
            "legend.fontsize": 9.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "egms-drive-publication-v1",
        }
    )


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    paths: dict[str, Path] = {}
    try:
        for suffix, kwargs in [
            ("png", {"dpi": 600}),
            ("pdf", {}),
            ("svg", {}),
        ]:
            path = stem.with_suffix(f".{suffix}")
            temporary = path.with_name(path.name + ".tmp")
            fig.savefig(temporary, format=suffix, facecolor="white", **kwargs)
            paths[suffix] = temporary

        png, pdf, svg = (paths[key] for key in ("png", "pdf", "svg"))
        if png.stat().st_size < 1024 or not png.read_bytes().startswith(b"\x89PNG"):
            raise RuntimeError(f"Invalid PNG figure export: {png}")
        pdf_bytes = pdf.read_bytes()
        if len(pdf_bytes) < 1024 or not pdf_bytes.startswith(b"%PDF") or b"%%EOF" not in pdf_bytes[-2048:]:
            raise RuntimeError(f"Invalid PDF figure export: {pdf}")
        if svg.stat().st_size < 1024:
            raise RuntimeError(f"Invalid or empty SVG figure export: {svg}")
        ET.parse(svg)
        for suffix, temporary in paths.items():
            temporary.replace(stem.with_suffix(f".{suffix}"))
    finally:
        plt.close(fig)
        for temporary in paths.values():
            temporary.unlink(missing_ok=True)


def _format_axes(ax: plt.Axes, *, show_grid: bool = True) -> None:
    """Apply a clean boxed-axis treatment with consistent major ticks."""

    ax.grid(show_grid, which="major", axis="both")
    ax.tick_params(
        axis="both",
        which="major",
        top=False,
        right=False,
        bottom=True,
        left=True,
        labelsize=10.5,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)


def _format_colorbar(ax: plt.Axes) -> None:
    """Match colorbar labels and numerical ticks to the enlarged axis text."""

    colorbar = ax.collections[0].colorbar
    if colorbar is None:
        return
    colorbar.ax.tick_params(labelsize=10.5, length=4.5, width=0.9, direction="out")
    colorbar.ax.yaxis.label.set_size(12.5)


def _assumptions_table(protocol: dict[str, Any]) -> pd.DataFrame:
    primary = protocol["primary_design"]
    simulation = protocol["simulation"]
    scenario = protocol["scenario_matrix"]
    return pd.DataFrame(
        [
            ("Study status", protocol["project"]["status"], "Prospective design only"),
            ("Primary endpoint", primary["endpoint"], primary["endpoint_definition"]),
            ("Estimand", primary["estimand"], "Risk difference"),
            ("Planning control probability", primary["control_rate_assumption"], "Assumed, not observed"),
            ("Planning candidate probability", primary["candidate_rate_assumption"], "Assumed, not observed"),
            ("Two-sided alpha", primary["alpha"], "One confirmatory primary contrast"),
            ("Target power", primary["target_power"], "Prospective design target"),
            ("Monte Carlo repetitions", simulation["canonical_repetitions"], "Complete simulated trials"),
            ("Master seed", simulation["master_seed"], "Deterministic labeled RNG streams"),
            ("Scenario allocation block", scenario["allocation_block"], "5 scenarios × 3 environments × 3 densities"),
        ],
        columns=["quantity", "value", "interpretation"],
    )


def _independent_results(protocol: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    primary = protocol["primary_design"]
    simulation = protocol["simulation"]
    p0 = float(primary["control_rate_assumption"])
    p1 = float(primary["candidate_rate_assumption"])
    alpha = float(primary["alpha"])
    target = float(primary["target_power"])
    repetitions = int(simulation["canonical_repetitions"])
    master_seed = int(simulation["master_seed"])
    raw_n, required_n = required_n_two_independent_proportions(
        p0, p1, alpha=alpha, target_power=target
    )
    rows: list[dict[str, Any]] = []
    for n in simulation["sample_sizes"]:
        analytic = two_independent_proportions_power(n, p0, p1, alpha=alpha)
        estimate = simulate_unpaired_power(
            int(n),
            p0,
            p1,
            alpha=alpha,
            repetitions=repetitions,
            seed=stable_stream_seed(master_seed, ["unpaired", n, p0, p1, alpha]),
        )
        rows.append(
            {
                "episodes_per_arm": int(n),
                "analytic_power": analytic,
                "mc_power": estimate.estimate,
                "mcse": estimate.mcse,
                "mc_ci_low": estimate.ci_low,
                "mc_ci_high": estimate.ci_high,
                "mc_rejections": estimate.rejections,
                "mc_repetitions": estimate.repetitions,
            }
        )
    summary = {
        "raw_required_n": raw_n,
        "required_n": required_n,
        "power_150": two_independent_proportions_power(150, p0, p1, alpha=alpha),
        "power_450": two_independent_proportions_power(450, p0, p1, alpha=alpha),
        "power_required": two_independent_proportions_power(required_n, p0, p1, alpha=alpha),
    }
    return pd.DataFrame(rows), summary


def _null_calibration(protocol: dict[str, Any]) -> pd.DataFrame:
    primary = protocol["primary_design"]
    simulation = protocol["simulation"]
    p_null = float(primary["control_rate_assumption"])
    alpha = float(primary["alpha"])
    repetitions = int(simulation["canonical_repetitions"])
    master_seed = int(simulation["master_seed"])
    sample_sizes = [150, 450, 749, 900, 1200]
    rows: list[dict[str, Any]] = []
    for n in sample_sizes:
        estimate = simulate_unpaired_power(
            n,
            p_null,
            p_null,
            alpha=alpha,
            repetitions=repetitions,
            seed=stable_stream_seed(master_seed, ["null", n, p_null, alpha]),
        )
        rows.append(
            {
                "episodes_per_arm": n,
                "null_rate_both_arms": p_null,
                "nominal_alpha": alpha,
                "estimated_type1_error": estimate.estimate,
                "mcse": estimate.mcse,
                "mc_ci_low": estimate.ci_low,
                "mc_ci_high": estimate.ci_high,
                "within_3mcse_plus_0_005": abs(estimate.estimate - alpha)
                <= 3.0 * estimate.mcse + 0.005,
            }
        )
    return pd.DataFrame(rows)


def _coverage_calibration(protocol: dict[str, Any]) -> pd.DataFrame:
    primary = protocol["primary_design"]
    simulation = protocol["simulation"]
    repetitions = int(simulation["canonical_repetitions"])
    master_seed = int(simulation["master_seed"])
    confidence = float(simulation["monte_carlo_confidence"])
    rates = [
        float(primary["candidate_rate_assumption"]),
        float(primary["control_rate_assumption"]),
    ]
    rows: list[dict[str, Any]] = []
    for rate in rates:
        for n in [150, 450, 749, 1200]:
            estimate = simulate_wilson_coverage(
                n,
                rate,
                confidence=confidence,
                repetitions=repetitions,
                seed=stable_stream_seed(master_seed, ["coverage", rate, n, confidence]),
            )
            rows.append(
                {
                    "true_probability": rate,
                    "episodes": n,
                    "nominal_coverage": confidence,
                    "estimated_coverage": estimate.estimate,
                    "mcse": estimate.mcse,
                    "mc_ci_low": estimate.ci_low,
                    "mc_ci_high": estimate.ci_high,
                    "coverage_in_prespecified_range": 0.93 <= estimate.estimate <= 0.98,
                }
            )
    return pd.DataFrame(rows)


def _paired_results(protocol: dict[str, Any]) -> tuple[pd.DataFrame, tuple[float, float]]:
    primary = protocol["primary_design"]
    paired = protocol["paired_design"]
    simulation = protocol["simulation"]
    p0 = float(primary["control_rate_assumption"])
    p1 = float(primary["candidate_rate_assumption"])
    alpha = float(primary["alpha"])
    target = float(primary["target_power"])
    repetitions = int(simulation["canonical_repetitions"])
    master_seed = int(simulation["master_seed"])
    bounds = feasible_correlation_bounds(p0, p1)
    rows: list[dict[str, Any]] = []
    for correlation in paired["correlations"]:
        correlation = float(correlation)
        cells = paired_joint_probabilities(p0, p1, correlation)
        minimum_n, power_at_minimum = find_minimum_exact_mcnemar_n(
            p0,
            p1,
            correlation,
            alpha=alpha,
            target_power=target,
        )
        mc = simulate_paired_power(
            minimum_n,
            p0,
            p1,
            correlation,
            alpha=alpha,
            repetitions=repetitions,
            seed=stable_stream_seed(
                master_seed, ["paired", correlation, minimum_n, p0, p1, alpha]
            ),
        )
        rows.append(
            {
                "correlation": correlation,
                "pi00_both_safe": cells["both_safe"],
                "pi01_candidate_only_collision": cells["candidate_only"],
                "pi10_control_only_collision": cells["control_only"],
                "pi11_both_collision": cells["both_collision"],
                "discordant_probability": cells["candidate_only"] + cells["control_only"],
                "exact_power_at_450_pairs": exact_mcnemar_power(
                    450, p0, p1, correlation, alpha=alpha
                ),
                "minimum_pairs_exact": minimum_n,
                "exact_power_at_minimum": power_at_minimum,
                "mc_power_at_minimum": mc.estimate,
                "mcse": mc.mcse,
                "mc_ci_low": mc.ci_low,
                "mc_ci_high": mc.ci_high,
            }
        )
    return pd.DataFrame(rows), bounds


def _cluster_results(
    protocol: dict[str, Any], paired_frame: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    primary = protocol["primary_design"]
    paired = protocol["paired_design"]
    clustered = protocol["clustered_design"]
    scenario = protocol["scenario_matrix"]
    p0 = float(primary["control_rate_assumption"])
    p1 = float(primary["candidate_rate_assumption"])
    alpha = float(primary["alpha"])
    planning_correlation = float(paired["planning_correlation"])
    row = paired_frame.loc[
        np.isclose(paired_frame["correlation"], planning_correlation)
    ]
    if row.empty:
        base_pairs, _ = find_minimum_exact_mcnemar_n(
            p0,
            p1,
            planning_correlation,
            alpha=alpha,
            target_power=float(primary["target_power"]),
        )
    else:
        base_pairs = int(row.iloc[0]["minimum_pairs_exact"])
    mean_cluster = float(clustered["mean_cluster_size"])
    invalid_fraction = float(clustered["invalid_episode_fraction"])
    block = int(scenario["allocation_block"])
    rows: list[dict[str, Any]] = []
    for cv in clustered["cluster_size_cv_values"]:
        for icc in clustered["icc_values"]:
            effect = design_effect(mean_cluster, float(icc), float(cv))
            raw_planned = base_pairs * effect / (1.0 - invalid_fraction)
            planned = round_up_to_block(raw_planned, block)
            effective_pairs = planned * (1.0 - invalid_fraction) / effect
            rows.append(
                {
                    "cluster_size_cv": float(cv),
                    "icc": float(icc),
                    "design_effect": effect,
                    "independent_equivalent_pairs_required": base_pairs,
                    "raw_pairs_with_clustering_and_invalid_allowance": raw_planned,
                    "planned_pairs": planned,
                    "episodes_per_condition": planned,
                    "total_episodes_primary_two_condition": 2 * planned,
                    "seeds_per_45_cell_matrix": planned // block,
                    "effective_pairs_after_adjustment": effective_pairs,
                    "approx_exact_power_at_effective_pairs": exact_mcnemar_power(
                        max(1, int(math.floor(effective_pairs))),
                        p0,
                        p1,
                        planning_correlation,
                        alpha=alpha,
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    selected = frame[
        np.isclose(frame["cluster_size_cv"], 0.0)
        & np.isclose(frame["icc"], float(clustered["planning_icc"]))
    ].iloc[0]
    recommendation = {
        "planning_correlation": planning_correlation,
        "planning_icc": float(clustered["planning_icc"]),
        "cluster_size_cv": 0.0,
        "invalid_episode_fraction": invalid_fraction,
        "base_exact_pairs": base_pairs,
        "planned_pairs": int(selected["planned_pairs"]),
        "episodes_per_condition": int(selected["episodes_per_condition"]),
        "total_episodes_primary_two_condition": int(selected["total_episodes_primary_two_condition"]),
        "seeds_per_matrix_cell": int(selected["seeds_per_45_cell_matrix"]),
    }
    return frame, recommendation


def _ablation_results(protocol: dict[str, Any]) -> pd.DataFrame:
    primary = protocol["primary_design"]
    ablation = protocol["ablation_planning"]
    alpha_primary = float(primary["alpha"])
    alpha_multiplicity = float(protocol["multiplicity"]["conservative_ablation_alpha"])
    target = float(primary["target_power"])
    rows: list[dict[str, Any]] = []
    for scenario in ablation["collision_scenarios"]:
        p0 = float(scenario["control_rate"])
        p1 = float(scenario["candidate_rate"])
        for alpha, label in [
            (alpha_primary, "single_prespecified_contrast"),
            (alpha_multiplicity, "four_comparison_bonferroni_planning_bound"),
        ]:
            raw_n, n = required_n_two_independent_proportions(
                p0, p1, alpha=alpha, target_power=target
            )
            rows.append(
                {
                    "scenario": scenario["label"],
                    "control_rate_assumption": p0,
                    "candidate_rate_assumption": p1,
                    "absolute_difference": abs(p0 - p1),
                    "alpha": alpha,
                    "multiplicity_interpretation": label,
                    "raw_required_episodes_per_arm": raw_n,
                    "required_episodes_per_arm": n,
                    "power_at_150_per_arm": two_independent_proportions_power(
                        150, p0, p1, alpha=alpha
                    ),
                    "power_at_450_per_arm": two_independent_proportions_power(
                        450, p0, p1, alpha=alpha
                    ),
                }
            )
    return pd.DataFrame(rows)


def _sensitivity_results(protocol: dict[str, Any]) -> pd.DataFrame:
    primary = protocol["primary_design"]
    sensitivity = protocol["sensitivity"]
    alpha = float(primary["alpha"])
    rows: list[dict[str, Any]] = []
    for control_rate in sensitivity["control_rates"]:
        for reduction in sensitivity["absolute_reductions"]:
            candidate_rate = float(control_rate) - float(reduction)
            if candidate_rate <= 0:
                continue
            for target_power in sensitivity["target_powers"]:
                raw_n, n = required_n_two_independent_proportions(
                    float(control_rate),
                    candidate_rate,
                    alpha=alpha,
                    target_power=float(target_power),
                )
                rows.append(
                    {
                        "control_rate_assumption": float(control_rate),
                        "candidate_rate_assumption": candidate_rate,
                        "absolute_reduction": float(reduction),
                        "target_power": float(target_power),
                        "raw_required_episodes_per_arm": raw_n,
                        "required_episodes_per_arm": n,
                    }
                )
    return pd.DataFrame(rows)


def _precision_results(protocol: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = protocol["primary_design"]
    sensitivity = protocol["sensitivity"]
    p0 = float(primary["control_rate_assumption"])
    p1 = float(primary["candidate_rate_assumption"])
    rows: list[dict[str, Any]] = []
    for n in sensitivity["representative_ci_sample_sizes"]:
        for label, rate in [("control", p0), ("candidate", p1)]:
            events, low, high, width = representative_wilson_interval(int(n), rate)
            rows.append(
                {
                    "episodes_per_arm": int(n),
                    "arm": label,
                    "assumed_rate": rate,
                    "representative_event_count": events,
                    "wilson_ci_low": low,
                    "wilson_ci_high": high,
                    "wilson_ci_width": width,
                    "risk_difference_half_width": normal_risk_difference_half_width(
                        int(n), p0, p1
                    ),
                }
            )
    precision_rows = []
    for half_width in sensitivity["precision_half_widths"]:
        precision_rows.append(
            {
                "target_risk_difference_half_width": float(half_width),
                "required_episodes_per_arm": minimum_n_for_risk_difference_precision(
                    float(half_width), p0, p1
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(precision_rows)


def _allocation_results(protocol: dict[str, Any]) -> pd.DataFrame:
    primary = protocol["primary_design"]
    clustered = protocol["clustered_design"]
    scenario = protocol["scenario_matrix"]
    p0 = float(primary["control_rate_assumption"])
    p1 = float(primary["candidate_rate_assumption"])
    alpha = float(primary["alpha"])
    block = int(scenario["allocation_block"])
    effect = design_effect(
        float(clustered["mean_cluster_size"]),
        float(clustered["planning_icc"]),
        0.0,
    )
    rows: list[dict[str, Any]] = []
    for seeds in range(3, 41):
        total = block * seeds
        effective = total / effect
        rows.append(
            {
                "seeds_per_scenario_environment_density_cell": seeds,
                "episodes_per_condition": total,
                "independent_power": two_independent_proportions_power(
                    total, p0, p1, alpha=alpha
                ),
                "effective_n_at_planning_icc": effective,
                "design_effect_approx_power": two_independent_proportions_power(
                    effective, p0, p1, alpha=alpha
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_all(
    output: Path,
    protocol: dict[str, Any],
    independent: pd.DataFrame,
    paired: pd.DataFrame,
    clustered: pd.DataFrame,
    sensitivity: pd.DataFrame,
    precision: pd.DataFrame,
    allocation: pd.DataFrame,
) -> None:
    _style()
    primary = protocol["primary_design"]
    simulation = protocol["simulation"]
    p0 = float(primary["control_rate_assumption"])
    p1 = float(primary["candidate_rate_assumption"])
    alpha = float(primary["alpha"])
    target = float(primary["target_power"])

    grid = np.arange(
        int(simulation["power_curve_start"]),
        int(simulation["power_curve_stop"]) + 1,
        int(simulation["power_curve_step"]),
    )
    power_grid = pd.DataFrame(
        {
            "episodes_per_arm": grid,
            "analytic_power": [
                two_independent_proportions_power(n, p0, p1, alpha=alpha) for n in grid
            ],
        }
    )
    power_grid.to_csv(output / "data" / "figure1_unpaired_power_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.20, 4.744))
    fig.subplots_adjust(left=0.165, right=0.975, bottom=0.175, top=0.965)
    ax.plot(grid, power_grid["analytic_power"], color="#174A7E", lw=2.2, label="Analytic power")
    ax.scatter(
        independent["episodes_per_arm"],
        independent["mc_power"],
        color="#D1495B",
        s=28,
        zorder=3,
        label="Monte Carlo estimate",
    )
    ax.axhline(target, color="#2A9D8F", ls="--", lw=1.4, label="80% target")
    for n in [150, 450, 749]:
        power = two_independent_proportions_power(n, p0, p1, alpha=alpha)
        ax.annotate(
            f"{n}: {power:.3f}",
            (n, power),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=10,
        )
    ax.set(
        xlabel=r"Number of episodes per arm, $n$",
        ylabel=r"Statistical power, $1-\beta$",
        xlim=(0, 2000),
        ylim=(0, 1.02),
    )
    ax.set_xticks(np.arange(0, 2001, 250))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    _format_axes(ax)
    ax.legend(loc="lower right")
    _save_figure(fig, output / "figures" / "figure1_unpaired_power_curve")

    paired.to_csv(output / "data" / "figure2_paired_correlation.csv", index=False)
    fig, ax1 = plt.subplots(figsize=(6.24, 4.744))
    fig.subplots_adjust(left=0.215, right=0.975, bottom=0.175, top=0.965)
    ax1.plot(
        paired["correlation"],
        paired["minimum_pairs_exact"],
        marker="o",
        color="#6A4C93",
        lw=2.2,
        ms=4.5,
    )
    ax1.set(
        xlabel=r"Cross-method collision correlation, $\rho$",
        ylabel=r"Minimum number of pairs for 80% power, $n$",
    )
    correlation_ticks = paired["correlation"].to_numpy()
    ax1.set_xticks(correlation_ticks)
    ax1.set_xticklabels([f"{value:g}" for value in correlation_ticks])
    _format_axes(ax1)
    _save_figure(fig, output / "figures" / "figure2_paired_correlation")

    clustered.to_csv(output / "data" / "figure3_cluster_sensitivity.csv", index=False)
    heat = clustered.pivot(
        index="icc", columns="cluster_size_cv", values="episodes_per_condition"
    )
    fig, ax = plt.subplots(figsize=(5.82, 4.838))
    fig.subplots_adjust(left=0.21, right=0.90, bottom=0.175, top=0.965)
    sns.heatmap(
        heat,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        linewidths=0.6,
        linecolor="#E1E1E1",
        annot_kws={"fontsize": 10.5},
        cbar_kws={"label": r"Number of episodes per condition, $n$", "pad": 0.035},
        ax=ax,
    )
    ax.set(
        xlabel=r"Cluster-size coefficient of variation, $\mathrm{CV}$",
        ylabel=r"Intracluster correlation, $\rho_{\mathrm{ICC}}$",
    )
    ax.set_xticklabels(["0", "0.25", "0.50"], rotation=0)
    ax.set_yticklabels(["0", "0.01", "0.05", "0.10", "0.20"], rotation=0)
    _format_axes(ax, show_grid=False)
    _format_colorbar(ax)
    _save_figure(fig, output / "figures" / "figure3_cluster_sensitivity")

    heat80 = sensitivity[np.isclose(sensitivity["target_power"], 0.80)].pivot(
        index="control_rate_assumption",
        columns="absolute_reduction",
        values="required_episodes_per_arm",
    )
    heat80.to_csv(output / "data" / "figure4_event_rate_sensitivity.csv")
    heat80_display = heat80.copy()
    heat80_display.columns = [100 * float(value) for value in heat80_display.columns]
    heat80_display.index = [100 * float(value) for value in heat80_display.index]
    fig, ax = plt.subplots(figsize=(5.98, 4.932))
    fig.subplots_adjust(left=0.19, right=0.90, bottom=0.175, top=0.965)
    sns.heatmap(
        heat80_display,
        annot=True,
        fmt=".0f",
        cmap="mako_r",
        linewidths=0.6,
        linecolor="#E1E1E1",
        annot_kws={"fontsize": 10.5},
        cbar_kws={"label": r"Number of episodes per arm, $n$", "pad": 0.035},
        ax=ax,
    )
    ax.set(
        xlabel=r"Absolute risk reduction, $\Delta p$ (percentage points)",
        ylabel=r"Control collision risk, $p_0$ (%)",
    )
    ax.set_xticklabels(["1", "2", "3", "4"], rotation=0)
    ax.set_yticklabels(["3", "6", "10", "15"], rotation=0)
    _format_axes(ax, show_grid=False)
    _format_colorbar(ax)
    _save_figure(fig, output / "figures" / "figure4_event_rate_sensitivity")

    precision_curve = pd.DataFrame(
        {
            "episodes_per_arm": grid,
            "risk_difference_half_width": [
                normal_risk_difference_half_width(n, p0, p1) for n in grid
            ],
        }
    )
    precision_curve.to_csv(output / "data" / "figure5_precision_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.084, 4.744))
    fig.subplots_adjust(left=0.225, right=0.975, bottom=0.175, top=0.965)
    ax.plot(
        precision_curve["episodes_per_arm"],
        100 * precision_curve["risk_difference_half_width"],
        color="#E76F51",
        lw=2.2,
    )
    ax.set(
        xlabel=r"Number of episodes per arm, $n$",
        ylabel="95% risk-difference half-width (percentage points)",
        xlim=(0, 2000),
    )
    ax.set_xticks(np.arange(0, 2001, 250))
    _format_axes(ax)
    _save_figure(fig, output / "figures" / "figure5_precision_curve")

    allocation.to_csv(output / "data" / "figure6_allocation_by_seed.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.20, 4.744))
    fig.subplots_adjust(left=0.165, right=0.975, bottom=0.175, top=0.965)
    ax.plot(
        allocation["seeds_per_scenario_environment_density_cell"],
        allocation["independent_power"],
        lw=2.1,
        label="Independent approximation",
        color="#264653",
        marker="o",
        markevery=4,
        ms=4,
        mfc="white",
    )
    ax.plot(
        allocation["seeds_per_scenario_environment_density_cell"],
        allocation["design_effect_approx_power"],
        lw=2.1,
        label=f"ICC={protocol['clustered_design']['planning_icc']:.2f} design-effect approximation",
        color="#F4A261",
        ls="--",
        marker="s",
        markevery=4,
        ms=4,
        mfc="white",
    )
    ax.axhline(target, color="#2A9D8F", ls=":", lw=1.5, label="80% target")
    ax.set(
        xlabel=r"Number of seeds per scenario–environment–density cell, $s$",
        ylabel=r"Approximate power, $1-\beta$",
        xlim=(3, 40),
        ylim=(0, 1.02),
    )
    ax.set_xticks(np.arange(5, 41, 5))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    _format_axes(ax)
    ax.legend(loc="lower right")
    _save_figure(fig, output / "figures" / "figure6_allocation_by_seed")


def _manuscript_ready_results(
    protocol: dict[str, Any],
    independent_summary: dict[str, float],
    paired: pd.DataFrame,
    recommendation: dict[str, Any],
    ablation: pd.DataFrame,
) -> str:
    rho0 = paired.loc[np.isclose(paired["correlation"], 0.0)].iloc[0]
    rare = ablation[
        (ablation["scenario"] == "two_events_in_150_gap")
        & np.isclose(ablation["alpha"], 0.05)
    ].iloc[0]
    return f"""# Manuscript-ready prospective power-analysis results

> {DISCLAIMER}

Under the prespecified planning scenario of a 6% episode-level collision probability in the control condition and 3% in the candidate condition, a two-sided equal-allocation score-test calculation with alpha = 0.05 and 80% power required {independent_summary['raw_required_n']:.3f} episodes per arm, rounded upward to **{int(independent_summary['required_n'])} episodes per arm** ({2 * int(independent_summary['required_n']):,} total). Under the same independence assumptions, 450 and 150 episodes per arm provided approximate powers of **{independent_summary['power_450']:.3f}** and **{independent_summary['power_150']:.3f}**, respectively. These quantities validate Reviewer 5's concern that the original 450-episode main comparison and 150-episode ablation blocks were underpowered for the assumed 3-percentage-point difference.

Pairing did not have a single fixed benefit because the marginal event probabilities do not determine the paired joint distribution. With zero assumed cross-method collision correlation, the exact two-sided McNemar design required **{int(rho0['minimum_pairs_exact'])} pairs**; at 450 pairs, exact power was **{rho0['exact_power_at_450_pairs']:.3f}**. Across the prespecified feasible correlation grid, the exact requirement ranged from **{int(paired['minimum_pairs_exact'].min())} to {int(paired['minimum_pairs_exact'].max())} pairs**. These are conditional sensitivity results, not estimates of the correlation that a future implementation will exhibit.

The provisional cluster-aware allocation uses cross-method correlation {recommendation['planning_correlation']:.2f}, within-scenario-seed ICC {recommendation['planning_icc']:.2f}, mean cluster size 9, equal cluster sizes, a 5% invalid-episode allowance, and complete 45-episode allocation blocks. It yields **{recommendation['planned_pairs']:,} paired scenario blocks**, corresponding to **{recommendation['episodes_per_condition']:,} episodes per condition** and **{recommendation['total_episodes_primary_two_condition']:,} episodes across the two-condition primary comparison**, or **{recommendation['seeds_per_matrix_cell']} seeds per scenario–environment–density cell**. This value is a conservative planning scenario and must be updated using a blinded pilot estimate of the discordant probabilities and ICC before formal execution.

The 6.000% versus 4.667% ablation contrast seen in the former 150-episode surrogate table corresponds to only two events and would require approximately **{int(rare['required_episodes_per_arm']):,} independent episodes per arm** for 80% power at alpha = 0.05 under those assumed rates. Therefore, collision should not be used to claim individual module effects from 150 episodes. Each learned ablation must be retrained independently across multiple training seeds and assigned a mechanism-specific primary endpoint; collision remains a separately powered safety endpoint.

No values in this report are empirical EGMS-Drive performance results. The future closed-loop study must retain pair IDs, scenario and training-seed clusters, independently generated outcomes in both discordant directions, prespecified exclusions, and the frozen analysis code.
"""


def _results_report(
    protocol: dict[str, Any],
    assumptions: pd.DataFrame,
    independent: pd.DataFrame,
    independent_summary: dict[str, float],
    null: pd.DataFrame,
    coverage: pd.DataFrame,
    paired: pd.DataFrame,
    bounds: tuple[float, float],
    clustered: pd.DataFrame,
    recommendation: dict[str, Any],
    ablation: pd.DataFrame,
    precision_targets: pd.DataFrame,
) -> str:
    return f"""# EGMS-Drive prospective protocol and power-analysis results

> **Design assumptions and power-analysis results only. No empirical EGMS-Drive performance is reported.**
>
> {DISCLAIMER}

## 1. Prespecified assumptions

{_frame_to_markdown(assumptions)}

The assumed 6% and 3% collision probabilities are planning inputs requested by the review analysis. They were not measured from CARLA, public datasets, a trained model, or real vehicles.

## 2. Independent two-arm design

The pooled-null/unpooled-alternative calculation gives a raw requirement of **{independent_summary['raw_required_n']:.3f}**, rounded to **{int(independent_summary['required_n'])} episodes per arm**. The original allocations would provide approximately **{independent_summary['power_450']:.3f} power at 450/arm** and **{independent_summary['power_150']:.3f} power at 150/arm**.

{_frame_to_markdown(independent, float_digits=6)}

## 3. Null calibration

The complete-trial Monte Carlo simulation was also run under p_control = p_candidate = 0.06. The rejection proportions below assess statistical calibration; they are not method comparisons.

{_frame_to_markdown(null, float_digits=6)}

## 4. Wilson-interval coverage calibration

The prospective binomial interval implementation was evaluated through complete samples at both assumed event rates. Coverage is itself a Monte Carlo proportion and includes MCSE and a Wilson Monte Carlo interval.

{_frame_to_markdown(coverage, float_digits=6)}

## 5. Paired exact-McNemar sensitivity

For the 6% and 3% marginals, the feasible Pearson-correlation range is **[{bounds[0]:.6f}, {bounds[1]:.6f}]**. Each row reports all four joint probabilities, allowing both directions of discordance.

{_frame_to_markdown(paired, float_digits=6)}

## 6. Cluster and incomplete-episode sensitivity

The planning ICC is distinct from the cross-method paired correlation. The table uses the approximate design effect and rounds upward to complete 45-episode scenario blocks.

{_frame_to_markdown(clustered, float_digits=6)}

Provisional scenario: **{recommendation['planned_pairs']:,} paired scenario blocks**, equal to **{recommendation['episodes_per_condition']:,} episodes per condition** and **{recommendation['total_episodes_primary_two_condition']:,} total episodes for the primary two-condition contrast** ({recommendation['seeds_per_matrix_cell']} seeds per scenario–environment–density cell), under correlation {recommendation['planning_correlation']:.2f}, ICC {recommendation['planning_icc']:.2f}, and 5% invalid episodes. A blinded pilot should replace these dependence assumptions.

## 7. Rare-event and ablation planning

{_frame_to_markdown(ablation, float_digits=6)}

The four module ablations have separate mechanism-specific endpoints in `docs/statistical_analysis_plan.md`. A 150-episode block is not treated as adequate evidence of a collision-rate module effect.

## 8. Precision targets

{_frame_to_markdown(precision_targets, float_digits=6)}

## Interpretation boundary

The outputs answer **how many future episodes may be needed under stated assumptions**. They do not answer how EGMS-Drive performs. The repository deliberately contains no simulated F1, latency, VRAM, LLM consistency, or method-ranking outputs.
"""


def _validation_checks(
    protocol: dict[str, Any],
    independent_summary: dict[str, float],
    independent: pd.DataFrame,
    null: pd.DataFrame,
    coverage: pd.DataFrame,
    paired: pd.DataFrame,
    report_text: str,
) -> pd.DataFrame:
    p0 = float(protocol["primary_design"]["control_rate_assumption"])
    p1 = float(protocol["primary_design"]["candidate_rate_assumption"])
    alpha = float(protocol["primary_design"]["alpha"])
    lower, upper = feasible_correlation_bounds(p0, p1)
    expected = [
        (1, "Protocol schema loaded", True, protocol["project"]["version"]),
        (2, "Prospective-only status", protocol["project"]["status"] == "prospective_design_only", protocol["project"]["status"]),
        (3, "No empirical data flag", protocol["project"]["empirical_data_present"] is False, str(protocol["project"]["empirical_data_present"])),
        (4, "No empirical performance claims flag", protocol["project"]["empirical_performance_claims"] is False, str(protocol["project"]["empirical_performance_claims"])),
        (5, "Primary endpoint defined", bool(protocol["primary_design"]["endpoint"]), protocol["primary_design"]["endpoint"]),
        (6, "Estimand defined", bool(protocol["primary_design"]["estimand"]), protocol["primary_design"]["estimand"]),
        (7, "Alpha valid", 0 < alpha < 1, f"alpha={alpha}"),
        (8, "Target power valid", 0 < protocol["primary_design"]["target_power"] < 1, f"power={protocol['primary_design']['target_power']}"),
        (9, "Two-sided analysis", protocol["primary_design"]["sidedness"] == "two-sided", protocol["primary_design"]["sidedness"]),
        (10, "Probability bounds", 0 < p1 < 1 and 0 < p0 < 1, f"p0={p0}, p1={p1}"),
        (11, "Nonzero planning effect", not math.isclose(p0, p1), f"difference={p0-p1:.6f}"),
        (12, "Headline raw sample-size check", abs(independent_summary["raw_required_n"] - 748.388) < 0.01, f"raw_n={independent_summary['raw_required_n']:.6f}"),
        (13, "Headline ceiling sample-size check", independent_summary["required_n"] == 749, f"n={independent_summary['required_n']}"),
        (14, "450-episode power check", abs(independent_summary["power_450"] - 0.583696) < 1e-5, f"power={independent_summary['power_450']:.6f}"),
        (15, "150-episode power check", abs(independent_summary["power_150"] - 0.239938) < 1e-5, f"power={independent_summary['power_150']:.6f}"),
        (16, "Complete-trial simulation", independent["mc_repetitions"].min() >= 10000, f"min repetitions={independent['mc_repetitions'].min()}"),
        (17, "Monte Carlo uncertainty reported", independent["mcse"].notna().all(), "MCSE and Wilson CI columns present"),
        (18, "Monte Carlo precision", independent["mcse"].max() <= 0.005, f"max MCSE={independent['mcse'].max():.6f}"),
        (19, "Null calibration scenarios", len(null) >= 3, f"rows={len(null)}"),
        (20, "Null calibration within tolerance", bool(null["within_3mcse_plus_0_005"].all()), f"range={null['estimated_type1_error'].min():.4f}-{null['estimated_type1_error'].max():.4f}"),
        (21, "Wilson coverage calibration", bool(coverage["coverage_in_prespecified_range"].all()), f"range={coverage['estimated_coverage'].min():.4f}-{coverage['estimated_coverage'].max():.4f}"),
        (22, "Paired distributions valid", bool((paired[["pi00_both_safe", "pi01_candidate_only_collision", "pi10_control_only_collision", "pi11_both_collision"]] >= 0).all().all()) and bool(np.allclose(paired[["pi00_both_safe", "pi01_candidate_only_collision", "pi10_control_only_collision", "pi11_both_collision"]].sum(axis=1), 1.0)), "nonnegative cells summing to one"),
        (23, "Control marginal recovered", bool(np.allclose(paired["pi10_control_only_collision"] + paired["pi11_both_collision"], p0)), "pi10+pi11=p_control"),
        (24, "Candidate marginal recovered", bool(np.allclose(paired["pi01_candidate_only_collision"] + paired["pi11_both_collision"], p1)), "pi01+pi11=p_candidate"),
        (25, "Both discordant directions possible", bool((paired["pi10_control_only_collision"] > 0).all() and (paired["pi01_candidate_only_collision"] > 0).all()), "pi10>0 and pi01>0"),
        (26, "Exact paired test specified", protocol["primary_design"]["paired_test"] == "exact_mcnemar", protocol["primary_design"]["paired_test"]),
        (27, "ICC range valid", all(0 <= value < 1 for value in protocol["clustered_design"]["icc_values"]), str(protocol["clustered_design"]["icc_values"])),
        (28, "Positive cluster size", protocol["clustered_design"]["mean_cluster_size"] > 0, str(protocol["clustered_design"]["mean_cluster_size"])),
        (29, "Primary comparison count", protocol["multiplicity"]["primary_comparisons"] == 1, str(protocol["multiplicity"]["primary_comparisons"])),
        (30, "Secondary multiplicity plan", protocol["multiplicity"]["secondary_adjustment"] == "holm", protocol["multiplicity"]["secondary_adjustment"]),
        (31, "Scenario allocation complete", protocol["scenario_matrix"]["allocation_block"] == protocol["scenario_matrix"]["scenario_families"] * protocol["scenario_matrix"]["environments"] * protocol["scenario_matrix"]["traffic_densities"], f"block={protocol['scenario_matrix']['allocation_block']}"),
        (32, "Interpretation disclaimer present", "not empirical" in report_text.lower() or "no empirical" in report_text.lower(), "report disclaimer found"),
        (33, "Legacy performance claims absent", all(term not in report_text.lower() for term in ["f1 of 0.868", "reduced collisions by 51.9", "achieved a collision rate of 2.9"]), "claim guard passed"),
    ]
    return pd.DataFrame(expected, columns=["check_id", "check", "passed", "evidence"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(
    root: Path,
    output: Path,
    protocol: dict[str, Any],
    validation: pd.DataFrame,
) -> dict[str, Any]:
    source_files = sorted((root / "src").rglob("*.py"))
    output_files = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "run_manifest.json"
    )
    manifest = {
        "project": protocol["project"]["title"],
        "version": __version__,
        "status": protocol["project"]["status"],
        "generated_at_utc": _generated_at_utc(),
        "disclaimer": DISCLAIMER,
        "protocol_sha256": hashlib.sha256(canonical_protocol_json(protocol).encode("utf-8")).hexdigest(),
        "master_seed": protocol["simulation"]["master_seed"],
        "canonical_repetitions": protocol["simulation"]["canonical_repetitions"],
        "validation_checks_passed": int(validation["passed"].sum()),
        "validation_checks_total": int(len(validation)),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
        "source_files": {
            str(path.relative_to(root)): _sha256(path) for path in source_files
        },
        "output_files": {
            str(path.relative_to(root)): _sha256(path) for path in output_files
        },
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_power_analysis(
    config_path: str | Path,
    output_dir: str | Path = "outputs",
) -> dict[str, Any]:
    """Run the canonical protocol and power-analysis workflow."""

    root = project_root()
    protocol = load_protocol(config_path)
    output = prepare_output_directory(root, output_dir)

    assumptions = _assumptions_table(protocol)
    independent, independent_summary = _independent_results(protocol)
    null = _null_calibration(protocol)
    coverage = _coverage_calibration(protocol)
    paired, bounds = _paired_results(protocol)
    clustered, recommendation = _cluster_results(protocol, paired)
    ablation = _ablation_results(protocol)
    sensitivity = _sensitivity_results(protocol)
    precision, precision_targets = _precision_results(protocol)
    allocation = _allocation_results(protocol)

    tables = {
        "table1_protocol_assumptions": assumptions,
        "table2_independent_power": independent,
        "table3_null_calibration": null,
        "table4_wilson_coverage": coverage,
        "table5_paired_sensitivity": paired,
        "table6_cluster_sensitivity": clustered,
        "table7_ablation_power": ablation,
        "table8_event_rate_sensitivity": sensitivity,
        "table9_precision_intervals": precision,
        "table10_precision_targets": precision_targets,
        "table11_balanced_allocation": allocation,
    }
    for name, frame in tables.items():
        _save_table(frame, output / "tables" / name)

    _plot_all(
        output,
        protocol,
        independent,
        paired,
        clustered,
        sensitivity,
        precision,
        allocation,
    )

    report = _results_report(
        protocol,
        assumptions,
        independent,
        independent_summary,
        null,
        coverage,
        paired,
        bounds,
        clustered,
        recommendation,
        ablation,
        precision_targets,
    )
    (output / "POWER_ANALYSIS_RESULTS.md").write_text(report, encoding="utf-8")
    manuscript = _manuscript_ready_results(
        protocol, independent_summary, paired, recommendation, ablation
    )
    (output / "MANUSCRIPT_READY_RESULTS.md").write_text(manuscript, encoding="utf-8")

    summary = {
        "disclaimer": DISCLAIMER,
        "independent_raw_required_n": independent_summary["raw_required_n"],
        "independent_required_n_per_arm": independent_summary["required_n"],
        "independent_power_at_450": independent_summary["power_450"],
        "independent_power_at_150": independent_summary["power_150"],
        "feasible_paired_correlation": {"lower": bounds[0], "upper": bounds[1]},
        "cluster_aware_provisional_recommendation": recommendation,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    validation = _validation_checks(
        protocol, independent_summary, independent, null, coverage, paired, report
    )
    validation.to_csv(output / "validation_report.csv", index=False)
    if not bool(validation["passed"].all()):
        failures = validation.loc[~validation["passed"], ["check_id", "check", "evidence"]]
        raise RuntimeError("Validation failed:\n" + failures.to_string(index=False))
    manifest = _write_manifest(root, output, protocol, validation)

    return {
        "output_directory": str(output),
        "summary": summary,
        "validation": {
            "passed": int(validation["passed"].sum()),
            "total": len(validation),
        },
        "manifest": manifest,
    }
