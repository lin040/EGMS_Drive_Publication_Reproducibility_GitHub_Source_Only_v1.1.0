"""Publication outputs for post-hoc exploratory Study 1-R2."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from egms_publication.utils import write_table

from .plotting import build_figure2_study1r2, build_figure2r_study1r_vs_r2
from .statistics import METRIC_BY_ENDPOINT, summarize_study1r


def _read_frame(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.read_csv(Path(value))


def _signed(value: float, decimals: int) -> str:
    return f"{value:+.{decimals}f}"


def build_figure_inputs(
    metric_summary: pd.DataFrame,
    paired_contrasts: pd.DataFrame,
) -> pd.DataFrame:
    """Return one fully auditable source row for every Figure 2 endpoint."""

    rows: list[dict[str, object]] = []
    for contrast in paired_contrasts.itertuples(index=False):
        endpoint = str(contrast.endpoint)
        marginal = metric_summary.loc[metric_summary["endpoint"].eq(endpoint)].set_index("method")
        if set(marginal.index) != {"Baseline B", "Structured-R2"}:
            raise ValueError(f"Expected two marginal rows for {endpoint}")
        baseline = marginal.loc["Baseline B"]
        structured = marginal.loc["Structured-R2"]
        rows.append(
            {
                **contrast._asdict(),
                "baseline_ci_low": float(baseline["ci_low"]),
                "baseline_ci_high": float(baseline["ci_high"]),
                "structured_ci_low": float(structured["ci_low"]),
                "structured_ci_high": float(structured["ci_high"]),
            }
        )
    return pd.DataFrame(rows)


def build_table2_source(paired_contrasts: pd.DataFrame) -> pd.DataFrame:
    """Build manuscript-compatible Study 1-R rows from recomputed contrasts."""

    module = {
        "action": "S1-R2 action (exploratory)",
        "event": "S1-R2 closed loop (exploratory)",
        "ttc": "S1-R2 conditional TTC (exploratory)",
        "jerk": "S1-R2 jerk proxy (exploratory)",
    }
    # Export six prespecified Study 1-R rows for the revised 20-row main Table 2;
    # the remaining four endpoints stay in Figure 2-R and the full contrast table.
    selected_endpoints = (
        "macro_f1",
        "collision",
        "near_miss",
        "route_completion",
        "ttc_p5",
        "jerk_p95",
    )
    selected = paired_contrasts.set_index("endpoint").loc[list(selected_endpoints)].reset_index()
    rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        spec = METRIC_BY_ENDPOINT[str(row.endpoint)]
        scale = float(spec.display_scale)
        effect = scale * float(row.raw_difference_structured_minus_baseline)
        low = scale * float(row.raw_ci_low)
        high = scale * float(row.raw_ci_high)
        decimals = 2 if scale == 100.0 else 4 if spec.endpoint in {"macro_f1", "nll", "brier", "ece"} else 3
        suffix = " pp" if scale == 100.0 else " s" if spec.endpoint == "ttc_p5" else " m/s^3" if spec.endpoint == "jerk_p95" else ""
        arrow = "↑" if spec.orientation == "higher_is_better" else "↓"
        if bool(row.support_rule_met):
            status = "Structured-R2 benefit supported under the frozen R2 evaluation rule"
        elif float(row.benefit_ci_high) < 0.0:
            status = "Observed direction favors Baseline B; retained without filtering"
        else:
            status = "Difference not clearly established"
        rows.append(
            {
                "study_module": module[spec.panel],
                "comparison_endpoint": f"Structured-R2 vs Baseline B; {spec.label} {arrow}",
                "effect": effect,
                "ci_low": low,
                "ci_high": high,
                "effect_display": f"{_signed(effect, decimals)}{suffix}",
                "ci_display": f"{_signed(low, decimals)} to {_signed(high, decimals)}{suffix}",
                "evidence_status": status,
                "favorable_replicates": int(row.favorable_replicates),
                "tied_replicates": int(row.tied_replicates),
                "adverse_replicates": int(row.adverse_replicates),
                "holm_adjusted_p": float(row.holm_adjusted_p),
            }
        )
    return pd.DataFrame(rows)


def write_study1r2_reports(
    offline_predictions: pd.DataFrame | str | Path,
    episode_metrics: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    repetitions: int = 10_000,
    confidence: float = 0.95,
    bootstrap_seed: int = 171031,
) -> dict[str, object]:
    """Recompute statistics, write publication tables, and render Figure 2."""

    output = Path(output_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    data_dir = output / "data"
    for directory in (tables_dir, figures_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    results = summarize_study1r(
        _read_frame(offline_predictions),
        _read_frame(episode_metrics),
        repetitions=repetitions,
        confidence=confidence,
        bootstrap_seed=bootstrap_seed,
    )
    paths: dict[str, object] = {}
    filenames = {
        "metric_summary": "study1r2_metric_summary.csv",
        "paired_contrasts": "study1r2_paired_contrasts.csv",
        "replicate_effects": "study1r2_replicate_effects.csv",
        "bootstrap_draws": "study1r2_bootstrap_draws.csv",
    }
    for name, filename in filenames.items():
        path = data_dir / filename
        results[name].to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
        paths[name] = path

    figure_inputs = build_figure_inputs(
        results["metric_summary"],
        results["paired_contrasts"],
    )
    figure_input_path = data_dir / "Figure_2B_Study1R2_Exploratory_inputs.csv"
    figure_inputs.to_csv(
        figure_input_path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    paths["figure_inputs"] = figure_input_path

    table2 = build_table2_source(results["paired_contrasts"])
    paths["table2"] = write_table(table2, tables_dir / "Table_2B_Study1R2_Exploratory")
    paths["figure2"] = build_figure2_study1r2(
        results["metric_summary"],
        results["paired_contrasts"],
        figures_dir / "Figure_2B_Study1R2_Exploratory",
    )
    paths["results"] = results
    return paths


# A concise alias for orchestration code.
write_study1r_reports = write_study1r2_reports
write_study1r_outputs = write_study1r2_reports


FIGURE2R_ENDPOINTS = (
    "macro_f1", "nll", "brier", "ece", "collision", "near_miss",
    "critical_event", "route_completion", "ttc_p5", "jerk_p95",
)


def build_figure2r_inputs(
    r1_paired_contrasts: pd.DataFrame | str | Path,
    r2_paired_contrasts: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    """Combine two within-study contrast tables without pooling their tapes."""

    columns = [
        "endpoint", "metric", "panel", "orientation", "unit", "display_scale",
        "benefit_difference", "benefit_ci_low", "benefit_ci_high",
        "favorable_replicates", "tied_replicates", "adverse_replicates",
        "holm_adjusted_p",
    ]
    variants = (
        ("Frozen Study 1-R", "Structured", "frozen_formal", _read_frame(r1_paired_contrasts)),
        (
            "Exploratory Study 1-R2", "Structured-R2",
            "post_hoc_exploratory_final_evaluation", _read_frame(r2_paired_contrasts),
        ),
    )
    blocks: list[pd.DataFrame] = []
    expected = set(FIGURE2R_ENDPOINTS)
    for variant, candidate, status, frame in variants:
        missing = set(columns).difference(frame.columns)
        if missing or set(frame["endpoint"]) != expected or frame["endpoint"].duplicated().any():
            raise ValueError(f"{variant} has an invalid contrast table: {sorted(missing)}")
        block = frame.loc[:, columns].copy()
        block.insert(0, "analysis_status", status)
        block.insert(0, "candidate", candidate)
        block.insert(0, "study_variant", variant)
        total = (
            block["favorable_replicates"] + block["tied_replicates"]
            + block["adverse_replicates"]
        )
        if not total.eq(10).all():
            raise ValueError(f"{variant} directionality counts do not sum to ten")
        block["display_estimate"] = block["display_scale"] * block["benefit_difference"]
        block["display_ci_low"] = block["display_scale"] * block["benefit_ci_low"]
        block["display_ci_high"] = block["display_scale"] * block["benefit_ci_high"]
        blocks.append(block)
    combined = pd.concat(blocks, ignore_index=True)
    order = {endpoint: index for index, endpoint in enumerate(FIGURE2R_ENDPOINTS)}
    variant_order = {"Frozen Study 1-R": 0, "Exploratory Study 1-R2": 1}
    return combined.sort_values(
        ["endpoint", "study_variant"],
        key=lambda values: values.map(order if values.name == "endpoint" else variant_order),
        kind="stable",
    ).reset_index(drop=True)


def write_figure2r_comparison(
    r1_paired_contrasts: pd.DataFrame | str | Path,
    r2_paired_contrasts: pd.DataFrame | str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    output = Path(output_dir)
    data_dir = output / "data"
    figures_dir = output / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    inputs = build_figure2r_inputs(r1_paired_contrasts, r2_paired_contrasts)
    csv_path = data_dir / "Figure_2R_Study1R_vs_Exploratory_R2_inputs.csv"
    inputs.to_csv(csv_path, index=False, float_format="%.17g", lineterminator="\n")
    figures = build_figure2r_study1r_vs_r2(
        csv_path, figures_dir / "Figure_2R_Study1R_vs_Exploratory_R2"
    )
    return {"inputs": csv_path, "figures": figures}
