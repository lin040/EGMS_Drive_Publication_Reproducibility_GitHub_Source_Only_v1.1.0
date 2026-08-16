from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from .utils import select_one, write_table


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def _signed(value: float, decimals: int = 4) -> str:
    return f"{value:+.{decimals}f}"


def build_main_table2(
    study1_csv: Path,
    studies_tables: Path,
    output_stem: Path,
) -> pd.DataFrame:
    """Build the manuscript's 16-row Table 2 from numeric source tables."""

    s1 = pd.read_csv(study1_csv, keep_default_na=False)
    records: list[dict[str, str]] = []
    for row in s1.itertuples(index=False):
        records.append(
            {
                "Study/module": row.study_module,
                "Comparison and endpoint": row.comparison_endpoint,
                "Effect*": row.effect_display.strip(),
                "95% CI": row.ci_display.strip(),
                "Evidence status": row.evidence_status,
            }
        )

    s2 = pd.read_csv(studies_tables / "table_s2_primary_contrasts.csv")
    s2_specs = [
        (
            "Full vs Equal weighting",
            "S2 reliability",
            "Full vs Equal weighting; adverse NLL ↓",
            4,
            "Surrogate reliability response supported",
            "Benefit not clearly established",
        ),
        (
            "Full vs No alignment",
            "S2 alignment",
            "Full vs No alignment; clean SAS ↑",
            4,
            "Surrogate alignment response supported",
            "Benefit not clearly established",
        ),
        (
            "Full vs No temporal consistency",
            "S2 temporal",
            "Full vs No temporal; one-step error ↓",
            4,
            "Surrogate temporal response supported",
            "Benefit not clearly established",
        ),
        (
            "Full vs No modality dropout-distillation",
            "S2 missing modality",
            "Full vs No dropout–distillation; F1 drop ↓",
            4,
            "Surrogate missing-modality response supported",
            "Benefit not clearly established",
        ),
    ]
    for comparison, module, endpoint, decimals, supported_text, unsupported_text in s2_specs:
        row = select_one(s2, comparison=comparison)
        effect = float(row["difference_full_minus_ablation"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        records.append(
            {
                "Study/module": module,
                "Comparison and endpoint": endpoint,
                "Effect*": _signed(effect, decimals),
                "95% CI": f"{_signed(low, decimals)} to {_signed(high, decimals)}",
                "Evidence status": supported_text if _as_bool(row["mechanism_support_rule_met"]) else unsupported_text,
            }
        )

    s3 = pd.read_csv(studies_tables / "table_s3_primary_contrasts.csv")
    s3_specs = [
        (
            "Full temporal graph vs No graph",
            "S3 temporal graph",
            "Full vs No graph; history minFDE₆ ↓",
            1.0,
            4,
            " m",
            "History-dependent graph response supported",
            "Benefit not clearly established",
        ),
        (
            "Full temporal graph vs Spatial-only graph",
            "S3 temporal relation",
            "Full vs Spatial-only; history minFDE₆ ↓",
            1.0,
            4,
            " m",
            "Temporal-relation response supported",
            "Benefit not clearly established",
        ),
        (
            "Full temporal graph vs No predicted-intent gating",
            "S3 intent gate",
            "Full vs No intent gate; Brier-minFDE₆ ↓",
            1.0,
            4,
            "",
            "Intent-gating joint-endpoint response supported",
            "Benefit not clearly established",
        ),
        (
            "Full top-1 vs independently trained Single mode K=1",
            "S3 top-1",
            "Full top-1 vs fitted K=1; MR₁ ↓",
            100.0,
            2,
            " pp",
            "Top-1 response supported",
            "Top-1 benefit not clearly established",
        ),
    ]
    for comparison, module, endpoint, scale, decimals, unit, supported_text, unsupported_text in s3_specs:
        row = select_one(s3, comparison=comparison)
        effect = scale * float(row["difference_full_minus_ablation"])
        low = scale * float(row["ci_low"])
        high = scale * float(row["ci_high"])
        records.append(
            {
                "Study/module": module,
                "Comparison and endpoint": endpoint,
                "Effect*": f"{_signed(effect, decimals)}{unit}",
                "95% CI": f"{_signed(low, decimals)} to {_signed(high, decimals)}{unit}",
                "Evidence status": supported_text if _as_bool(row["mechanism_support_rule_met"]) else unsupported_text,
            }
        )

    controls = pd.read_csv(studies_tables / "table_s3_negative_controls.csv")
    control_specs = [
        (
            "independent: Full vs No graph",
            "Independent: Full vs No graph; minFDE₆",
            "Compatible with ±0.25-m descriptive range; not TOST",
        ),
        (
            "spatial: Full vs Spatial-only graph",
            "Spatial: Full vs Spatial-only; minFDE₆",
            "Not wholly within descriptive range; not TOST",
        ),
    ]
    for control_name, endpoint, status in control_specs:
        row = select_one(controls, negative_control=control_name)
        effect = float(row["difference_full_minus_ablation"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        records.append(
            {
                "Study/module": "S3 neg. control",
                "Comparison and endpoint": endpoint,
                "Effect*": f"{_signed(effect, 4)} m",
                "95% CI": f"{_signed(low, 4)} to {_signed(high, 4)} m",
                "Evidence status": status,
            }
        )

    table = pd.DataFrame.from_records(records)
    if len(table) != 16:
        raise ValueError(f"Table 2 must contain 16 rows; found {len(table)}")
    write_table(table, output_stem)
    return table


def _load_power_tables(power_output: Path) -> dict[str, pd.DataFrame]:
    names = {
        "independent": "table2_independent_power.csv",
        "null": "table3_null_calibration.csv",
        "coverage": "table4_wilson_coverage.csv",
        "paired": "table5_paired_sensitivity.csv",
        "cluster": "table6_cluster_sensitivity.csv",
        "ablation": "table7_ablation_power.csv",
        "event": "table8_event_rate_sensitivity.csv",
        "precision": "table9_precision_intervals.csv",
        "precision_targets": "table10_precision_targets.csv",
        "allocation": "table11_balanced_allocation.csv",
    }
    return {key: pd.read_csv(power_output / "tables" / name) for key, name in names.items()}


def build_table_s2(protocol_path: Path, power_output: Path, output_stem: Path) -> pd.DataFrame:
    """Build compact Supplementary Table S2 from power-analysis outputs."""

    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    summary = json.loads((power_output / "summary.json").read_text(encoding="utf-8"))
    tables = _load_power_tables(power_output)
    independent = tables["independent"]
    rows = {int(row.episodes_per_arm): row for row in independent.itertuples(index=False)}
    if not {150, 450, 749}.issubset(rows):
        raise ValueError("Power table is missing n=150, 450, or 749")
    null_min = float(tables["null"]["estimated_type1_error"].min())
    null_max = float(tables["null"]["estimated_type1_error"].max())
    cover_min = float(tables["coverage"]["estimated_coverage"].min())
    cover_max = float(tables["coverage"]["estimated_coverage"].max())
    max_mcse = max(
        float(independent["mcse"].max()),
        float(tables["null"]["mcse"].max()),
        float(tables["coverage"]["mcse"].max()),
    )
    primary = protocol["primary_design"]
    simulation = protocol["simulation"]
    scenario = protocol["scenario_matrix"]
    data = [
        ("Study status", "Prospective design only", "No empirical CARLA, public-data, real-vehicle, or LLM result"),
        ("Endpoint and estimand", "Episode collision; candidate − control risk difference", "One prespecified confirmatory primary contrast"),
        ("Planning probabilities", f"Control {primary['control_rate_assumption']:.2f}; candidate {primary['candidate_rate_assumption']:.2f}", "Assumed values, not observed rates"),
        ("Error and power targets", f"Two-sided α = {primary['alpha']:.2f}; power = {primary['target_power']:.2f}", "Independent equal-allocation calculation"),
        ("Required independent n", f"{summary['independent_raw_required_n']:.3f}; rounded to {int(summary['independent_required_n_per_arm'])}/condition", f"Analytic power = {rows[749].analytic_power:.4f}"),
        ("Power at 150 / 450 / 749", f"{rows[150].analytic_power:.4f} / {rows[450].analytic_power:.4f} / {rows[749].analytic_power:.4f} analytic", f"MC = {rows[150].mc_power:.4f} / {rows[450].mc_power:.4f} / {rows[749].mc_power:.4f}; {int(rows[150].mc_repetitions):,} trials"),
        ("Calibration range", f"Type-I {null_min:.5f}–{null_max:.5f}; coverage {cover_min:.5f}–{cover_max:.5f}", f"Maximum MCSE = {max_mcse:.6f}"),
        ("Allocation block", f"{scenario['allocation_block']} episodes/condition", f"{scenario['scenario_families']} conflict families × {scenario['environments']} environments × {scenario['traffic_densities']} densities"),
        ("Master seed", str(simulation["master_seed"]), "Deterministic labeled random-number streams"),
    ]
    table = pd.DataFrame(data, columns=["Item", "Value", "Interpretation"])
    write_table(table, output_stem)
    return table


def build_table_s3(power_output: Path, output_stem: Path) -> pd.DataFrame:
    """Build compact Supplementary Table S3 from detailed power tables."""

    tables = _load_power_tables(power_output)
    paired = tables["paired"]
    rho0 = select_one(paired, correlation=0.0)
    pair_min = int(paired["minimum_pairs_exact"].min())
    pair_max = int(paired["minimum_pairs_exact"].max())
    cluster = select_one(tables["cluster"], cluster_size_cv=0.0, icc=0.05)
    small = select_one(
        tables["ablation"],
        scenario="two_events_in_150_gap",
        alpha=0.05,
    )
    multiplicity = select_one(
        tables["ablation"],
        scenario="two_events_in_150_gap",
        alpha=0.0125,
    )
    event80 = tables["event"].loc[tables["event"]["target_power"].eq(0.8)]
    precision150 = select_one(tables["precision"], episodes_per_arm=150, arm="control")
    targets = tables["precision_targets"].sort_values("target_risk_difference_half_width", ascending=False)
    target_text = "/".join(f"{100 * row.target_risk_difference_half_width:.1f}" for row in targets.itertuples(index=False))
    required_text = "/".join(f"{int(row.required_episodes_per_arm):,}" for row in targets.itertuples(index=False))
    allocation = select_one(tables["allocation"], seeds_per_scenario_environment_density_cell=28)
    data = [
        (
            "Exact paired design",
            "ρ grid −0.04 to 0.60; 6% vs 3% marginals",
            f"{pair_min}–{pair_max} pairs; at ρ=0, {int(rho0['minimum_pairs_exact'])}; power at 450={float(rho0['exact_power_at_450_pairs']):.4f}",
            "Joint distributions are assumed",
        ),
        (
            "Provisional cluster-aware plan",
            "ρ=−0.04; mean cluster=9; ICC=0.05; CV=0; 5% invalid; block=45",
            f"{int(cluster['episodes_per_condition']):,}/condition; {int(cluster['total_episodes_primary_two_condition']):,} total; {int(cluster['seeds_per_45_cell_matrix'])} seeds/cell; approx. power={float(cluster['approx_exact_power_at_effective_pairs']):.4f}",
            "Pilot update required",
        ),
        (
            "Illustrative small effect",
            "6.000% vs 4.667%; α=0.05",
            f"{int(small['required_episodes_per_arm']):,}/condition; power at 150={float(small['power_at_150_per_arm']):.4f}",
            "Assumed contrast, not an ablation result",
        ),
        (
            "Conservative multiplicity bound",
            "Same small effect; α=0.0125",
            f"{int(multiplicity['required_episodes_per_arm']):,}/condition",
            "Bonferroni planning bound, not Holm-derived n",
        ),
        (
            "Event/effect grid",
            "Control 3%–15%; reductions 1–4 pp",
            f"{int(event80['required_episodes_per_arm'].min()):,}–{int(event80['required_episodes_per_arm'].max()):,}/condition for 80% power",
            "All rates and reductions are assumed",
        ),
        (
            "Risk-difference precision",
            "6% vs 3% planning probabilities",
            f"Half-width at n=150: {100 * float(precision150['risk_difference_half_width']):.3f} pp; {target_text} pp targets require {required_text}",
            "Illustrative precision only",
        ),
        (
            "Balanced-allocation diagnostic",
            f"{int(allocation['seeds_per_scenario_environment_density_cell'])} seeds/cell = {int(allocation['episodes_per_condition']):,}/condition",
            f"Independent power={float(allocation['independent_power']):.4f}; ICC-only approximation={float(allocation['design_effect_approx_power']):.4f}",
            "Not the primary paired model",
        ),
    ]
    table = pd.DataFrame(data, columns=["Planning component", "Assumption/model", "Result", "Boundary"])
    write_table(table, output_stem)
    return table


def build_table_s4(power_output: Path, output_stem: Path) -> pd.DataFrame:
    """Build compact Supplementary Table S4 from the 33 validation checks."""

    validation = pd.read_csv(power_output / "validation_report.csv")
    table = pd.DataFrame(
        {
            "ID": validation["check_id"].astype(int),
            "Check": validation["check"],
            "Status": validation["passed"].map(lambda value: "Pass" if _as_bool(value) else "Fail"),
            "Evidence": validation["evidence"],
        }
    )
    if len(table) != 33 or not table["Status"].eq("Pass").all():
        raise ValueError("Table S4 must contain 33 passing checks")
    write_table(table, output_stem)
    return table

