#!/usr/bin/env python3
"""Build manuscript Figure 2 from validated, reader-facing Study 1 CSVs.

The canonical plotting inputs contain only the public method names
``Baseline B`` and ``Structured fusion``.  Archived machine identifiers are
kept separately under ``data/study1_frozen/provenance`` for source-hash and
checkpoint auditing.  No model, raw output, endpoint, interval, or p value is
changed by this display layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


INTERNAL_BASELINE = "Baseline B"
INTERNAL_STRUCTURED = "Structured fusion"
DISPLAY_BASELINE = "Baseline B"
DISPLAY_STRUCTURED = "Structured fusion"

ENDPOINTS = (
    "macro_f1",
    "nll",
    "brier",
    "ece",
    "collision",
    "near_miss",
    "critical_event",
    "route_completion",
    "ttc_p5",
    "jerk_p95",
)

BASELINE_COLOR = "#1f5a91"
STRUCTURED_COLOR = "#cf4b5f"
ZERO_COLOR = "#228f87"
GRID_COLOR = "#d9dde2"
TEXT_COLOR = "#17191c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_one(frame: pd.DataFrame, **filters: object) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in filters.items():
        mask &= frame[column].astype(str).eq(str(value))
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}; found {len(selected)}")
    return selected.iloc[0]


def _validate(summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    if set(contrasts["endpoint"]) != set(ENDPOINTS):
        raise ValueError("Paired-contrast table must contain the ten frozen endpoints")
    if contrasts["endpoint"].duplicated().any():
        raise ValueError("Paired-contrast endpoints must be unique")
    expected_summary = {(endpoint, method) for endpoint in ENDPOINTS for method in (INTERNAL_BASELINE, INTERNAL_STRUCTURED)}
    observed_summary = set(zip(summary["endpoint"].astype(str), summary["method"].astype(str)))
    if observed_summary != expected_summary:
        raise ValueError("Metric-summary method/endpoint grid is incomplete")
    if not contrasts["total_training_replicates"].eq(10).all():
        raise ValueError("Figure 2 requires exactly ten paired training replicates")
    if not (
        contrasts["benefit_ci_low"].le(contrasts["benefit_difference"])
        & contrasts["benefit_difference"].le(contrasts["benefit_ci_high"])
    ).all():
        raise ValueError("At least one paired CI does not contain its point estimate")


def _p_text(value: float) -> str:
    return f"{value:.4f}" if value >= 0.0001 else "<0.0001"


def _symmetric_limits(values: np.ndarray, floor: float) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    bound = max(float(np.max(np.abs(finite))) if len(finite) else floor, floor)
    return -1.24 * bound, 1.24 * bound


def _range_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    low = float(finite.min())
    high = float(finite.max())
    span = max(high - low, 0.08 * max(abs(low), abs(high), 1.0))
    return max(0.0, low - 0.20 * span), high + 0.20 * span


def _format_axis(ax: plt.Axes) -> None:
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=7.2, width=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_COLOR)
        ax.spines[side].set_linewidth(0.8)


def _forest_panel(
    ax: plt.Axes,
    contrasts: pd.DataFrame,
    endpoints: tuple[str, ...],
    *,
    title: str,
    xlabel: str,
    scale: float,
    decimals: int,
    floor: float,
) -> None:
    rows = [_select_one(contrasts, endpoint=endpoint) for endpoint in endpoints]
    estimates = scale * np.asarray([float(row["benefit_difference"]) for row in rows])
    lows = scale * np.asarray([float(row["benefit_ci_low"]) for row in rows])
    highs = scale * np.asarray([float(row["benefit_ci_high"]) for row in rows])
    y = np.arange(len(rows) - 1, -1, -1, dtype=float)

    # Baseline B is the reference contrast and therefore equals zero by
    # definition in panels (a) and (b).  Plotting the reference markers makes
    # the reader-facing Baseline-versus-Structured legend mathematically true.
    ax.scatter(
        np.zeros_like(y),
        y,
        marker="o",
        s=24,
        facecolor="white",
        edgecolor=BASELINE_COLOR,
        linewidth=1.15,
        zorder=4,
    )

    for row, estimate, low, high, ypos in zip(rows, estimates, lows, highs, y):
        ax.errorbar(
            estimate,
            ypos,
            xerr=np.array([[estimate - low], [high - estimate]]),
            fmt="o",
            markersize=5.1,
            markerfacecolor="white",
            markeredgecolor=STRUCTURED_COLOR,
            markeredgewidth=1.25,
            ecolor=STRUCTURED_COLOR,
            elinewidth=1.35,
            capsize=3.0,
            capthick=1.25,
            zorder=3,
        )
        ax.annotate(
            f"Delta={estimate:+.{decimals}f}",
            (0.985, ypos),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="bottom",
            fontsize=6.0,
            color=TEXT_COLOR,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.25},
        )

    ax.axvline(0.0, color=ZERO_COLOR, linestyle="--", linewidth=1.1, zorder=1)
    ax.set_xlim(*_symmetric_limits(np.r_[estimates, lows, highs], floor))
    ax.set_ylim(-0.38, len(rows) - 0.62)
    ax.set_yticks(y, [str(row["metric"]) for row in rows])
    ax.set_xlabel(xlabel, fontsize=7.1, labelpad=4)
    ax.set_title(title, loc="left", fontsize=8.6, fontweight="bold", pad=6)
    _format_axis(ax)


def _method_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    endpoint: str,
    *,
    title: str,
    xlabel: str,
    decimals: int,
) -> None:
    rows = (
        _select_one(summary, endpoint=endpoint, method=INTERNAL_BASELINE),
        _select_one(summary, endpoint=endpoint, method=INTERNAL_STRUCTURED),
    )
    colors = (BASELINE_COLOR, STRUCTURED_COLOR)
    labels = (DISPLAY_BASELINE, DISPLAY_STRUCTURED)
    y = (0.32, 0.68)
    bounds: list[float] = []

    for row, color, label, ypos in zip(rows, colors, labels, y):
        estimate = float(row["estimate"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        bounds.extend((estimate, low, high))
        ax.errorbar(
            estimate,
            ypos,
            xerr=np.array([[estimate - low], [high - estimate]]),
            fmt="o",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.35,
            ecolor=color,
            elinewidth=1.4,
            capsize=3.1,
            capthick=1.3,
            zorder=3,
        )
        ax.annotate(
            f"{estimate:.{decimals}f}",
            (estimate, ypos),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.4,
            color=TEXT_COLOR,
        )

    contrast = _select_one(contrasts, endpoint=endpoint)
    delta = float(contrast["raw_difference_structured_minus_baseline"])
    low = float(contrast["raw_ci_low"])
    high = float(contrast["raw_ci_high"])
    note = (
        f"Paired difference={delta:+.{decimals}f} "
        f"[{low:+.{decimals}f}, {high:+.{decimals}f}]; "
        f"Holm p={_p_text(float(contrast['holm_adjusted_p']))}"
    )
    ax.text(0.50, 0.055, note, transform=ax.transAxes, ha="center", va="bottom", fontsize=5.9, color=TEXT_COLOR)
    ax.set_xlim(*_range_limits(np.asarray(bounds)))
    ax.set_ylim(0.10, 0.90)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel, fontsize=7.1, labelpad=4)
    ax.set_title(title, loc="left", fontsize=8.6, fontweight="bold", pad=6)
    _format_axis(ax)


def _publication_inputs(summary: pd.DataFrame, contrasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_inputs = summary.copy()
    method_inputs = method_inputs.drop(columns=["internal_method"], errors="ignore")
    contrast_inputs = contrasts.copy()
    if "display_contrast" not in contrast_inputs.columns:
        contrast_inputs.insert(0, "display_contrast", f"{DISPLAY_STRUCTURED} - {DISPLAY_BASELINE}")
    return method_inputs, contrast_inputs


def build(summary_path: Path, contrasts_path: Path, output_dir: Path) -> dict[str, Path]:
    summary = pd.read_csv(summary_path)
    contrasts = pd.read_csv(contrasts_path)
    _validate(summary, contrasts)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.62, 5.044), dpi=500)
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.135, top=0.865, wspace=0.43, hspace=0.60)

    _forest_panel(
        axes[0, 0],
        contrasts,
        ("macro_f1", "nll", "brier", "ece"),
        title="(a) Offline action and probability metrics",
        xlabel="Benefit-oriented paired difference\n(Baseline B = 0; positive favors Structured fusion)",
        scale=1.0,
        decimals=4,
        floor=0.01,
    )
    _forest_panel(
        axes[0, 1],
        contrasts,
        ("collision", "near_miss", "critical_event", "route_completion"),
        title="(b) Closed-loop outcomes",
        xlabel="Benefit-oriented paired difference (percentage points)\n(Baseline B = 0; positive favors Structured fusion)",
        scale=100.0,
        decimals=2,
        floor=1.0,
    )
    _method_panel(
        axes[1, 0],
        summary,
        contrasts,
        "ttc_p5",
        title="(c) Jointly collision-free TTC-P5",
        xlabel="Mean episode TTC-P5 (s)",
        decimals=3,
    )
    _method_panel(
        axes[1, 1],
        summary,
        contrasts,
        "jerk_p95",
        title="(d) Episode kinematic smoothness",
        xlabel="Mean episode jerk-P95 (m/s³)",
        decimals=3,
    )

    handles = (
        Line2D([0], [0], marker="o", linestyle="-", color=BASELINE_COLOR, markerfacecolor="white", markeredgecolor=BASELINE_COLOR, markersize=5.2, label=DISPLAY_BASELINE),
        Line2D([0], [0], marker="o", linestyle="-", color=STRUCTURED_COLOR, markerfacecolor="white", markeredgecolor=STRUCTURED_COLOR, markersize=5.2, label=DISPLAY_STRUCTURED),
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=2, frameon=False, fontsize=7.5)
    fig.text(
        0.5,
        0.025,
        "Ten paired training replicates; bars are 95% crossed-bootstrap confidence intervals. The revised Structured fusion evaluation is exploratory.",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color=TEXT_COLOR,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "Figure_2_Study1_Baseline_B_vs_Structured_fusion"
    outputs = {
        "png": stem.with_suffix(".png"),
        "pdf": stem.with_suffix(".pdf"),
        "svg": stem.with_suffix(".svg"),
    }
    for path in outputs.values():
        fig.savefig(path, dpi=500, facecolor="white")
    plt.close(fig)

    method_inputs, contrast_inputs = _publication_inputs(summary, contrasts)
    method_csv = output_dir / "Figure_2_method_estimates.csv"
    contrast_csv = output_dir / "Figure_2_paired_contrasts.csv"
    method_inputs.to_csv(method_csv, index=False, float_format="%.17g", lineterminator="\n")
    contrast_inputs.to_csv(contrast_csv, index=False, float_format="%.17g", lineterminator="\n")

    manifest_path = output_dir / "Figure_2_publication_manifest.json"
    manifest = {
        "schema": "egms-drive-study1-publication-figure-1.0",
        "display_labels": [DISPLAY_BASELINE, DISPLAY_STRUCTURED],
        "display_only_alias": False,
        "validated_inputs": {
            str(summary_path): _sha256(summary_path),
            str(contrasts_path): _sha256(contrasts_path),
        },
        "outputs": {},
        "note": "Reader-facing names are canonical here; archived machine identifiers remain only in provenance files.",
    }
    for path in (*outputs.values(), method_csv, contrast_csv):
        manifest["outputs"][path.name] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs.update({"method_inputs": method_csv, "contrast_inputs": contrast_csv, "manifest": manifest_path})
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--contrasts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = build(args.summary, args.contrasts, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
