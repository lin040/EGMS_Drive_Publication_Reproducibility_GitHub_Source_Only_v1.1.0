from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import GRAY, NAVY, RED, TEAL, TEXT, apply_style, format_axes, panel_title, save_figure
from .utils import select_one


def _as_bool(value: object) -> bool:
    """Parse CSV booleans without treating the string ``False`` as true."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value!r}")


def _horizontal_ci(
    ax: plt.Axes,
    *,
    estimate: float,
    low: float,
    high: float,
    y: float,
    color: str,
    marker: str = "o",
    filled: bool = False,
    size: float = 6.3,
) -> None:
    ax.errorbar(
        estimate,
        y,
        xerr=np.array([[estimate - low], [high - estimate]]),
        fmt=marker,
        markersize=size,
        markerfacecolor=color if filled else "white",
        markeredgecolor=color,
        markeredgewidth=1.35,
        ecolor=color,
        elinewidth=1.55,
        capsize=3.8,
        capthick=1.55,
        zorder=3,
    )


def _endpoint_label(
    ax: plt.Axes,
    text: str,
    *,
    low: float,
    high: float,
    y: float,
    side: str,
    top_row: bool = False,
) -> None:
    anchor = high if side == "right" else low
    dx = 5 if side == "right" else -5
    dy = -5 if top_row else 5
    ax.annotate(
        text,
        (anchor, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if side == "right" else "right",
        va="top" if top_row else "bottom",
        fontsize=8.2,
        color=TEXT,
    )


def _benefit_oriented(row: pd.Series) -> tuple[float, float, float]:
    estimate = float(row["estimate"])
    low = float(row["ci_low"])
    high = float(row["ci_high"])
    if row["orientation"] == "lower_is_better":
        return -estimate, -high, -low
    return estimate, low, high


def build_figure2(study1_csv: Path, stem: Path) -> dict[str, Path]:
    """Generate manuscript Figure 2 from the frozen Study-1 audit snapshot.

    This is a graphical reproduction of exported/audited summary estimates.
    Study-1 raw data, model training code, and some original CI procedures are
    unavailable; this function must not be described as a raw recomputation.
    """

    frame = pd.read_csv(study1_csv)
    apply_style()
    width_px, height_px = 3810, 2522
    fig, axes = plt.subplots(2, 2, figsize=(width_px / 600, height_px / 600), dpi=600)
    fig.subplots_adjust(left=0.105, right=0.982, bottom=0.105, top=0.950, wspace=0.43, hspace=0.50)

    # (a) Offline action and probability scores, displayed in a benefit direction.
    ax = axes[0, 0]
    metrics = ["Macro-F1", "NLL", "Brier", "ECE"]
    ys = np.arange(3, -1, -1, dtype=float)
    for row_index, (metric, y) in enumerate(zip(metrics, ys)):
        row = select_one(frame, panel="action", metric=metric)
        estimate, low, high = _benefit_oriented(row)
        supported = low > 0
        color = NAVY if supported else RED
        _horizontal_ci(
            ax,
            estimate=estimate,
            low=low,
            high=high,
            y=y,
            color=color,
            filled=supported,
        )
        ax.annotate(
            f"{estimate:+.4f}",
            (high, y),
            xytext=(5, -5 if row_index == 0 else 5),
            textcoords="offset points",
            ha="left",
            va="top" if row_index == 0 else "bottom",
            fontsize=8.2,
        )
    ax.axvline(0, color=TEAL, ls="--", lw=1.4)
    ax.set_xlim(-0.0077, 0.0425)
    ax.set_xticks([0.00, 0.01, 0.02, 0.03, 0.04])
    ax.set_yticks(ys, metrics)
    ax.set_ylim(-0.15, 3.15)
    ax.set_xlabel("Benefit-oriented raw difference")
    panel_title(ax, "(a) Offline action and probability scores")
    format_axes(ax)

    # (b) Event and completion proxies, also transformed to a benefit direction.
    ax = axes[0, 1]
    metrics = ["Collision", "Near miss", "Critical event", "Route completion"]
    labels = ["Collision", "Near miss", "Critical event", "Route completion"]
    ys = np.arange(3, -1, -1, dtype=float)
    label_sides = ["right", "left", "left", "right"]
    for row_index, (metric, y, side) in enumerate(zip(metrics, ys, label_sides)):
        row = select_one(frame, panel="event", metric=metric)
        estimate, low, high = _benefit_oriented(row)
        _horizontal_ci(
            ax,
            estimate=estimate,
            low=low,
            high=high,
            y=y,
            color=RED,
            filled=False,
        )
        anchor = high if side == "right" else low
        ax.annotate(
            f"{estimate:+.2f} pp",
            (anchor, y),
            xytext=(5 if side == "right" else -5, -5 if row_index == 0 else 5),
            textcoords="offset points",
            ha="left" if side == "right" else "right",
            va="top" if row_index == 0 else "bottom",
            fontsize=8.2,
        )
    ax.axvline(0, color=TEAL, ls="--", lw=1.4)
    ax.set_xlim(-7.6, 20.6)
    ax.set_yticks(ys, labels)
    ax.set_ylim(-0.15, 3.15)
    ax.set_xlabel("Benefit-oriented paired difference (pp)")
    panel_title(ax, "(b) Event and completion proxies")
    format_axes(ax)

    # (c) Conditional TTC descriptor. Marginal intervals are audit-snapshot values.
    ax = axes[1, 0]
    conditions = [
        ("Baseline B", 0.34, NAVY, "right"),
        ("Structured fusion", 0.66, RED, "left"),
    ]
    for condition, y, color, side in conditions:
        row = select_one(frame, panel="conditional_ttc", metric="TTC-P5", condition=condition)
        estimate, low, high = (float(row[key]) for key in ("estimate", "ci_low", "ci_high"))
        _horizontal_ci(ax, estimate=estimate, low=low, high=high, y=y, color=color)
        _endpoint_label(
            ax,
            f"{estimate:.3f} s",
            low=low,
            high=high,
            y=y,
            side=side,
        )
        ax.plot([], [], "o", color=color, markerfacecolor="white", markeredgewidth=1.2,
                markersize=5.3, label=condition)
    ax.set_xlim(0.58, 1.39)
    ax.set_ylim(0.18, 0.82)
    ax.set_yticks([])
    ax.set_xlabel("Collision-free TTC-P5 (s)")
    panel_title(ax, "(c) Conditional TTC descriptor")
    format_axes(ax)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.50, 1.00), handletextpad=0.4, columnspacing=1.0)

    # (d) Kinematic smoothness proxy. Marginal intervals are audit-snapshot values.
    ax = axes[1, 1]
    conditions = [
        ("Baseline B", 0.34, NAVY, "left"),
        ("Structured fusion", 0.66, RED, "right"),
    ]
    for condition, y, color, side in conditions:
        row = select_one(frame, panel="jerk", metric="Jerk-P95", condition=condition)
        estimate, low, high = (float(row[key]) for key in ("estimate", "ci_low", "ci_high"))
        _horizontal_ci(ax, estimate=estimate, low=low, high=high, y=y, color=color)
        _endpoint_label(
            ax,
            f"{estimate:.2f}",
            low=low,
            high=high,
            y=y,
            side=side,
        )
        ax.plot([], [], "o", color=color, markerfacecolor="white", markeredgewidth=1.2,
                markersize=5.3, label=condition)
    ax.set_xlim(20.4, 35.7)
    ax.set_xticks(np.arange(22, 35, 2))
    ax.set_ylim(0.18, 0.82)
    ax.set_yticks([])
    ax.set_xlabel("Mean episode jerk-P95 (m/s³)")
    panel_title(ax, "(d) Kinematic smoothness proxy")
    format_axes(ax)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.50, 1.00), handletextpad=0.4, columnspacing=1.0)

    return save_figure(fig, stem, width_px=width_px, height_px=height_px)


def build_figure3(contrast_csv: Path, stem: Path) -> dict[str, Path]:
    """Generate manuscript Figure 3 directly from Study-2 contrast CSV rows."""

    contrasts = pd.read_csv(contrast_csv)
    apply_style()
    width_px, height_px = 3810, 2472
    fig, axes = plt.subplots(2, 2, figsize=(width_px / 600, height_px / 600), dpi=600)
    fig.subplots_adjust(left=0.085, right=0.982, bottom=0.105, top=0.950, wspace=0.42, hspace=0.50)
    specifications = [
        {
            "comparison": "Full vs Equal weighting",
            "title": "(a) Reliability weighting",
            "xlim": (-0.0125, 0.0020),
            "xlabel": "Full − equal weighting (adverse NLL)",
            "better": "Full better ←",
        },
        {
            "comparison": "Full vs No alignment",
            "title": "(b) Cross-modal alignment",
            "xlim": (-0.45, 3.08),
            "xlabel": "Full − no alignment (clean SAS)",
            "better": "→ Full better",
        },
        {
            "comparison": "Full vs No temporal consistency",
            "title": "(c) Temporal consistency",
            "xlim": (-0.070, 0.0108),
            "xlabel": "Full − no temporal (one-step error)",
            "better": "Full better ←",
        },
        {
            "comparison": "Full vs No modality dropout-distillation",
            "title": "(d) Modality dropout–distillation",
            "xlim": (-0.00325, 0.00072),
            "xlabel": "Full − no dropout–distillation (F1 drop)",
            "better": "Full better ←",
        },
    ]

    for panel_index, (ax, specification) in enumerate(zip(axes.flat, specifications)):
        row = select_one(contrasts, comparison=specification["comparison"])
        estimate = float(row["difference_full_minus_ablation"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        supported = _as_bool(row["mechanism_support_rule_met"])
        color = NAVY if supported else RED
        _horizontal_ci(
            ax,
            estimate=estimate,
            low=low,
            high=high,
            y=0.14,
            color=color,
            filled=supported,
        )
        ax.axvline(0, color=TEAL, ls="--", lw=1.4)
        ax.set_xlim(*specification["xlim"])
        ax.set_ylim(-0.34, 0.34)
        ax.set_yticks([])
        ax.set_xlabel(specification["xlabel"])
        if panel_index == 0:
            ax.set_xticks([-0.012, -0.009, -0.006, -0.003, 0.000])
        panel_title(ax, specification["title"])
        format_axes(ax)
        ax.text(0.02, 0.88, specification["better"], transform=ax.transAxes,
                ha="left", va="top", fontsize=8.0, color=NAVY)
        status = "Effect supported" if supported else "Benefit not clearly\nestablished"
        ax.text(0.98, 0.88, status, transform=ax.transAxes, ha="right", va="top",
                fontsize=8.0, color=color, fontweight="bold")
        annotation = (
            f"Δ={estimate:+.4f} [{low:+.4f}, {high:+.4f}]\n"
            f"favorable replicates {int(row['favorable_training_replicates'])}/"
            f"{int(row['total_training_replicates'])}; Holm p={float(row['holm_adjusted_p']):.4f}"
        )
        ax.text(0.50, 0.12, annotation, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.5, color=TEXT)

    return save_figure(fig, stem, width_px=width_px, height_px=height_px)


def build_figure4(tables_dir: Path, stem: Path) -> dict[str, Path]:
    """Generate manuscript Figure 4 from frozen Study-3 derived tables."""

    regime_rows = pd.read_csv(tables_dir / "table_s3_by_regime.csv")
    main_rows = pd.read_csv(tables_dir / "table_s3_main.csv")
    contrast_rows = pd.read_csv(tables_dir / "table_s3_primary_contrasts.csv")
    control_rows = pd.read_csv(tables_dir / "table_s3_negative_controls.csv")
    latency_rows = pd.read_csv(tables_dir / "table_s3_latency.csv")

    apply_style()
    width_px, height_px = 3810, 2485
    fig, axes = plt.subplots(2, 2, figsize=(width_px / 600, height_px / 600), dpi=600)
    fig.subplots_adjust(
        left=0.160,
        right=0.982,
        bottom=0.115,
        top=0.895,
        wspace=0.45,
        hspace=0.66,
    )

    # (a) Graph mechanism by interaction regime.
    ax = axes[0, 0]
    regimes = ["independent", "spatial", "history_dependent"]
    regime_labels = ["Independent", "Spatial", "History-dependent"]
    method_specs = [
        ("Full temporal graph", -0.16, "o", NAVY, True, "Full temporal graph"),
        ("No graph", 0.00, "s", GRAY, False, "No graph"),
        ("Spatial-only graph", 0.16, "^", RED, False, "Spatial-only graph"),
    ]
    legend_handles: list[object] = []
    legend_labels: list[str] = []
    for method, offset, marker, color, filled, short_label in method_specs:
        for index, regime in enumerate(regimes):
            row = select_one(
                regime_rows,
                method=method,
                regime=regime,
                metric="minFDE_K (m)",
            )
            estimate, low, high = (float(row[key]) for key in ("estimate", "ci_low", "ci_high"))
            container = ax.errorbar(
                index + offset,
                estimate,
                yerr=np.array([[estimate - low], [high - estimate]]),
                fmt=marker,
                markersize=6.0,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=1.3,
                ecolor=color,
                elinewidth=1.5,
                capsize=3.5,
                capthick=1.5,
                zorder=3,
            )
            if index == 0:
                legend_handles.append(container)
                legend_labels.append(short_label)
    ax.set_xlim(-0.38, 2.38)
    ax.set_ylim(1.50, 2.68)
    ax.set_xticks(range(3), regime_labels)
    ax.set_yticks(np.arange(1.6, 2.61, 0.2))
    ax.set_ylabel("minFDE$_6$ (m)")
    panel_title(ax, "(a) Graph mechanism by interaction regime")
    format_axes(ax, grid_axis="y")

    independent = select_one(control_rows, negative_control="independent: Full vs No graph")
    spatial = select_one(control_rows, negative_control="spatial: Full vs Spatial-only graph")
    ax.text(
        0.02,
        -0.18,
        (
            f"Independent Δ={float(independent['difference_full_minus_ablation']):+.3f} m: "
            "within ±0.25-m descriptive range\n"
            f"Spatial Δ={float(spatial['difference_full_minus_ablation']):+.3f} m: "
            "range not met (not TOST)"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.9,
        color=GRAY,
        linespacing=1.05,
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.995),
        ncol=3,
        frameon=False,
        fontsize=7.0,
        handletextpad=0.35,
        columnspacing=1.0,
    )

    # (b) Predicted-intent gating.
    ax = axes[0, 1]
    gating_specs = [
        ("Full temporal graph", "Full", 1.0, NAVY, True, "right"),
        ("No predicted-intent gating", "No intent gate", 0.0, RED, False, "left"),
    ]
    for method, label, y, color, filled, side in gating_specs:
        row = select_one(main_rows, method=method, metric="Brier-minFDE_6")
        estimate, low, high = (float(row[key]) for key in ("estimate", "ci_low", "ci_high"))
        _horizontal_ci(ax, estimate=estimate, low=low, high=high, y=y, color=color, filled=filled)
        _endpoint_label(ax, f"{estimate:.3f}", low=low, high=high, y=y, side=side, top_row=y == 1.0)
    gate_contrast = select_one(
        contrast_rows,
        comparison="Full temporal graph vs No predicted-intent gating",
    )
    ax.text(
        0.04,
        0.50,
        (
            f"Paired Δ={float(gate_contrast['difference_full_minus_ablation']):+.3f} "
            f"[{float(gate_contrast['ci_low']):+.3f}, "
            f"{float(gate_contrast['ci_high']):+.3f}]\n"
            "Effect supported"
        ),
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color=NAVY,
        linespacing=1.05,
    )
    ax.set_xlim(2.2, 10.55)
    ax.set_ylim(-0.24, 1.24)
    ax.set_yticks([1, 0], ["Full", "No intent gate"])
    ax.set_xlabel("Brier-minFDE$_6$ (lower is better)")
    panel_title(ax, "(b) Predicted-intent gating")
    format_axes(ax)

    # (c) Top-1 multimodal output versus independently fitted K=1.
    ax = axes[1, 0]
    top1_specs = [
        ("Full temporal graph", "Full top-1", 1.0, NAVY, "right"),
        ("Single mode K=1", "Independent K=1", 0.0, RED, "left"),
    ]
    for method, label, y, color, side in top1_specs:
        row = select_one(main_rows, method=method, metric="MR_1 at 2 m")
        estimate, low, high = (100.0 * float(row[key]) for key in ("estimate", "ci_low", "ci_high"))
        _horizontal_ci(ax, estimate=estimate, low=low, high=high, y=y, color=color)
        _endpoint_label(ax, f"{estimate:.2f}%", low=low, high=high, y=y, side=side, top_row=y == 1.0)
    top1_contrast = select_one(
        contrast_rows,
        comparison="Full top-1 vs independently trained Single mode K=1",
    )
    ax.text(
        0.04,
        0.50,
        (
            f"Paired Δ={100 * float(top1_contrast['difference_full_minus_ablation']):+.2f} pp "
            f"[{100 * float(top1_contrast['ci_low']):+.2f}, "
            f"{100 * float(top1_contrast['ci_high']):+.2f}]\n"
            "Benefit not clearly established"
        ),
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=RED,
        linespacing=1.05,
    )
    ax.set_xlim(75.70, 79.75)
    ax.set_ylim(-0.24, 1.24)
    ax.set_yticks([1, 0], ["Full top-1", "Independent K=1"])
    ax.set_xlabel("MR$_1$ at 2 m (%) — lower is better")
    panel_title(ax, "(c) Top-1 multimodal output vs K=1")
    format_axes(ax)

    # (d) CPU timing diagnostic.
    ax = axes[1, 1]
    latency_specs = [
        ("Full temporal graph", "Full", 4.0, NAVY, True, "right"),
        ("No graph", "No graph", 3.0, GRAY, False, "right"),
        ("Spatial-only graph", "Spatial-only", 2.0, GRAY, False, "right"),
        ("No predicted-intent gating", "No intent gate", 1.0, GRAY, False, "left"),
        ("Single mode K=1", "K=1", 0.0, RED, False, "right"),
    ]
    latency_metric = "End-to-end batch-1 CPU latency P95 (ms)"
    for method, label, y, color, filled, side in latency_specs:
        row = select_one(latency_rows, method=method, metric=latency_metric)
        estimate, low, high = (float(row[key]) for key in ("estimate", "ci_low", "ci_high"))
        _horizontal_ci(ax, estimate=estimate, low=low, high=high, y=y, color=color, filled=filled)
        _endpoint_label(ax, f"{estimate:.3f}", low=low, high=high, y=y, side=side, top_row=y == 4.0)
    ax.set_xlim(0.58, 1.055)
    ax.set_ylim(-0.24, 4.24)
    ax.set_yticks([4, 3, 2, 1, 0], ["Full", "No graph", "Spatial-only", "No intent gate", "K=1"])
    ax.set_xlabel("Batch-1 CPU latency P95 (ms)")
    panel_title(ax, "(d) Deployment timing diagnostic")
    format_axes(ax)

    return save_figure(fig, stem, width_px=width_px, height_px=height_px)

