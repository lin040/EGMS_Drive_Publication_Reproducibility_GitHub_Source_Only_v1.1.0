"""Direction-neutral publication figures for exploratory Study 1-R2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from egms_publication.style import GRAY, NAVY, TEAL, TEXT, apply_style, format_axes, panel_title, save_figure


def _select_one(frame: pd.DataFrame, **filters: object) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in filters.items():
        mask &= frame[column].astype(str).eq(str(value))
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}; found {len(selected)}")
    return selected.iloc[0]


def _symmetric_limits(values: np.ndarray, floor: float) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    limit = max(floor, float(np.max(np.abs(finite))) if len(finite) else floor)
    return -1.22 * limit, 1.22 * limit


def _range_limits(values: np.ndarray, *, nonnegative: bool = False) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return (0.0, 1.0)
    low = float(finite.min())
    high = float(finite.max())
    span = max(high - low, 0.08 * max(abs(high), abs(low), 1.0))
    lower = low - 0.18 * span
    upper = high + 0.18 * span
    if nonnegative:
        lower = max(0.0, lower)
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def _p_text(value: float) -> str:
    return f"{value:.4f}" if value >= 0.0001 else "<0.0001"


def _forest_panel(
    ax: plt.Axes,
    contrasts: pd.DataFrame,
    endpoints: list[str],
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
    for index, (row, estimate, low, high, ypos) in enumerate(zip(rows, estimates, lows, highs, y)):
        ax.errorbar(
            estimate,
            ypos,
            xerr=np.array([[estimate - low], [high - estimate]]),
            fmt="o",
            markersize=6.0,
            markerfacecolor="white",
            markeredgecolor=NAVY,
            markeredgewidth=1.35,
            ecolor=NAVY,
            elinewidth=1.5,
            capsize=3.6,
            capthick=1.5,
            zorder=3,
        )
        annotation = f"Δb={estimate:+.{decimals}f}"
        ax.annotate(
            annotation,
            (0.985, ypos),
            xycoords=("axes fraction", "data"),
            xytext=(0, -6 if index == 0 else 5),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=7.5,
            color=TEXT,
        )
    ax.axvline(0.0, color=TEAL, linestyle="--", linewidth=1.35, zorder=1)
    ax.set_xlim(*_symmetric_limits(np.r_[estimates, lows, highs], floor))
    ax.set_ylim(-0.28, len(rows) - 0.72)
    ax.set_yticks(y, [str(row["metric"]) for row in rows])
    ax.set_xlabel(xlabel)
    panel_title(ax, title)
    format_axes(ax)


def _marginal_panel(
    ax: plt.Axes,
    metric_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    endpoint: str,
    *,
    title: str,
    xlabel: str,
    decimals: int,
) -> None:
    rows = [
        _select_one(metric_summary, endpoint=endpoint, method="Baseline B"),
        _select_one(metric_summary, endpoint=endpoint, method="Structured-R2"),
    ]
    colors = [GRAY, NAVY]
    y = [0.34, 0.66]
    bounds: list[float] = []
    for row, color, ypos in zip(rows, colors, y):
        estimate = float(row["estimate"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        bounds.extend([estimate, low, high])
        ax.errorbar(
            estimate,
            ypos,
            xerr=np.array([[estimate - low], [high - estimate]]),
            fmt="o",
            markersize=6.2,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.4,
            ecolor=color,
            elinewidth=1.55,
            capsize=3.8,
            capthick=1.55,
            zorder=3,
        )
        ax.annotate(
            f"{estimate:.{decimals}f}",
            (estimate, ypos),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=TEXT,
        )
    contrast = _select_one(contrasts, endpoint=endpoint)
    delta = float(contrast["raw_difference_structured_minus_baseline"])
    low = float(contrast["raw_ci_low"])
    high = float(contrast["raw_ci_high"])
    annotation = (
        f"Paired Δ(S−B)={delta:+.{decimals}f} [{low:+.{decimals}f}, {high:+.{decimals}f}]\n"
        f"favorable/tie/adverse={int(contrast['favorable_replicates'])}/"
        f"{int(contrast['tied_replicates'])}/{int(contrast['adverse_replicates'])}; "
        f"Holm p={_p_text(float(contrast['holm_adjusted_p']))}"
    )
    ax.text(
        0.50,
        0.08,
        annotation,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=TEXT,
    )
    ax.set_xlim(*_range_limits(np.asarray(bounds), nonnegative=True))
    ax.set_ylim(0.12, 0.88)
    ax.set_yticks(y, ["Baseline B", "Structured-R2"])
    ax.set_xlabel(xlabel)
    panel_title(ax, title)
    format_axes(ax)


def build_figure2_study1r2(
    metric_summary: pd.DataFrame | str | Path,
    paired_contrasts: pd.DataFrame | str | Path,
    stem: str | Path,
) -> dict[str, Path]:
    """Render the raw-recomputed four-panel exploratory Study 1-R2 figure."""

    summary = (
        metric_summary.copy()
        if isinstance(metric_summary, pd.DataFrame)
        else pd.read_csv(Path(metric_summary))
    )
    contrasts = (
        paired_contrasts.copy()
        if isinstance(paired_contrasts, pd.DataFrame)
        else pd.read_csv(Path(paired_contrasts))
    )
    apply_style()
    width_px, height_px = 3810, 2522
    fig, axes = plt.subplots(2, 2, figsize=(width_px / 600, height_px / 600), dpi=600)
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.105, top=0.950, wspace=0.42, hspace=0.50)

    _forest_panel(
        axes[0, 0],
        contrasts,
        ["macro_f1", "nll", "brier", "ece"],
        title="(a) Offline action and probability metrics",
        xlabel="Benefit-oriented raw difference\n(positive = Structured-R2 better)",
        scale=1.0,
        decimals=4,
        floor=0.01,
    )
    _forest_panel(
        axes[0, 1],
        contrasts,
        ["collision", "near_miss", "critical_event", "route_completion"],
        title="(b) Closed-loop outcomes",
        xlabel="Benefit-oriented difference (pp)\n(positive = Structured-R2 better)",
        scale=100.0,
        decimals=2,
        floor=1.0,
    )
    _marginal_panel(
        axes[1, 0],
        summary,
        contrasts,
        "ttc_p5",
        title="(c) Jointly collision-free TTC-P5",
        xlabel="Mean episode TTC-P5 (s)",
        decimals=3,
    )
    _marginal_panel(
        axes[1, 1],
        summary,
        contrasts,
        "jerk_p95",
        title="(d) Episode kinematic smoothness",
        xlabel="Mean episode jerk-P95 (m/s³)",
        decimals=3,
    )
    return save_figure(
        fig,
        Path(stem),
        width_px=width_px,
        height_px=height_px,
        dpi=600,
    )


# Compatibility aliases for copied orchestration code.
build_figure2_study1r = build_figure2_study1r2
build_figure2 = build_figure2_study1r2


_VARIANT_STYLE = {
    "Frozen Study 1-R": {"marker": "o", "color": GRAY, "face": "white", "offset": 0.11},
    "Exploratory Study 1-R2": {"marker": "s", "color": NAVY, "face": NAVY, "offset": -0.11},
}


def _comparison_forest_panel(
    ax: plt.Axes,
    inputs: pd.DataFrame,
    endpoints: tuple[str, ...],
    *,
    title: str,
    xlabel: str,
    floor: float,
) -> None:
    base_y = np.arange(len(endpoints) - 1, -1, -1, dtype=float)
    bounds: list[float] = []
    for variant, style in _VARIANT_STYLE.items():
        rows = inputs.loc[inputs["study_variant"].eq(variant)].set_index("endpoint").loc[list(endpoints)]
        estimate = rows["display_estimate"].to_numpy(float)
        low = rows["display_ci_low"].to_numpy(float)
        high = rows["display_ci_high"].to_numpy(float)
        y = base_y + float(style["offset"])
        bounds.extend(np.r_[estimate, low, high])
        ax.errorbar(
            estimate, y, xerr=np.vstack([estimate - low, high - estimate]),
            fmt=str(style["marker"]), markersize=5.8,
            markerfacecolor=str(style["face"]), markeredgecolor=str(style["color"]),
            markeredgewidth=1.3, ecolor=str(style["color"]), elinewidth=1.45,
            capsize=3.4, capthick=1.45, zorder=3,
        )
    ax.axvline(0.0, color=TEAL, linestyle="--", linewidth=1.35)
    ax.set_xlim(*_symmetric_limits(np.asarray(bounds), floor))
    ax.set_ylim(-0.38, len(endpoints) - 0.62)
    labels = [str(inputs.loc[inputs["endpoint"].eq(e), "metric"].iloc[0]) for e in endpoints]
    ax.set_yticks(base_y, labels)
    ax.set_xlabel(xlabel)
    panel_title(ax, title)
    format_axes(ax)


def build_figure2r_study1r_vs_r2(
    comparison_inputs: pd.DataFrame | str | Path,
    stem: str | Path,
) -> dict[str, Path]:
    frame = comparison_inputs.copy() if isinstance(comparison_inputs, pd.DataFrame) else pd.read_csv(Path(comparison_inputs))
    if len(frame) != 20:
        raise ValueError("Figure 2R requires two variants by ten endpoints")
    apply_style()
    width_px, height_px = 3810, 2522
    fig, axes = plt.subplots(2, 2, figsize=(width_px / 600, height_px / 600), dpi=600)
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.145, top=0.885, wspace=0.42, hspace=0.52)
    _comparison_forest_panel(
        axes[0, 0], frame, ("macro_f1", "nll", "brier", "ece"),
        title="(a) Offline action and probability metrics",
        xlabel="Within-study benefit-oriented difference\n(positive = candidate better)", floor=0.01,
    )
    _comparison_forest_panel(
        axes[0, 1], frame, ("collision", "near_miss", "critical_event", "route_completion"),
        title="(b) Closed-loop outcomes",
        xlabel="Within-study benefit-oriented difference (pp)\n(positive = candidate better)", floor=1.0,
    )
    _comparison_forest_panel(
        axes[1, 0], frame, ("ttc_p5",), title="(c) Jointly collision-free TTC-P5",
        xlabel="Within-study benefit-oriented difference (s)", floor=0.05,
    )
    _comparison_forest_panel(
        axes[1, 1], frame, ("jerk_p95",), title="(d) Episode kinematic smoothness",
        xlabel="Within-study benefit-oriented difference (m/s³)", floor=0.05,
    )
    handles = [
        Line2D([0], [0], marker=s["marker"], color=s["color"], markerfacecolor=s["face"],
               markeredgecolor=s["color"], linestyle="-", markersize=6, label=v)
        for v, s in _VARIANT_STYLE.items()
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=2, frameon=False)
    fig.text(
        0.5, 0.025,
        "Within-study contrasts only; no direct R2-versus-R1 test. Study 1-R2 is post hoc exploratory and all endpoints are retained.",
        ha="center", va="bottom", fontsize=7.5, color=TEXT,
    )
    return save_figure(fig, Path(stem), width_px=width_px, height_px=height_px, dpi=600)
