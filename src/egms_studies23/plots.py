from __future__ import annotations

import os
from pathlib import Path
import tempfile
from xml.etree import ElementTree
import zlib

_MPL_CACHE = Path(tempfile.gettempdir()) / "egms_studies23_matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "Full": "#176B87",
    "Equal weighting": "#64A7A0",
    "No alignment": "#D95F59",
    "No temporal consistency": "#E7A84B",
    "No modality dropout-distillation": "#7B6BA8",
    "Full temporal graph": "#176B87",
    "No graph": "#D95F59",
    "Spatial-only graph": "#E7A84B",
    "No predicted-intent gating": "#7B6BA8",
    "Single mode K=1": "#777777",
}


DISPLAY_LABELS = {
    "Full": "Full",
    "Equal weighting": "Equal",
    "No alignment": "No\nalign.",
    "No temporal consistency": "No\ntemporal",
    "No modality dropout-distillation": "No\ndropout",
    "Full temporal graph": "Full graph",
    "No graph": "No graph",
    "Spatial-only graph": "Spatial",
    "No predicted-intent gating": "No intent gate",
    "Single mode K=1": "K=1",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    specifications = (("png", 600), ("pdf", None), ("svg", None))
    for file_format, dpi in specifications:
        destination = directory / f"{stem}.{file_format}"
        temporary = directory / f".{stem}.{file_format}.tmp"
        for attempt in range(3):
            if temporary.exists():
                temporary.unlink()
            kwargs = {
                "format": file_format,
                "bbox_inches": "tight",
                "facecolor": "white",
            }
            if dpi is not None:
                kwargs["dpi"] = dpi
            fig.savefig(temporary, **kwargs)
            if _valid_figure_file(temporary, file_format):
                os.replace(temporary, destination)
                break
            if attempt == 2:
                raise RuntimeError(f"Failed to write a valid {file_format.upper()} figure: {stem}")
    plt.close(fig)


def _valid_figure_file(path: Path, file_format: str) -> bool:
    data = path.read_bytes()
    if file_format == "pdf":
        return data.startswith(b"%PDF-") and b"%%EOF" in data[-1024:]
    if file_format == "svg":
        try:
            ElementTree.parse(path)
            return True
        except ElementTree.ParseError:
            return False
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


def _metric_block(table: pd.DataFrame, metric: str, **filters: object) -> pd.DataFrame:
    block = table[table["metric"] == metric]
    for key, value in filters.items():
        block = block[block[key] == value]
    return block.copy()


def _bar_with_ci(ax: plt.Axes, block: pd.DataFrame, category: str, title: str, ylabel: str) -> None:
    labels = block[category].tolist()
    values = block["estimate"].to_numpy(float)
    lower = values - block["ci_low"].to_numpy(float)
    upper = block["ci_high"].to_numpy(float) - values
    positions = np.arange(len(labels))
    ax.bar(
        positions,
        values,
        yerr=np.vstack([lower, upper]),
        capsize=3,
        color=[COLORS.get(label, "#4C78A8") for label in labels],
        edgecolor="white",
        linewidth=0.5,
    )
    display_labels = [DISPLAY_LABELS.get(label, label) for label in labels]
    ax.set_xticks(positions, display_labels, rotation=0, fontsize=7)
    ax.tick_params(axis="x", pad=3)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)


def plot_all(
    s2_results: dict[str, pd.DataFrame],
    s3_results: dict[str, object],
    s2_tables: dict[str, pd.DataFrame],
    s3_tables: dict[str, pd.DataFrame],
    directory: Path,
) -> None:
    _style()
    plot_s2_alignment(s2_tables["table_s2_main"], directory)
    plot_s2_predictive(s2_tables["table_s2_main"], directory)
    plot_s2_missing(s2_results["study2_missing_modality"], directory)
    plot_s2_calibration(s2_results["study2_calibration_bins"], directory)
    plot_s3_metrics(s3_tables["table_s3_main"], directory)
    plot_s3_regimes(s3_tables["table_s3_by_regime"], directory)
    plot_s3_intent_latency(s3_tables, directory)
    plot_s3_confusion(s3_results["study3_predictions"], directory)
    plot_s3_examples(s3_results["study3_examples"], directory)


def plot_s2_alignment(table: pd.DataFrame, directory: Path) -> None:
    clean = table[table["domain"] == "clean_controlled"]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.3))
    _bar_with_ci(axes[0], _metric_block(clean, "SAS (Eq. 40)"), "method", "Standardized alignment", "SAS")
    _bar_with_ci(axes[1], _metric_block(clean, "Bidirectional Recall@1"), "method", "Cross-modal retrieval", "Recall@1")
    _bar_with_ci(axes[2], _metric_block(clean, "Held-out one-step feature prediction error"), "method", "Temporal prediction", "Prediction error")
    fig.suptitle("Study 2 controlled synthetic alignment and temporal diagnostics", fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Secondary error bars: 95% t CI across training replicates, conditional on fixed scenes; "
        "scene-decomposable primary contrasts use paired crossed bootstrap (SAS uses a paired replicate-only CI).",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    save_figure(fig, directory, "figure_s2_1_alignment_temporal")


def plot_s2_predictive(table: pd.DataFrame, directory: Path) -> None:
    stress = table[table["domain"] == "adverse_controlled_injected"]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.3))
    for ax, metric, title, ylabel in zip(
        axes,
        ["Action macro-F1", "NLL", "Multiclass Brier (0-2)", "ECE (15 equal-mass bins)"],
        ["Action classification", "Proper log score", "Quadratic score", "Calibration gap"],
        ["Macro-F1", "NLL", "Brier", "ECE"],
    ):
        _bar_with_ci(ax, _metric_block(stress, metric), "method", title, ylabel)
    fig.suptitle("Study 2 performance under controlled injected degradation", fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Macro-F1: higher is better; NLL/Brier/ECE: lower is better. Controlled injected perturbations are not CARLA natural-weather or RADIATE evidence.\n"
        "Secondary error bars: 95% t CI across training replicates, conditional on fixed scenes; the primary contrast uses paired crossed bootstrap.",
        ha="center",
        va="bottom",
        fontsize=7.2,
    )
    fig.tight_layout(rect=[0, 0.11, 1, 0.92])
    save_figure(fig, directory, "figure_s2_2_predictive_stress")


def plot_s2_missing(frame: pd.DataFrame, directory: Path) -> None:
    patterns = [p for p in frame["missing_pattern"].drop_duplicates() if p != "none"]
    methods = frame["method"].drop_duplicates().tolist()
    means = frame.groupby(["method", "missing_pattern"])["macro_f1_absolute_drop"].mean()
    matrix = np.array([[means.loc[(method, pattern)] for pattern in patterns] for method in methods])
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    limit = max(abs(matrix.min()), abs(matrix.max()), 0.01)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(patterns)), patterns)
    ax.set_yticks(np.arange(len(methods)), methods)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = 0.0 if abs(matrix[row, col]) < 0.0005 else matrix[row, col]
            ax.text(col, row, f"{value:+.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Clean - masked macro-F1 difference (controlled synthetic)")
    ax.set_xlabel("Masked modality pattern")
    fig.colorbar(
        image,
        ax=ax,
        label="Clean - masked macro-F1 (positive = drop; negative = apparent gain)",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    save_figure(fig, directory, "figure_s2_3_missing_modality")


def plot_s2_calibration(frame: pd.DataFrame, directory: Path) -> None:
    summary = frame.groupby("bin", as_index=False).agg(
        mean_confidence=("mean_confidence", "mean"),
        accuracy=("accuracy", "mean"),
        count=("count", "mean"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 5.8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    axes[0].plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1, label="Perfect calibration")
    active = summary["count"] > 0
    axes[0].plot(summary.loc[active, "mean_confidence"], summary.loc[active, "accuracy"], marker="o", color=COLORS["Full"], label="Full")
    axes[0].set_ylabel("Observed accuracy")
    axes[0].set_title("Study 2 reliability diagram, clean controlled test")
    axes[0].legend(frameon=False)
    axes[0].grid(color="#D9D9D9", linewidth=0.6)
    axes[1].bar(summary["mean_confidence"].fillna((summary["bin"] - 0.5) / 15), summary["count"], width=0.045, color="#90B6C5")
    axes[1].set_xlabel("Predicted confidence")
    axes[1].set_ylabel("Mean frames per replicate")
    fig.text(
        0.5,
        0.01,
        "15 deterministic equal-mass confidence bins; bin statistics are averaged across 10 training replicates.",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_figure(fig, directory, "figure_s2_4_reliability_diagram")


def plot_s3_metrics(table: pd.DataFrame, directory: Path) -> None:
    k6 = table[(table["K"] == 6) & table["method"].ne("Single mode K=1")]
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 4.2))
    for ax, metric, title, ylabel in zip(
        axes,
        ["minADE_6 (m)", "minFDE_6 (m)", "MR_6 at 2 m", "Brier-minFDE_6"],
        ["ADE of minFDE-selected\ntrajectory", "Best endpoint error", "Endpoint miss rate", "Probability-aware endpoint"],
        ["m", "m", "Rate", "Score"],
    ):
        _bar_with_ci(ax, _metric_block(k6, metric), "method", title, ylabel)
    fig.suptitle("Study 3 K=6 controlled synthetic trajectory metrics", fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "j* = argmin_j FDE_j; the same j* defines minADE_6 and Brier-minFDE_6 = FDE_j* + (1 - p_j*)^2. Lower is better.\n"
        "Secondary error bars: 95% t CI across training replicates, conditional on fixed scenes; primary contrasts use paired crossed bootstrap. "
        "Controlled synthetic only; not an Argoverse 2 benchmark result.",
        ha="center",
        va="bottom",
        fontsize=7.0,
    )
    fig.tight_layout(rect=[0, 0.13, 1, 0.92])
    save_figure(fig, directory, "figure_s3_1_trajectory_metrics")


def plot_s3_regimes(table: pd.DataFrame, directory: Path) -> None:
    block = table[(table["metric"] == "minFDE_K (m)") & table["method"].isin(["Full temporal graph", "No graph", "Spatial-only graph"])]
    regimes = ["independent", "spatial", "history_dependent"]
    methods = ["Full temporal graph", "No graph", "Spatial-only graph"]
    x = np.arange(len(regimes))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for index, method in enumerate(methods):
        selected = block[block["method"] == method].set_index("regime").loc[regimes]
        values = selected["estimate"].to_numpy(float)
        yerr = np.vstack([values - selected["ci_low"], selected["ci_high"] - values])
        ax.bar(x + (index - 1) * width, values, width, yerr=yerr, capsize=3, color=COLORS[method], label=method)
    ax.set_xticks(x, ["Independent-agent\nnull", "Spatial-only\ninteraction", "History-dependent\ninteraction"])
    ax.set_ylabel("minFDE₆ (m)")
    ax.set_title("minFDE₆ by prespecified synthetic interaction regime")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    fig.text(
        0.5,
        0.01,
        "Lower is better. Secondary error bars: 95% t CI across training replicates, conditional on fixed scenes; primary contrasts use paired crossed bootstrap.",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    save_figure(fig, directory, "figure_s3_2_mechanism_regimes")


def plot_s3_intent_latency(tables: dict[str, pd.DataFrame], directory: Path) -> None:
    main = _metric_block(tables["table_s3_main"], "Synthetic dominant-maneuver macro-F1")
    latency = _metric_block(tables["table_s3_latency"], "End-to-end batch-1 CPU latency P95 (ms)")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3))
    _bar_with_ci(axes[0], main, "method", "Synthetic maneuver classification", "Macro-F1")
    _bar_with_ci(axes[1], latency, "method", "End-to-end CPU latency", "P95 ms")
    fig.suptitle("Study 3 classification and runtime diagnostics", fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Macro-F1 error bars: secondary 95% t CI across training replicates, conditional on fixed scenes; primary contrasts use paired crossed bootstrap.\n"
        "Latency bars: median across replicate-level empirical P95 values with a 95% percentile bootstrap CI (10,000 draws); each P95 uses 120 batch-1 scenes on Linux x86_64, with no dedicated warm-up; diagnostic only, not real-time.\n"
        "Full graph, no intent gate, and K=1 share the intent head, so identical macro-F1 is expected.",
        ha="center",
        va="bottom",
        fontsize=6.8,
    )
    fig.tight_layout(rect=[0, 0.17, 1, 0.92])
    save_figure(fig, directory, "figure_s3_3_intent_latency")


def plot_s3_confusion(frame: pd.DataFrame, directory: Path) -> None:
    block = frame[frame["method"] == "Full temporal graph"]
    matrix = np.zeros((8, 8), dtype=float)
    for truth, pred in zip(block["intent_true"].to_numpy(int), block["intent_pred"].to_numpy(int)):
        matrix[truth, pred] += 1
    matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    labels = ["straight", "slow", "stop", "cut-in", "lane change", "turn L", "turn R", "cross"]
    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(8), labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(8), labels)
    ax.set_xlabel("Predicted synthetic dominant maneuver")
    ax.set_ylabel("True synthetic dominant maneuver")
    ax.set_title("Full temporal graph: row-normalized confusion matrix")
    for row in range(8):
        for col in range(8):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=7, color="white" if matrix[row, col] > 0.55 else "black")
    fig.colorbar(image, ax=ax, label="Row proportion")
    fig.text(
        0.5,
        0.01,
        "Row-normalized predictions pooled across 10 training replicates; all 8 synthetic dynamic maneuver classes have test support.",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_figure(fig, directory, "figure_s3_4_intent_confusion")


def plot_s3_examples(payload: dict[str, np.ndarray], directory: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.8))
    for index, ax in enumerate(axes.flat):
        history = payload["history"][index]
        future = payload["future"][index]
        candidates = payload["candidates"][index]
        probabilities = payload["probabilities"][index]
        ax.plot(history[:, 0], history[:, 1], color="#555555", linewidth=2, label="Observed history")
        for mode in range(candidates.shape[0]):
            alpha = 0.25 + 0.65 * probabilities[mode] / max(probabilities.max(), 1e-12)
            label = "Predicted modes (opacity proportional to probability)" if mode == 0 else "_nolegend_"
            ax.plot(candidates[mode, :, 0], candidates[mode, :, 1], color="#4C9F70", alpha=alpha, linewidth=1.2, label=label)
        ax.plot(future[:, 0], future[:, 1], color="#D95F59", linewidth=2.2, label="Realized future")
        ax.scatter([0], [0], color="#176B87", s=28, zorder=5)
        ax.set_title(str(payload["scene_ids"][index]))
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(color="#E5E5E5", linewidth=0.5)
        ax.set_xlabel("Longitudinal displacement (m)")
        ax.set_ylabel("Lateral displacement (m)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("Full temporal graph K=6 trajectory examples (controlled synthetic)", y=0.99, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Fixed test-scene indices [3, 19, 41, 77] from the first prespecified training replicate; examples were not error-selected.",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.87])
    save_figure(fig, directory, "figure_s3_5_trajectory_examples")
