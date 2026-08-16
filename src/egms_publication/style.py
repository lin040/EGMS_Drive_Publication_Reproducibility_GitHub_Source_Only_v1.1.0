from __future__ import annotations

from pathlib import Path
import struct

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NAVY = "#174A7E"
RED = "#D1495B"
TEAL = "#2A9D8F"
GRAY = "#6B7280"
GRID = "#D6D6D6"
TEXT = "#111111"


def apply_style(scale: float = 1.0) -> None:
    """Apply the manuscript's Figure-S1-derived graphics profile."""

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 9.3 * scale,
            "axes.labelsize": 10.0 * scale,
            "axes.labelpad": 5.0 * scale,
            "axes.linewidth": 0.9 * scale,
            "axes.edgecolor": "black",
            "axes.grid": False,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6 * scale,
            "grid.alpha": 0.70,
            "xtick.labelsize": 8.5 * scale,
            "ytick.labelsize": 8.5 * scale,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4.5 * scale,
            "ytick.major.size": 4.5 * scale,
            "xtick.major.width": 0.9 * scale,
            "ytick.major.width": 0.9 * scale,
            "xtick.major.pad": 3.5 * scale,
            "ytick.major.pad": 3.5 * scale,
            "legend.fontsize": 8.0 * scale,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "egms-drive-publication-v1",
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )


def format_axes(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.grid(True, which="major", axis=grid_axis)
    ax.tick_params(axis="both", which="major", top=False, right=False, bottom=True, left=True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(plt.rcParams["axes.linewidth"])


def panel_title(ax: plt.Axes, text: str, scale: float = 1.0) -> None:
    ax.set_title(text, loc="left", fontsize=10.5 * scale, fontweight="bold", pad=4.5 * scale)


def save_figure(
    fig: plt.Figure,
    stem: Path,
    *,
    width_px: int,
    height_px: int,
    dpi: int = 600,
) -> dict[str, Path]:
    """Save PNG/PDF/SVG without tight cropping and validate each file."""

    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches((width_px + 0.5) / dpi, (height_px + 0.5) / dpi, forward=True)
    paths = {suffix: stem.with_suffix(f".{suffix}") for suffix in ("png", "pdf", "svg")}
    fig.savefig(paths["pdf"], facecolor="white", edgecolor="white", bbox_inches=None, pad_inches=0)
    fig.savefig(paths["svg"], facecolor="white", edgecolor="white", bbox_inches=None, pad_inches=0)
    fig.savefig(
        paths["png"],
        dpi=dpi,
        facecolor="white",
        edgecolor="white",
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)

    data = paths["png"].read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or not data.endswith(b"IEND\xaeB`\x82"):
        raise ValueError(f"Invalid PNG: {paths['png']}")
    if struct.unpack(">II", data[16:24]) != (width_px, height_px):
        raise ValueError(f"Unexpected PNG dimensions: {paths['png']}")
    if paths["pdf"].stat().st_size < 1024 or not paths["pdf"].read_bytes().startswith(b"%PDF"):
        raise ValueError(f"Invalid or empty PDF: {paths['pdf']}")
    svg_head = paths["svg"].read_text(encoding="utf-8", errors="replace")[:1000]
    if "<svg" not in svg_head:
        raise ValueError(f"Invalid SVG: {paths['svg']}")
    return paths
