from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from PIL import Image

from .utils import sha256, write_json


EXPECTED_FIGURES = {
    "manuscript/figures/Figure_2_Study1.png": (3810, 2522),
    "manuscript/figures/Figure_3_Study2.png": (3810, 2472),
    "manuscript/figures/Figure_4_Study3.png": (3810, 2485),
    "power_full/figures/figure1_unpaired_power_curve.png": (3720, 2846),
}

EXPECTED_TABLE_ROWS = {
    "manuscript/tables/Table_2_main_effects.csv": 16,
    "supplement/compact/Tables/Table_S2_planning_summary.csv": 9,
    "supplement/compact/Tables/Table_S3_condensed_planning.csv": 7,
    "supplement/compact/Tables/Table_S4_validation_checks.csv": 33,
}

POWER_FIGURE_STEMS = (
    "figure1_unpaired_power_curve",
    "figure2_paired_correlation",
    "figure3_cluster_sensitivity",
    "figure4_event_rate_sensitivity",
    "figure5_precision_curve",
    "figure6_allocation_by_seed",
)


def validate_input_boundary(data_root: Path, source_root: Path) -> list[str]:
    """Reject image/Word inputs and source code that attempts to read them."""

    failures: list[str] = []
    disallowed = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf", ".svg", ".docx"}
    for path in data_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in disallowed:
            failures.append(f"Disallowed plotting input: {path}")
    for path in source_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if ("word" + "/media") in text:
            failures.append(f"Source refers to Word media: {path}")
    return failures


def _validate_png(path: Path, expected: tuple[int, int]) -> None:
    with Image.open(path) as image:
        if image.size != expected:
            raise ValueError(f"{path}: expected {expected}, found {image.size}")
        dpi = image.info.get("dpi", (0.0, 0.0))
        if not all(595.0 <= float(value) <= 605.0 for value in dpi[:2]):
            raise ValueError(f"{path}: expected approximately 600 dpi, found {dpi}")


def _validate_siblings(png_path: Path) -> None:
    pdf = png_path.with_suffix(".pdf")
    svg = png_path.with_suffix(".svg")
    pdf_bytes = pdf.read_bytes()
    if len(pdf_bytes) < 1024 or not pdf_bytes.startswith(b"%PDF") or b"%%EOF" not in pdf_bytes[-2048:]:
        raise ValueError(f"Invalid or empty PDF: {pdf}")
    if svg.stat().st_size < 1024:
        raise ValueError(f"Invalid or empty SVG: {svg}")
    ET.parse(svg)


def validate_outputs(repo_root: Path, output_root: Path) -> dict[str, object]:
    failures = validate_input_boundary(repo_root / "data", repo_root / "src" / "egms_publication")
    for relative, dimensions in EXPECTED_FIGURES.items():
        path = output_root / relative
        try:
            _validate_png(path, dimensions)
            _validate_siblings(path)
        except Exception as exc:  # noqa: BLE001 - collect all QA failures
            failures.append(str(exc))
    for stem in POWER_FIGURE_STEMS:
        png = output_root / "power_full" / "figures" / f"{stem}.png"
        try:
            if not png.is_file() or png.stat().st_size < 1024:
                raise ValueError(f"Missing or empty power PNG: {png}")
            _validate_siblings(png)
        except Exception as exc:  # noqa: BLE001 - collect all QA failures
            failures.append(str(exc))
    for relative, expected_rows in EXPECTED_TABLE_ROWS.items():
        path = output_root / relative
        if not path.is_file():
            failures.append(f"Missing table: {path}")
            continue
        rows = len(pd.read_csv(path))
        if rows != expected_rows:
            failures.append(f"{path}: expected {expected_rows} rows, found {rows}")
    table_s4 = output_root / "supplement/compact/Tables/Table_S4_validation_checks.csv"
    if table_s4.is_file() and not pd.read_csv(table_s4)["Status"].eq("Pass").all():
        failures.append("Table S4 contains a non-passing validation check")

    report = {
        "passed": not failures,
        "failures": failures,
        "figure_checks": EXPECTED_FIGURES,
        "table_row_checks": EXPECTED_TABLE_ROWS,
        "power_figure_format_checks": list(POWER_FIGURE_STEMS),
        "interpretation_boundary": (
            "Study 1 is an exported-summary graphical reproduction; Studies 2–3 are "
            "controlled synthetic mechanism surrogates; power results are prospective "
            "planning quantities. None is CARLA or real-world safety evidence."
        ),
    }
    write_json(output_root / "validation_report.json", report)
    if failures:
        raise RuntimeError("Publication validation failed:\n- " + "\n- ".join(failures))
    return report


def write_manifest(repo_root: Path, output_root: Path) -> Path:
    input_files = sorted(
        path for base in (repo_root / "configs", repo_root / "data")
        for path in base.rglob("*") if path.is_file()
    )
    output_files = sorted(
        path for path in output_root.rglob("*")
        if path.is_file() and path.name not in {"artifact_manifest.json", "publication_outputs.zip"}
    )
    payload = {
        "schema": "egms-publication-artifact-manifest-1.0",
        "inputs": {
            str(path.relative_to(repo_root)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in input_files
        },
        "outputs": {
            str(path.relative_to(output_root)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in output_files
        },
    }
    manifest = output_root / "artifact_manifest.json"
    write_json(manifest, payload)
    return manifest


def verify_manifest(output_root: Path, manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for relative, expected in payload["outputs"].items():
        path = output_root / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Manifest mismatch: {relative}")
