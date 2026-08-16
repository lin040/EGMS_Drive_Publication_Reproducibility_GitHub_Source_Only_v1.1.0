"""Build the readable EGMS-Drive publication-reproduction Colab notebook.

Every canonical Python, YAML, and CSV input is presented verbatim in its own
``%%writefile`` cell. This intentionally favors human auditability over a
compact notebook size: no source file is hidden in an encoded or compressed
payload.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "EGMS_Drive_Publication_Reproduction_Colab.ipynb"

# Complete executable sources for the fast publication reproduction and the
# optional full controlled Studies 2-3 refit, followed by the canonical
# text configuration and numeric CSV inputs.
CANONICAL_PATHS = (
    "run_publication.py",
    "run_studies.py",
    "src/egms_publication/__init__.py",
    "src/egms_publication/figures.py",
    "src/egms_publication/runner.py",
    "src/egms_publication/style.py",
    "src/egms_publication/tables.py",
    "src/egms_publication/utils.py",
    "src/egms_publication/validation.py",
    "src/egms_power/__init__.py",
    "src/egms_power/cli.py",
    "src/egms_power/config.py",
    "src/egms_power/pipeline.py",
    "src/egms_power/statistics.py",
    "src/egms_studies23/__init__.py",
    "src/egms_studies23/common.py",
    "src/egms_studies23/plots.py",
    "src/egms_studies23/reporting.py",
    "src/egms_studies23/runner.py",
    "src/egms_studies23/study2.py",
    "src/egms_studies23/study3.py",
    "src/egms_studies23/validation.py",
    "configs/power_protocol.yaml",
    "configs/studies23.yaml",
    "data/study1/study1_figure_inputs.csv",
    "data/study1/study1_table2_inputs.csv",
    "data/studies23_frozen/tables/table_s2_primary_contrasts.csv",
    "data/studies23_frozen/tables/table_s3_by_regime.csv",
    "data/studies23_frozen/tables/table_s3_latency.csv",
    "data/studies23_frozen/tables/table_s3_main.csv",
    "data/studies23_frozen/tables/table_s3_negative_controls.csv",
    "data/studies23_frozen/tables/table_s3_primary_contrasts.csv",
)

FORBIDDEN_EMBEDDED_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".pdf",
    ".svg",
    ".docx",
    ".gz",
    ".zip",
}


def _canonical_texts() -> tuple[dict[str, str], dict[str, str]]:
    """Return verbatim UTF-8 source text and its canonical SHA-256 digests."""

    texts: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for relative in CANONICAL_PATHS:
        path = ROOT / relative
        if path.suffix.lower() in FORBIDDEN_EMBEDDED_SUFFIXES:
            raise ValueError(f"Forbidden notebook input: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Canonical input is not readable UTF-8 text: {relative}") from exc
        if text.encode("utf-8") != data:
            raise ValueError(f"UTF-8 round-trip changed canonical input: {relative}")
        texts[relative] = text
        hashes[relative] = hashlib.sha256(data).hexdigest()
    return texts, hashes


def _markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code(source: str, *, tags: list[str] | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if tags:
        metadata["tags"] = tags
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source,
    }


def _writefile_cell(relative: str, text: str) -> dict[str, object]:
    """Create one human-readable Colab cell containing one complete file."""

    if not text.endswith("\n"):
        raise ValueError(f"Canonical text must end with a newline: {relative}")
    return _code(
        f"%%writefile {relative}\n{text}",
        tags=["canonical-source", "readable-writefile"],
    )


def build() -> Path:
    texts, hashes = _canonical_texts()
    path_literal = json.dumps(list(CANONICAL_PATHS), ensure_ascii=False, indent=2)
    hash_literal = json.dumps(hashes, ensure_ascii=False, indent=2, sort_keys=True)

    prepare_source = f'''import importlib.metadata
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from packaging.requirements import Requirement
except ModuleNotFoundError:
    from pip._vendor.packaging.requirements import Requirement

try:
    IS_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:
    IS_COLAB = False

requirements = {{
    "numpy": "numpy>=2.0,<3",
    "pandas": "pandas>=2.2,<3",
    "scipy": "scipy>=1.12,<2",
    "matplotlib": "matplotlib>=3.8,<4",
    "seaborn": "seaborn>=0.13,<1",
    "yaml": "PyYAML>=6.0,<7",
    "PIL": "Pillow>=10,<13",
    "sklearn": "scikit-learn>=1.4,<2",
}}


def needs_install(module, spec):
    if importlib.util.find_spec(module) is None:
        return True
    requirement = Requirement(spec)
    try:
        installed = importlib.metadata.version(requirement.name)
    except importlib.metadata.PackageNotFoundError:
        return True
    return not requirement.specifier.contains(installed, prereleases=True)


missing_or_incompatible = [
    spec for module, spec in requirements.items() if needs_install(module, spec)
]
if missing_or_incompatible:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            *missing_or_incompatible,
        ],
        check=True,
    )

print("Runtime:", "Google Colab" if IS_COLAB else "local validation")
for distribution in ("numpy", "pandas", "scipy", "matplotlib", "seaborn", "PyYAML", "Pillow", "scikit-learn"):
    try:
        print(f"{{distribution}}: {{importlib.metadata.version(distribution)}}")
    except importlib.metadata.PackageNotFoundError:
        pass

CANONICAL_PATHS = {path_literal}
requested_work_root = os.environ.get("EGMS_COLAB_WORK_ROOT", "").strip()
if requested_work_root:
    WORK_ROOT = Path(requested_work_root).expanduser().resolve()
    WORK_ROOT.mkdir(parents=True, exist_ok=False)
else:
    runtime_parent = Path("/content") if IS_COLAB else None
    WORK_ROOT = Path(tempfile.mkdtemp(prefix="EGMS_Drive_Publication_", dir=runtime_parent))
for relative in CANONICAL_PATHS:
    (WORK_ROOT / relative).parent.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_ROOT)
print("Working directory:", WORK_ROOT)
print(f"Prepared directories for {{len(CANONICAL_PATHS)}} readable source/config/CSV files")
'''

    verify_source = f'''import hashlib
import os
import sys
from pathlib import Path

EXPECTED_HASHES = {hash_literal}
FORBIDDEN_INPUT_SUFFIXES = {{".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf", ".svg", ".docx", ".gz", ".zip"}}

assert set(CANONICAL_PATHS) == set(EXPECTED_HASHES)
assert not any(Path(relative).suffix.lower() in FORBIDDEN_INPUT_SUFFIXES for relative in CANONICAL_PATHS)
for relative in CANONICAL_PATHS:
    source_path = WORK_ROOT / relative
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    observed = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if observed != EXPECTED_HASHES[relative]:
        raise RuntimeError(f"Canonical source hash mismatch: {{relative}}")

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(WORK_ROOT / ".mplconfig")
sys.path.insert(0, str(WORK_ROOT / "src"))
print(f"SHA-256 verified {{len(CANONICAL_PATHS)}} readable source/config/CSV files")
print("No image, PDF, Word, archive, or compressed data file was embedded as an input")
'''

    cells: list[dict[str, object]] = [
        _markdown(
            """# EGMS-Drive publication figures and tables: readable, self-contained reproduction

Run all cells to regenerate **Figures 2–4, Figure S1, Table 2, and Tables S2–S4** from the same Python source, frozen YAML protocols, and machine-readable numeric inputs used by the GitHub release. Every program, configuration, and CSV input appears below as complete readable text in its own `%%writefile` cell. No source is hidden in an encoded or compressed payload, and the notebook does not embed or read any PNG, PDF, SVG, Word, archive, or compressed data file as a plotting input.

Evidence boundary: Study 1 is an audited exported-summary reproduction; Studies 2–3 are controlled synthetic mechanism surrogates; the power analysis is prospective. These outputs are not CARLA, public-dataset, real-vehicle, or empirical LLM evidence."""
        ),
        _markdown("## 1. Check the runtime and prepare a clean working directory"),
        _code(prepare_source),
        _markdown(
            """## 2. Write and SHA-256-verify the canonical release inputs

The following cells show each complete Python, YAML, and CSV file directly. They are ordinary readable source cells; editing a cell changes the file that is written and causes the later SHA-256 verification to fail until the expected release hash is deliberately updated."""
        ),
    ]

    for relative in CANONICAL_PATHS:
        cells.append(_markdown(f"### `{relative}`"))
        cells.append(_writefile_cell(relative, texts[relative]))

    cells.extend(
        [
            _markdown("### Verify every written canonical input against its release SHA-256"),
            _code(verify_source),
            _markdown("## 3. Execute the complete publication-reproduction pipeline"),
            _code(
                '''from pathlib import Path
from egms_publication.runner import run_publication

result = run_publication(Path("outputs/colab_publication_reproduction"))
OUTPUT_DIR = Path(result["output_directory"])
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = WORK_ROOT / OUTPUT_DIR
assert result["validation"]["passed"] is True
print("Publication reproduction: PASS")
print("Output directory:", OUTPUT_DIR)
print("Table rows:", result["table_rows"])
'''
            ),
            _markdown("## 4. Display Table 2 and Supplementary Tables S2–S4"),
            _code(
                '''import html
import importlib.util
import pandas as pd

table_paths = [
    ("Table 2 — principal effect estimates", OUTPUT_DIR / "manuscript/tables/Table_2_main_effects.csv"),
    ("Table S2 — planning summary", OUTPUT_DIR / "supplement/compact/Tables/Table_S2_planning_summary.csv"),
    ("Table S3 — condensed planning sensitivity", OUTPUT_DIR / "supplement/compact/Tables/Table_S3_condensed_planning.csv"),
    ("Table S4 — 33 validation checks", OUTPUT_DIR / "supplement/compact/Tables/Table_S4_validation_checks.csv"),
]

HAS_IPYTHON = importlib.util.find_spec("IPython") is not None
if HAS_IPYTHON:
    from IPython.display import HTML, display

TABLE_CSS = """
<style>
.egms-table-card {font-family: Arial, sans-serif; margin: 1.25rem 0 2rem;}
.egms-table-card h3 {color: #17365d; margin: 0 0 .35rem; font-size: 1.15rem;}
.egms-table-card .egms-meta {color: #5f6b7a; margin: 0 0 .6rem; font-size: .9rem;}
.egms-table-wrap {overflow: auto; max-height: 640px; border: 1px solid #cbd5e1; border-radius: 6px;}
.egms-table {border-collapse: collapse; width: 100%; font-size: .88rem; line-height: 1.35;}
.egms-table th {background: #17365d; color: white; padding: .55rem; text-align: left; white-space: nowrap; position: sticky; top: 0; z-index: 1;}
.egms-table td {border-top: 1px solid #dbe3ec; padding: .5rem .55rem; vertical-align: top;}
.egms-table tbody tr:nth-child(even) {background: #f5f8fb;}
.egms-table tbody tr:hover {background: #eaf2fb;}
</style>
"""

def render_table_html(title, path, frame):
    table = frame.to_html(index=False, border=0, classes="egms-table", na_rep="—", escape=True)
    return (
        TABLE_CSS
        + '<section class="egms-table-card">'
        + f"<h3>{html.escape(title)}</h3>"
        + f'<p class="egms-meta">{len(frame)} rows · Source: {html.escape(str(path.relative_to(OUTPUT_DIR)))}</p>'
        + f'<div class="egms-table-wrap">{table}</div></section>'
    )

for title, path in table_paths:
    frame = pd.read_csv(path)
    if HAS_IPYTHON:
        display(HTML(render_table_html(title, path, frame)))
    else:
        print("\\n", title)
        print(frame.to_string(index=False))
'''
            ),
            _markdown("## 5. Display generated Figures 2–4 and Figure S1"),
            _code(
                '''from PIL import Image as PILImage

figure_paths = [
    ("Figure 2 — Study 1 exported-summary diagnostics", OUTPUT_DIR / "manuscript/figures/Figure_2_Study1.png"),
    ("Figure 3 — Study 2 mechanism contrasts", OUTPUT_DIR / "manuscript/figures/Figure_3_Study2.png"),
    ("Figure 4 — Study 3 graph, intent, and trajectory diagnostics", OUTPUT_DIR / "manuscript/figures/Figure_4_Study3.png"),
    ("Figure S1 — prospective independent-design power", OUTPUT_DIR / "power_full/figures/figure1_unpaired_power_curve.png"),
]

if HAS_IPYTHON:
    from IPython.display import HTML, Image as IPythonImage, display

for title, path in figure_paths:
    assert path.is_file() and path.stat().st_size > 0
    with PILImage.open(path) as generated_figure:
        width_px, height_px = generated_figure.size
    if HAS_IPYTHON:
        relative_path = path.relative_to(OUTPUT_DIR)
        display(HTML(
            '<section style="font-family:Arial,sans-serif;margin:1.25rem 0 .5rem">'
            f'<h3 style="color:#17365d;margin:0 0 .3rem">{title}</h3>'
            f'<p style="color:#5f6b7a;margin:0">{width_px}×{height_px} px · Generated from code · {relative_path}</p>'
            '</section>'
        ))
        display(IPythonImage(filename=str(path), width=1050))
    else:
        print(f"{title} [{width_px}×{height_px} px] -> {path}")
'''
            ),
            _markdown("## 6. Verify and download the generated result ZIP"),
            _code(
                '''import hashlib
import json

validation_path = OUTPUT_DIR / "validation_report.json"
validation = json.loads(validation_path.read_text(encoding="utf-8"))
assert validation["passed"] is True, validation["failures"]

ZIP_PATH = Path(result["output_zip"])
if not ZIP_PATH.is_absolute():
    ZIP_PATH = WORK_ROOT / ZIP_PATH
assert ZIP_PATH.is_file() and ZIP_PATH.stat().st_size > 0
zip_sha256 = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()
print("Validation: PASS")
print("Generated ZIP:", ZIP_PATH)
print("ZIP bytes:", ZIP_PATH.stat().st_size)
print("ZIP SHA-256:", zip_sha256)

if IS_COLAB:
    from google.colab import files
    files.download(str(ZIP_PATH))
'''
            ),
            _markdown(
                """## 7. Optional full controlled Studies 2–3 refit

The readable notebook includes `run_studies.py` and every module under `src/egms_studies23/`, so the complete controlled synthetic refit is inspectable. It is intentionally disabled by default because it is much slower and produces large intermediate prediction files. Set the flag in the next cell to `True` only when that full refit is required."""
            ),
            _code(
                '''RUN_FULL_CONTROLLED_REFIT = False

if RUN_FULL_CONTROLLED_REFIT:
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "run_studies.py",
            "run",
            "--config",
            "configs/studies23.yaml",
            "--output",
            "outputs/full_controlled_refit",
        ],
        check=True,
    )
    print("Full controlled Studies 2–3 refit: complete")
else:
    print("Full controlled Studies 2–3 refit skipped (default).")
'''
            ),
            _markdown(
                """## Interpretation boundary

The notebook regenerates the publication graphics and tables from numeric inputs and executable statistical code; it never copies an attached manuscript image. Study 1 remains limited by unavailable raw training/evaluation provenance. Studies 2–3 remain controlled synthetic mechanism validation, and all power values remain prospective planning quantities."""
            ),
        ]
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    build()
