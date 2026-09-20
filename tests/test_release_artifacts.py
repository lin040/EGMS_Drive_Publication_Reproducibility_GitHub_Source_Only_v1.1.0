from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(
    os.environ.get(
        "EGMS_PUBLICATION_OUTPUT",
        REPO_ROOT / "outputs" / "publication_reproduction",
    )
).resolve()

EXPECTED_RASTERS = {
    "manuscript/figures/Figure_2_Study1_Baseline_B_vs_Structured_fusion.png": (3810, 2522),
    "manuscript/figures/Figure_3_Study2.png": (3810, 2472),
    "manuscript/figures/Figure_4_Study3.png": (3810, 2485),
    "power_full/figures/figure1_unpaired_power_curve.png": (3720, 2846),
}

EXPECTED_TABLE_ROWS = {
    "manuscript/tables/Table_2_main_effects.csv": 20,
    "supplement/compact/Tables/Table_S2_planning_summary.csv": 9,
    "supplement/compact/Tables/Table_S3_condensed_planning.csv": 7,
    "supplement/compact/Tables/Table_S4_validation_checks.csv": 33,
}

EXPECTED_FIGURE_STEMS = (
    "manuscript/figures/Figure_2_Study1_Baseline_B_vs_Structured_fusion",
    "manuscript/figures/Figure_3_Study2",
    "manuscript/figures/Figure_4_Study3",
    "power_full/figures/figure1_unpaired_power_curve",
    "power_full/figures/figure2_paired_correlation",
    "power_full/figures/figure3_cluster_sensitivity",
    "power_full/figures/figure4_event_rate_sensitivity",
    "power_full/figures/figure5_precision_curve",
    "power_full/figures/figure6_allocation_by_seed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReleaseArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not OUTPUT_ROOT.is_dir():
            raise AssertionError(
                f"Generated publication output is missing: {OUTPUT_ROOT}. "
                "Run `python run_publication.py --overwrite` first."
            )

    def test_numeric_input_boundary_contains_no_images_or_word_files(self) -> None:
        disallowed = {
            ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff",
            ".svg", ".pdf", ".doc", ".docx",
        }
        offenders = [
            path.relative_to(REPO_ROOT)
            for path in (REPO_ROOT / "data").rglob("*")
            if path.is_file() and path.suffix.lower() in disallowed
        ]
        self.assertEqual(
            offenders,
            [],
            f"Plotting inputs must be numeric, not image/Word files: {offenders}",
        )

        manifest = json.loads(
            (OUTPUT_ROOT / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        manifest_offenders = [
            relative
            for relative in manifest["inputs"]
            if Path(relative).suffix.lower() in disallowed
        ]
        self.assertEqual(manifest_offenders, [])

    def test_publication_plotters_do_not_read_word_media(self) -> None:
        source_root = REPO_ROOT / "src" / "egms_publication"
        offenders = []
        # validation.py deliberately names forbidden extensions so it can
        # reject them; only data-consuming plot/table builders are inspected.
        for name in ("figures.py", "study1_figure.py", "tables.py", "runner.py"):
            path = source_root / name
            text = path.read_text(encoding="utf-8").lower()
            if "word/media" in text or "word\\media" in text or ".docx" in text:
                offenders.append(path.name)
        self.assertEqual(
            offenders,
            [],
            f"Publication source refers to Word/raster extraction: {offenders}",
        )

    def test_key_figure_dimensions_and_dpi(self) -> None:
        for relative, dimensions in EXPECTED_RASTERS.items():
            with self.subTest(figure=relative):
                path = OUTPUT_ROOT / relative
                self.assertTrue(path.is_file() and path.stat().st_size > 1024)
                with Image.open(path) as image:
                    self.assertEqual(image.size, dimensions)
                    self.assertEqual(image.format, "PNG")
                    dpi = image.info.get("dpi")
                    self.assertIsNotNone(dpi)
                    assert dpi is not None
                    self.assertGreaterEqual(len(dpi), 2)
                    expected_dpi = 500.0 if "Figure_2_" in relative else 600.0
                    for value in dpi[:2]:
                        self.assertGreaterEqual(float(value), expected_dpi - 5.0)
                        self.assertLessEqual(float(value), expected_dpi + 5.0)

    def test_all_figure_pdf_and_svg_files_are_complete(self) -> None:
        for stem in EXPECTED_FIGURE_STEMS:
            with self.subTest(figure=stem):
                png = OUTPUT_ROOT / f"{stem}.png"
                pdf = OUTPUT_ROOT / f"{stem}.pdf"
                svg = OUTPUT_ROOT / f"{stem}.svg"
                self.assertTrue(png.is_file() and png.stat().st_size > 1024)

                payload = pdf.read_bytes()
                self.assertGreater(len(payload), 1024)
                self.assertTrue(payload.startswith(b"%PDF-"))
                self.assertIn(b"%%EOF", payload[-2048:])

                self.assertGreater(svg.stat().st_size, 1024)
                root = ET.parse(svg).getroot()
                self.assertEqual(root.tag.rsplit("}", 1)[-1], "svg")

    def test_publication_table_row_counts(self) -> None:
        for relative, rows in EXPECTED_TABLE_ROWS.items():
            with self.subTest(table=relative):
                table = pd.read_csv(OUTPUT_ROOT / relative, keep_default_na=False)
                self.assertEqual(len(table), rows)

    def test_table_s4_has_exactly_33_passing_checks(self) -> None:
        table = pd.read_csv(
            OUTPUT_ROOT / "supplement/compact/Tables/Table_S4_validation_checks.csv",
            keep_default_na=False,
        )
        self.assertEqual(table["ID"].tolist(), list(range(1, 34)))
        self.assertTrue(table["Status"].eq("Pass").all())
        self.assertTrue(table["Check"].str.strip().ne("").all())
        self.assertTrue(table["Evidence"].astype(str).str.strip().ne("").all())

    def test_generated_output_zip_excludes_temporary_files(self) -> None:
        with zipfile.ZipFile(OUTPUT_ROOT / "publication_outputs.zip") as archive:
            self.assertFalse(any(name.endswith(".tmp") for name in archive.namelist()))

    def test_artifact_manifest_matches_current_inputs_and_non_graphic_outputs(self) -> None:
        manifest = json.loads(
            (OUTPUT_ROOT / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        for relative, expected in manifest["inputs"].items():
            with self.subTest(input=relative):
                path = REPO_ROOT / relative
                self.assertTrue(path.is_file())
                self.assertEqual(path.stat().st_size, expected["bytes"])
                self.assertEqual(_sha256(path), expected["sha256"])
        for relative, expected in manifest["outputs"].items():
            # Raster/vector/PDF figure bytes can legitimately vary with the
            # Matplotlib/font backend. Their dimensions, DPI, and structural
            # completeness are tested above instead of relying on hashes.
            if Path(relative).suffix.lower() in {".png", ".pdf", ".svg"}:
                continue
            with self.subTest(output=relative):
                path = OUTPUT_ROOT / relative
                self.assertTrue(path.is_file())
                self.assertEqual(path.stat().st_size, expected["bytes"])
                self.assertEqual(_sha256(path), expected["sha256"])


if __name__ == "__main__":
    unittest.main()
