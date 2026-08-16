from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.build_colab import CANONICAL_PATHS


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "EGMS_Drive_Publication_Reproduction_Colab.ipynb"
FORBIDDEN = {
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


def _cell_source(cell: dict[str, object]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _parse_writefile(source: str) -> tuple[str, str]:
    first_line, body = source.split("\n", 1)
    prefix = "%%writefile "
    if not first_line.startswith(prefix):
        raise AssertionError("Not a %%writefile cell")
    relative = first_line[len(prefix) :].strip()
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise AssertionError(f"Unsafe %%writefile target: {relative!r}")
    return relative, body


def _literal_assignment(source: str, name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Assignment not found: {name}")


class ColabReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.sources = [_cell_source(cell) for cell in cls.notebook["cells"]]
        cls.writefile_sources = [source for source in cls.sources if source.startswith("%%writefile ")]
        cls.writefiles = dict(_parse_writefile(source) for source in cls.writefile_sources)
        cls.verify_cell = next(source for source in cls.sources if "EXPECTED_HASHES =" in source)

    def test_notebook_schema_and_required_outputs(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(self.notebook["metadata"]["kernelspec"]["name"], "python3")
        combined = "\n".join(self.sources)
        for label in (
            "Figure 2",
            "Figure 3",
            "Figure 4",
            "Figure S1",
            "Table 2",
            "Table S2",
            "Table S3",
            "Table S4",
        ):
            self.assertIn(label, combined)
        self.assertIn("files.download", combined)
        self.assertIn("RUN_FULL_CONTROLLED_REFIT = False", combined)
        self.assertIn('"sklearn": "scikit-learn>=1.4,<2"', combined)
        self.assertIn('"run_studies.py",\n            "run",', combined)

    def test_setup_cell_can_install_missing_or_incompatible_dependencies(self):
        setup_cell = next(
            source
            for source in self.sources
            if "requirements = {" in source and "subprocess.run(" in source
        )
        tree = ast.parse(setup_cell)
        imported_names = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertIn("subprocess", imported_names)
        self.assertIn('"numpy": "numpy>=2.0,<3"', setup_cell)
        self.assertIn('"pandas": "pandas>=2.2,<3"', setup_cell)
        self.assertIn('"scipy": "scipy>=1.12,<2"', setup_cell)
        self.assertIn("requirement.specifier.contains", setup_cell)

    def test_each_canonical_file_is_complete_readable_text(self):
        self.assertEqual(len(self.writefile_sources), len(self.writefiles), "Duplicate %%writefile target")
        self.assertEqual(tuple(self.writefiles), CANONICAL_PATHS)
        self.assertGreaterEqual(len(self.writefiles), 30)
        for relative, body in self.writefiles.items():
            self.assertNotIn(Path(relative).suffix.lower(), FORBIDDEN)
            local = (ROOT / relative).read_bytes()
            self.assertEqual(body.encode("utf-8"), local, relative)
            if relative.endswith(".py"):
                ast.parse(body, filename=relative)

    def test_sha256_verification_matches_readable_cells(self):
        hashes = _literal_assignment(self.verify_cell, "EXPECTED_HASHES")
        self.assertEqual(set(hashes), set(self.writefiles))
        for relative, body in self.writefiles.items():
            observed = hashlib.sha256(body.encode("utf-8")).hexdigest()
            self.assertEqual(observed, hashes[relative], relative)

    def test_no_encoded_or_compressed_payload_is_embedded(self):
        combined_lower = "\n".join(self.sources).lower()
        for forbidden_text in (
            "import base64",
            "from base64",
            "b64decode",
            "b64encode",
            "import gzip",
            "from gzip",
            "gzip.decompress",
            "gzip.compress",
            "bytes.fromhex",
        ):
            self.assertNotIn(forbidden_text, combined_lower)
        self.assertFalse(any(relative.startswith("outputs/") for relative in self.writefiles))

    def test_standard_python_writefile_simulation_reconstructs_sources(self):
        with tempfile.TemporaryDirectory(prefix="egms_colab_source_test_") as temp_dir:
            reconstructed = Path(temp_dir)
            for relative, body in self.writefiles.items():
                destination = reconstructed / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(body, encoding="utf-8", newline="")
            for relative in CANONICAL_PATHS:
                self.assertEqual(
                    (reconstructed / relative).read_bytes(),
                    (ROOT / relative).read_bytes(),
                    relative,
                )


if __name__ == "__main__":
    unittest.main()
