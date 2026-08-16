from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.build_github_release import (
    ALLOWED_DATA_FILES,
    EXCLUDED_PARTS,
    EXCLUDED_PART_PREFIXES,
    EXCLUDED_PART_SUFFIXES,
    EXCLUDED_SUFFIXES,
    MAX_WEB_FILE_BYTES,
    MAX_WEB_FILES,
    RELEASE_VERSION,
    ROOT,
    build,
    included_files,
    is_excluded_release_path,
)


class GitHubPackagingTests(unittest.TestCase):
    def test_required_repository_files_exist(self) -> None:
        for name in (
            ".gitattributes",
            ".github/workflows/ci.yml",
            "README.md",
            "GITHUB_UPLOAD_GUIDE.md",
            "SOURCE_CODE_INDEX.md",
            "COMPLETE_SOURCE_CODE.md",
            "LICENSE",
            "CITATION.cff",
        ):
            self.assertTrue((ROOT / name).is_file(), name)

    def test_release_is_source_only_and_web_upload_sized(self) -> None:
        files = included_files()
        relatives = {path.relative_to(ROOT).as_posix() for path in files}
        self.assertEqual(relatives & {name for name in relatives if name.startswith("data/")}, ALLOWED_DATA_FILES)
        self.assertLessEqual(len(files) + 1, MAX_WEB_FILES)
        self.assertTrue(all(path.stat().st_size <= MAX_WEB_FILE_BYTES for path in files))
        self.assertFalse(any(name.startswith("outputs/") for name in relatives))
        self.assertFalse(any(name.endswith("_Executed.ipynb") for name in relatives))
        self.assertFalse(any(Path(name).suffix.lower() in EXCLUDED_SUFFIXES for name in relatives))
        self.assertFalse(any("studies23_frozen/data/" in name for name in relatives))

    def test_local_environments_and_generated_metadata_are_always_excluded(self) -> None:
        for directory in (".venv", "venv", "env", "outputs", "build", "dist"):
            with self.subTest(directory=directory):
                self.assertIn(directory, EXCLUDED_PARTS)
                self.assertTrue(
                    is_excluded_release_path(Path(directory) / "nested" / "payload.py")
                )

        self.assertIn(".egg-info", EXCLUDED_PART_SUFFIXES)
        for relative in (
            Path("src/egms_drive_publication_reproducibility.egg-info/PKG-INFO"),
            Path("SRC/EGMS_DRIVE.EGG-INFO/requires.txt"),
            Path("tools/outputs/reproduction/table.csv"),
            Path("ENV/Lib/site-packages/example.py"),
        ):
            with self.subTest(relative=relative):
                self.assertTrue(is_excluded_release_path(relative))

        for prefix, relative in (
            (".mplconfig", Path(".mplconfig-ci/fontlist-v390.json")),
            (".mplconfig", Path("cache/.MPLCONFIG-Windows/fontlist.json")),
            (".coverage", Path(".coverage")),
            (".coverage", Path("reports/.coverage.worker-1")),
        ):
            with self.subTest(relative=relative):
                self.assertIn(prefix, EXCLUDED_PART_PREFIXES)
                self.assertTrue(is_excluded_release_path(relative))

        for relative in (
            Path("src/egms_publication/runner.py"),
            Path(".github/workflows/ci.yml"),
            Path("data/study1/study1_figure_inputs.csv"),
        ):
            with self.subTest(relative=relative):
                self.assertFalse(is_excluded_release_path(relative))

    def test_binary_and_serialized_result_formats_are_excluded(self) -> None:
        for suffix in (
            ".dll",
            ".exe",
            ".npy",
            ".npz",
            ".onnx",
            ".parquet",
            ".pkl",
            ".pt",
            ".so",
            ".whl",
            ".zip",
        ):
            with self.subTest(suffix=suffix):
                self.assertIn(suffix, EXCLUDED_SUFFIXES)
                self.assertTrue(is_excluded_release_path(Path(f"artifact{suffix}")))

    def test_ci_and_release_version_remain_present(self) -> None:
        self.assertTrue((ROOT / ".github/workflows/ci.yml").is_file())
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".mplconfig*/", gitignore)
        self.assertIn(".coverage*", gitignore)
        self.assertEqual(RELEASE_VERSION, "1.1.0")
        self.assertIn(
            'version = "1.1.0"',
            (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        )
        self.assertIn(
            "__version__ = \"1.1.0\"",
            (ROOT / "src/egms_publication/__init__.py").read_text(encoding="utf-8"),
        )
        self.assertIn(
            "version: 1.1.0",
            (ROOT / "CITATION.cff").read_text(encoding="utf-8"),
        )

    def test_builder_writes_a_complete_verified_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="egms_source_release_test_") as temp_dir:
            destination = Path(temp_dir) / "source-only.zip"
            report = build(destination, overwrite=False)
            self.assertEqual(report["members"], len(included_files()) + 1)
            self.assertLess(report["members"], MAX_WEB_FILES + 1)
            with zipfile.ZipFile(destination) as archive:
                self.assertIsNone(archive.testzip())
                manifest_bytes = archive.read("RELEASE_MANIFEST.json")
                manifest = json.loads(manifest_bytes)
                self.assertEqual(manifest["release_version"], RELEASE_VERSION)
                self.assertEqual(
                    set(archive.namelist()),
                    set(manifest["files"]) | {"RELEASE_MANIFEST.json"},
                )
                for relative, expected in manifest["files"].items():
                    payload = archive.read(relative)
                    self.assertEqual(len(payload), expected["bytes"], relative)
                    self.assertEqual(
                        hashlib.sha256(payload).hexdigest(),
                        expected["sha256"],
                        relative,
                    )


if __name__ == "__main__":
    unittest.main()
