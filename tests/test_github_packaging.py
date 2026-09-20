from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.build_github_release import (
    ALLOWED_ASSET_FILES,
    ALLOWED_DATA_FILES,
    EXCLUDED_SUFFIXES,
    MAX_WEB_FILE_BYTES,
    MAX_WEB_FILES,
    ROOT,
    build,
    included_files,
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

    def test_release_is_curated_and_web_upload_sized(self) -> None:
        files = included_files()
        relatives = {path.relative_to(ROOT).as_posix() for path in files}
        self.assertEqual(relatives & {name for name in relatives if name.startswith("data/")}, ALLOWED_DATA_FILES)
        self.assertEqual(
            relatives & {name for name in relatives if name.startswith("publication_assets/")},
            ALLOWED_ASSET_FILES,
        )
        self.assertLessEqual(len(files) + 1, MAX_WEB_FILES)
        self.assertTrue(all(path.stat().st_size <= MAX_WEB_FILE_BYTES for path in files))
        self.assertFalse(any(name.startswith("outputs/") for name in relatives))
        self.assertFalse(any(name.endswith("_Executed.ipynb") for name in relatives))
        self.assertFalse(
            any(
                Path(name).suffix.lower() in EXCLUDED_SUFFIXES
                and name not in ALLOWED_ASSET_FILES
                for name in relatives
            )
        )
        self.assertFalse(any("studies23_frozen/data/" in name for name in relatives))

    def test_builder_writes_a_complete_verified_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="egms_source_release_test_") as temp_dir:
            destination = Path(temp_dir) / "github-release.zip"
            report = build(destination, overwrite=False)
            self.assertEqual(report["members"], len(included_files()) + 1)
            self.assertLess(report["members"], MAX_WEB_FILES + 1)
            with zipfile.ZipFile(destination) as archive:
                self.assertIsNone(archive.testzip())
                manifest_bytes = archive.read("RELEASE_MANIFEST.json")
                manifest = json.loads(manifest_bytes)
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
