from __future__ import annotations

from pathlib import Path
import unittest

from tools.build_complete_source_document import (
    ROOT,
    canonical_source_files,
    render_document,
    sha256_bytes,
)


class CompleteSourceDocumentTests(unittest.TestCase):
    def test_complete_source_document_is_current(self) -> None:
        generated = ROOT / "COMPLETE_SOURCE_CODE.md"
        self.assertTrue(generated.is_file())
        self.assertEqual(generated.read_text(encoding="utf-8"), render_document())

    def test_contains_verbatim_canonical_source(self) -> None:
        document = render_document()
        files = canonical_source_files()
        self.assertTrue(files)
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            content = data.decode("utf-8").rstrip("\n")
            self.assertIn(f"### `{relative}`", document)
            self.assertIn(f"- SHA-256: `{sha256_bytes(data)}`", document)
            self.assertIn(content, document)

    def test_has_no_encoded_transport_payload(self) -> None:
        document = render_document().lower()
        self.assertNotIn("base64", document)
        self.assertNotIn("gzip", document)
        self.assertNotIn(".ipynb", document)
        self.assertNotIn("payload = {'", document)

    def test_inventory_is_limited_to_python_and_yaml(self) -> None:
        files = canonical_source_files()
        self.assertTrue(
            all(path.suffix.lower() in {".py", ".yaml", ".yml"} for path in files)
        )
        self.assertTrue(
            all(Path("outputs") not in path.relative_to(ROOT).parents for path in files)
        )


if __name__ == "__main__":
    unittest.main()
