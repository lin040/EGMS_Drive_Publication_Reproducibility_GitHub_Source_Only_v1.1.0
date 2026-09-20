#!/usr/bin/env python3
"""Build a deterministic GitHub source-and-revised-assets ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

try:
    from .build_complete_source_document import render_document
except ImportError:  # Direct execution: python tools/build_github_release.py
    from build_complete_source_document import render_document


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", ".mplconfig", ".pytest_cache", "__pycache__"}
EXCLUDED_NAMES = {
    ".DS_Store",
    "RELEASE_MANIFEST.json",
    "publication_outputs.zip",
    "COMPLETE_EVIDENCE_RELEASE_MANIFEST.json",
}
EXCLUDED_SUFFIXES = {
    ".pyc", ".pyo", ".tmp", ".bak", ".orig", ".swp",
    ".png", ".jpg", ".jpeg", ".pdf", ".svg", ".doc", ".docx",
    ".gz", ".npz", ".zip",
}
ALLOWED_DATA_FILES = {
    "data/study1_frozen/PROVENANCE.json",
    "data/study1_frozen/study1_metric_summary.csv",
    "data/study1_frozen/study1_paired_contrasts.csv",
    "data/study1_frozen/study1_replicate_effects.csv",
    "data/study1_frozen/provenance/model_architecture.csv",
    "data/study1_frozen/provenance/run_manifest.json",
    "data/study1_frozen/provenance/seed_registry.csv",
    "data/study1_frozen/provenance/study1_metric_summary_internal.csv",
    "data/study1_frozen/provenance/study1_paired_contrasts_internal.csv",
    "data/study1_frozen/provenance/study1_replicate_effects_internal.csv",
    "data/study1_frozen/provenance/validation_report.json",
    "data/studies23_frozen/tables/table_s2_primary_contrasts.csv",
    "data/studies23_frozen/tables/table_s3_by_regime.csv",
    "data/studies23_frozen/tables/table_s3_latency.csv",
    "data/studies23_frozen/tables/table_s3_main.csv",
    "data/studies23_frozen/tables/table_s3_negative_controls.csv",
    "data/studies23_frozen/tables/table_s3_primary_contrasts.csv",
}
ALLOWED_ASSET_FILES = {
    "publication_assets/manuscript/Figure_2_Study1_Baseline_B_vs_Structured_fusion.png",
    "publication_assets/manuscript/Figure_2_Study1_Baseline_B_vs_Structured_fusion.pdf",
    "publication_assets/manuscript/Figure_2_Study1_Baseline_B_vs_Structured_fusion.svg",
    "publication_assets/manuscript/Table_2_main_effects.csv",
    "publication_assets/manuscript/Table_2_main_effects.md",
    "publication_assets/manuscript/Table_2_main_effects.tex",
}
MAX_WEB_FILES = 100
MAX_WEB_FILE_BYTES = 25 * 1024 * 1024
REQUIRED_READABLE_SOURCE_FILES = (
    ROOT / "SOURCE_CODE_INDEX.md",
    ROOT / "COMPLETE_SOURCE_CODE.md",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def included_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        relative_text = relative.as_posix()
        if relative.parts and relative.parts[0] in {"outputs", "deliverables", "formal_run"}:
            continue
        if relative.parts and relative.parts[0] == "data" and relative_text not in ALLOWED_DATA_FILES:
            continue
        if relative.parts and relative.parts[0] == "publication_assets" and relative_text not in ALLOWED_ASSET_FILES:
            continue
        if relative_text.endswith("_Executed.ipynb"):
            continue
        if path.name in EXCLUDED_NAMES:
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES and relative_text not in ALLOWED_ASSET_FILES:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def validate_readable_source() -> None:
    """Refuse a release with missing or stale human-readable source views."""

    missing = [path for path in REQUIRED_READABLE_SOURCE_FILES if not path.is_file()]
    if missing:
        names = ", ".join(path.relative_to(ROOT).as_posix() for path in missing)
        raise FileNotFoundError(f"Missing readable-source release files: {names}")
    complete_source = ROOT / "COMPLETE_SOURCE_CODE.md"
    if complete_source.read_text(encoding="utf-8") != render_document():
        raise RuntimeError(
            "COMPLETE_SOURCE_CODE.md is stale; run "
            "python tools/build_complete_source_document.py before building the ZIP"
        )


def build(destination: Path, *, overwrite: bool) -> dict[str, object]:
    destination = destination.resolve()
    if destination.suffix.lower() != ".zip":
        raise ValueError("Release destination must end in .zip")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination exists: {destination}; pass --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)

    validate_readable_source()
    files = included_files()
    missing_data = sorted(ALLOWED_DATA_FILES - {path.relative_to(ROOT).as_posix() for path in files})
    if missing_data:
        raise FileNotFoundError(f"Missing required publication input(s): {missing_data}")
    if len(files) + 1 > MAX_WEB_FILES:
        raise RuntimeError(
            f"Source-only release has {len(files) + 1} files; GitHub web upload limit is {MAX_WEB_FILES}"
        )
    oversized = [
        path.relative_to(ROOT).as_posix()
        for path in files
        if path.stat().st_size > MAX_WEB_FILE_BYTES
    ]
    if oversized:
        raise RuntimeError(f"Files exceed the GitHub web-upload size target: {oversized}")
    manifest = {
        "schema": "egms-drive-github-release-manifest-2.1",
        "note": (
            "GitHub source-and-revised-assets package: no generated output workspace, "
            "executed notebook, raw prediction archive, checkpoint, or bootstrap-draw file. "
            "Reader-facing Study 1 inputs use Baseline B and Structured fusion only; archived "
            "machine identifiers occur solely in frozen source and provenance records. "
            "Full raw Study 1 outputs, splits, checkpoints, and SHA-256 evidence are "
            "distributed in the companion complete-evidence archive. The revised Figure 2 "
            "and 20-row Table 2 are included under publication_assets; Figures 3–4 and the "
            "Study 2–3/power inputs remain unchanged. "
            "Canonical Python/YAML is plain text and is also reproduced verbatim in "
            "COMPLETE_SOURCE_CODE.md."
        ),
        "files": {
            path.relative_to(ROOT).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in files
        },
    }
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    temporary = destination.with_name(destination.name + ".tmp")
    manifest_path = ROOT / "RELEASE_MANIFEST.json"
    manifest_temporary = ROOT / ".RELEASE_MANIFEST.json.tmp"
    temporary.unlink(missing_ok=True)
    manifest_temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                relative = path.relative_to(ROOT).as_posix()
                info = zipfile.ZipInfo(relative, date_time=(2026, 8, 26, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            info = zipfile.ZipInfo("RELEASE_MANIFEST.json", date_time=(2026, 8, 26, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, manifest_bytes, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        with zipfile.ZipFile(temporary) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise RuntimeError(f"ZIP CRC validation failed: {bad}")
            expected_members = {*manifest["files"], "RELEASE_MANIFEST.json"}
            observed_members = set(archive.namelist())
            if observed_members != expected_members:
                missing = sorted(expected_members - observed_members)
                extra = sorted(observed_members - expected_members)
                raise RuntimeError(
                    f"ZIP member-set validation failed; missing={missing}, extra={extra}"
                )
            for relative, record in manifest["files"].items():
                archived = archive.read(relative)
                if len(archived) != record["bytes"]:
                    raise RuntimeError(f"ZIP byte-count validation failed: {relative}")
                observed_sha256 = hashlib.sha256(archived).hexdigest()
                if observed_sha256 != record["sha256"]:
                    raise RuntimeError(f"ZIP SHA-256 validation failed: {relative}")
            if archive.read("RELEASE_MANIFEST.json") != manifest_bytes:
                raise RuntimeError("ZIP release manifest differs from the generated manifest")
        temporary.replace(destination)
        manifest_temporary.write_bytes(manifest_bytes)
        manifest_temporary.replace(manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)
    return {
        "zip": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": digest(destination),
        "members": len(files) + 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.output, overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
