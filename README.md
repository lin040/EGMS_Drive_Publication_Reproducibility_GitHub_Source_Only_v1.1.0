# EGMS-Drive Publication Reproducibility — Source-Only Edition v1.1.0

**Release version:** `v1.1.0` (software version `1.1.0`; released 2026-08-16)

This repository regenerates the publication figures, tables, compact
supplement, and prospective power analysis for **EGMS-Drive** from readable
Python, YAML, and eight small canonical CSV inputs.

本版專為 GitHub Code 區整理：只保留完整原始碼、設定、測試、乾淨 Colab
與重建出版圖表不可缺少的小型數值輸入。它**不包含**預先生成的
PNG/PDF/SVG、結果表、output ZIP、已執行 Notebook，或大型 prediction/raw
results archive。

## What is included

- Complete source under `src/`, plus `run_publication.py` and `run_studies.py`.
- Two auditable YAML configurations under `configs/`.
- Eight canonical publication-input CSV files (about 42 KB total).
- A human-readable Colab notebook; every source/input file is shown in a
  separate `%%writefile` cell and SHA-256 verified. There is no encoded payload.
- Tests, GitHub Actions, documentation, license, citation metadata, and a
  deterministic release builder.
- `COMPLETE_SOURCE_CODE.md`, a generated reading copy of every canonical
  computational Python/YAML file.

## What is intentionally excluded

- The entire pre-generated `outputs/` tree.
- PNG, PDF, SVG, Word, and screenshot inputs or outputs.
- The executed notebook containing embedded result images/tables.
- Persisted Study 2–3 prediction frames, raw metrics, model/split manifests,
  and previous run reports.

The six small Study 2–3 summary tables and two Study 1 audit tables retained in
`data/` are **required numeric inputs**, not pre-generated deliverables. Without
them, Figures 2–4 and Table 2 cannot be reproduced in quick publication mode.

## Quick start

Python 3.11 or 3.12 is supported.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps

python run_publication.py --output outputs/reproduction
```

To replace an output directory previously created by this program:

```bash
python run_publication.py --output outputs/reproduction --overwrite
```

The runner creates all figures/tables, validates them, writes SHA-256 records,
and packages the newly generated results as
`outputs/reproduction/publication_outputs.zip`.

## Reproduced manuscript items

| Manuscript item | Generated path |
|---|---|
| Figure 2 | `manuscript/figures/Figure_2_Study1.{png,pdf,svg}` |
| Figure 3 | `manuscript/figures/Figure_3_Study2.{png,pdf,svg}` |
| Figure 4 | `manuscript/figures/Figure_4_Study3.{png,pdf,svg}` |
| Table 2 | `manuscript/tables/Table_2_main_effects.{csv,md,tex}` |
| Figure S1 | `power_full/figures/figure1_unpaired_power_curve.{png,pdf,svg}` |
| Tables S2–S4 | `supplement/compact/Tables/` |
| Supplement S3 inventory | `supplement/compact/S3_Digital_Reproducibility_Inventory.*` |

Every figure is drawn by the program from structured numeric inputs. The
pipeline never reads manuscript media, an existing figure, or a screenshot.

## Canonical input boundary

Publication mode reads only:

```text
configs/power_protocol.yaml
data/study1/study1_figure_inputs.csv
data/study1/study1_table2_inputs.csv
data/studies23_frozen/tables/table_s2_primary_contrasts.csv
data/studies23_frozen/tables/table_s3_by_regime.csv
data/studies23_frozen/tables/table_s3_latency.csv
data/studies23_frozen/tables/table_s3_main.csv
data/studies23_frozen/tables/table_s3_negative_controls.csv
data/studies23_frozen/tables/table_s3_primary_contrasts.csv
```

The power engine regenerates all prospective planning tables/figures from the
YAML protocol. The publication runner validates required files before running.

## Optional full controlled-synthetic refit

The complete Studies 2–3 source and frozen configuration remain included. A
new controlled-synthetic run can be generated from scratch:

```bash
python run_studies.py run \
  --config configs/studies23.yaml \
  --output outputs/studies23_refit
```

This source-only edition does not include the previous persisted raw-output
archive, so the old publication option `--reanalyze-controlled` has been
removed. Use the command above when a fresh full refit is required.

## Google Colab

Open `notebooks/EGMS_Drive_Publication_Reproduction_Colab.ipynb` and run all
cells. The notebook reconstructs the same readable source/config/CSV files,
SHA-256 verifies them, runs the publication pipeline, displays Figures 2–4,
Figure S1, Table 2, and Tables S2–S4, then offers the newly generated output
ZIP for download. It contains no saved execution outputs.

## Validation

```bash
python -m compileall -q src tools tests run_publication.py run_studies.py
python tools/build_complete_source_document.py --check
python -m egms_power.cli validate --config configs/power_protocol.yaml
python run_publication.py --output outputs/qa
EGMS_PUBLICATION_OUTPUT="$PWD/outputs/qa" python -m unittest discover -s tests -v
```

The checks cover source/input boundaries, numeric selections, figure
dimensions and formats, table row counts, the 33 passing Supplement checks,
manifest hashes, readable Colab reconstruction, and release packaging.

## GitHub upload

GitHub does not unpack a ZIP placed in the Code area. For a normal repository,
extract the delivered ZIP first, then commit/push its contents. This package
contains fewer than 100 files and no file near GitHub's 25 MiB browser-upload
limit, so it can also be uploaded through **Add file → Upload files** after
extraction. See `GITHUB_UPLOAD_GUIDE.md` for exact commands.

To rebuild the deterministic source-only ZIP:

```bash
python tools/build_complete_source_document.py
python tools/build_colab.py
python tools/build_github_release.py \
  --output EGMS_Drive_Publication_Reproducibility_GitHub_Source_Only_v1.1.0.zip \
  --overwrite
```

`RELEASE_MANIFEST.json` records the byte size and SHA-256 of every archived
file. The builder excludes all `outputs/`, binary result formats, temporary
files, executed notebooks, and bulky archived run data.

## Evidence boundary

1. Study 1 is an exported/audited summary reproduction. Its original training
   program, checkpoint, split manifest, raw frames, and full CI procedure are
   unavailable.
2. Studies 2–3 are controlled synthetic lightweight mechanism surrogates, not
   the complete EGMS-Drive neural architecture and not CARLA/public-dataset/
   real-vehicle evidence.
3. Collision-power outputs are prospective planning quantities conditional on
   assumed risks and dependence, not observed safety performance.

## License and citation

Released under the MIT License. Cite the software metadata in `CITATION.cff`
and retain the evidence-boundary statements when using generated artifacts.
