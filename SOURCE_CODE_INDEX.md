# EGMS-Drive Source Code Index — v1.1.0

This source-only release contains the complete executable publication, power,
and controlled-synthetic Studies 2–3 code. Generated results and persisted raw
run archives are intentionally excluded.

## Entry points

| Path | Purpose |
|---|---|
| `run_publication.py` | Regenerate Figures 2–4, Figure S1, Table 2, Tables S2–S4, validation reports, and a generated output ZIP. |
| `run_studies.py` | Run, analyze, or validate a newly generated controlled-synthetic Studies 2–3 workspace. |
| `src/egms_publication/runner.py` | Safe output handling and publication orchestration. |
| `src/egms_power/cli.py` | Prospective power-protocol validation and execution. |
| `tools/build_colab.py` | Rebuild the readable source-only Colab. |
| `tools/build_complete_source_document.py` | Rebuild `COMPLETE_SOURCE_CODE.md`. |
| `tools/build_github_release.py` | Build and verify the deterministic source-only GitHub ZIP. |

## Publication package

| Module | Responsibility |
|---|---|
| `src/egms_publication/figures.py` | Programmatic Figures 2–4 from numeric CSV inputs. |
| `src/egms_publication/tables.py` | Main Table 2 and compact Tables S2–S4. |
| `src/egms_publication/style.py` | Shared STIX styling and PNG/PDF/SVG export validation. |
| `src/egms_publication/validation.py` | Input boundary, dimensions, formats, row counts, and SHA-256 checks. |
| `src/egms_publication/utils.py` | Table/JSON serialization, row selection, and hashing helpers. |

## Power package

| Module | Responsibility |
|---|---|
| `src/egms_power/config.py` | Load and validate the prospective protocol. |
| `src/egms_power/statistics.py` | Proportion power, McNemar, clustering, precision, and allocation calculations. |
| `src/egms_power/pipeline.py` | Six figures, eleven detailed tables, summaries, and 33 validation checks. |
| `src/egms_power/cli.py` | `validate` and `run` commands. |

## Controlled synthetic Studies 2–3 package

| Module | Responsibility |
|---|---|
| `src/egms_studies23/study2.py` | Reliability, alignment, temporal, and missing-modality operational surrogates. |
| `src/egms_studies23/study3.py` | Graph, intent, multimodal trajectory, and diagnostic-latency surrogates. |
| `src/egms_studies23/reporting.py` | Seed/scene summaries, contrasts, bootstrap intervals, and tables. |
| `src/egms_studies23/plots.py` | Diagnostic plotting from newly generated numeric outputs. |
| `src/egms_studies23/validation.py` | Metric, split, model, raw-output, table, and provenance checks for a newly generated run. |
| `src/egms_studies23/runner.py` | `run`, `analyze`, and `validate` workflow. |
| `src/egms_studies23/common.py` | Seeds, metrics, calibration, adjustment, and manifest helpers. |

## Configurations and required publication inputs

| Path | Role |
|---|---|
| `configs/power_protocol.yaml` | Prospective collision-power assumptions. |
| `configs/studies23.yaml` | From-scratch controlled-synthetic Studies 2–3 configuration. |
| `data/study1/study1_figure_inputs.csv` | Audited Study 1 Figure 2 snapshot. |
| `data/study1/study1_table2_inputs.csv` | Audited Study 1 Table 2 rows. |
| `data/studies23_frozen/tables/table_s2_primary_contrasts.csv` | Figure 3 and Table 2 Study 2 contrasts. |
| `data/studies23_frozen/tables/table_s3_by_regime.csv` | Figure 4 graph-regime panel. |
| `data/studies23_frozen/tables/table_s3_latency.csv` | Figure 4 latency panel. |
| `data/studies23_frozen/tables/table_s3_main.csv` | Figure 4 intent and trajectory panels. |
| `data/studies23_frozen/tables/table_s3_negative_controls.csv` | Figure 4 negative-control text and Table 2. |
| `data/studies23_frozen/tables/table_s3_primary_contrasts.csv` | Figure 4/Table 2 primary contrasts. |

## Reading and auditing

- `COMPLETE_SOURCE_CODE.md` reproduces every canonical Python/YAML file
  verbatim with its SHA-256.
- `notebooks/EGMS_Drive_Publication_Reproduction_Colab.ipynb` presents the same
  source and small inputs as readable `%%writefile` cells.
- `RELEASE_MANIFEST.json` is regenerated with the ZIP and verifies every
  released file.

No CSV/JSON/NPZ/GZ archive containing previous raw prediction results is
included. The publication-input CSVs listed above are the smallest verified
set needed to reproduce the manuscript assets.
