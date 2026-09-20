# EGMS-Drive source and asset index

This v2.1 release contains the complete computational source used to rebuild
the revised manuscript graphics and tables. Study 1 is updated; Studies 2–3
and the prospective power analysis retain their v1.1 numerical inputs and
implementations.

## Entry points

| Path | Purpose |
|---|---|
| `run_publication.py` | Regenerate Figures 2–4, Figure S1, Table 2, Tables S2–S4, validation reports, and the generated-output ZIP. |
| `run_study1.py` | Public entry point for validation/audit of the frozen Study 1 implementation. |
| `run_study1r2.py` | Byte-frozen internal entry point retained only for the final source manifest. |
| `run_studies.py` | Run, analyze, or validate a new controlled-synthetic Studies 2–3 workspace. |
| `tools/build_colab.py` | Rebuild the readable self-contained publication notebook. |
| `tools/build_complete_source_document.py` | Rebuild `COMPLETE_SOURCE_CODE.md`. |
| `tools/build_github_release.py` | Build and verify the deterministic GitHub ZIP. |

## Study 1 publication layer

| Path | Responsibility |
|---|---|
| `src/egms_publication/study1_figure.py` | Figure 2 from the two public method names and validated numeric inputs. |
| `src/egms_publication/tables.py` | All ten Study 1 rows plus unchanged Study 2–3 rows in Table 2. |
| `data/study1_frozen/study1_metric_summary.csv` | Reader-facing Baseline B and Structured fusion method estimates. |
| `data/study1_frozen/study1_paired_contrasts.csv` | Ten paired effects, crossed-bootstrap intervals, directionality, sign-flip tests, and Holm-adjusted results. |
| `data/study1_frozen/study1_replicate_effects.csv` | One row per training replicate and endpoint using public method column names. |
| `data/study1_frozen/PROVENANCE.json` | Hash mapping, evidence boundary, run counts, internal/public naming rule, and companion evidence archive. |
| `docs/STUDY1_PUBLICATION_MAP.md` | Concise map from validated inputs to Figure 2 and Table 2. |

## Frozen Study 1 implementation

The complete final implementation remains under `src/egms_study1r2/` with its
archived machine identifier because these exact paths and bytes are recorded
in the freeze and run manifests. Reader-facing outputs never use that label.

| Module | Responsibility |
|---|---|
| `common.py` | Protocol loading, constants, hashing, scenario cells, and schema helpers. |
| `generator.py` | Method-blind episode/frame generator shared by both fitted pipelines. |
| `models.py` | Baseline B and Structured fusion encoders and portable deterministic MLP checkpoints. |
| `training.py` | Paired training, fixed-validation selection, calibration, logs, and checkpoint serialization. |
| `rollout.py` | Paired fixed-tape action/event simulation and episode derivation. |
| `statistics.py` | Raw-output metrics, crossed paired bootstrap CIs, exact sign-flip tests, and Holm adjustment. |
| `reporting.py` | Frozen internal analysis tables and direction-neutral figures. |
| `validation.py` | Completeness, hashes, checkpoint replay, raw recomputation, and direction-neutral validation. |
| `runner.py` | Development, freeze, once-only final evaluation, and validation orchestration. |

`src/egms_study1r/generator.py` is a byte-identical compatibility copy required
by the freeze manifest. The remaining files in that namespace are thin public
compatibility shims; they do not define another model or result version.

## Publication, Studies 2–3, and power

| Path | Responsibility |
|---|---|
| `src/egms_publication/figures.py` | Programmatic Figures 3–4 from unchanged numeric inputs. |
| `src/egms_publication/runner.py` | Safe output handling and complete publication orchestration. |
| `src/egms_publication/validation.py` | Input boundaries, dimensions, formats, row counts, and manifests. |
| `src/egms_studies23/` | Unchanged controlled-synthetic Study 2–3 generators, fitted surrogates, statistics, plots, and validators. |
| `data/studies23_frozen/tables/` | Unchanged six small Study 2–3 publication input tables. |
| `src/egms_power/` | Unchanged prospective power, sensitivity, precision, allocation, and validation code. |
| `configs/power_protocol.yaml` | Unchanged prospective collision-power assumptions. |

## Ready-to-use manuscript assets

| Path | Contents |
|---|---|
| `publication_assets/manuscript/Figure_2_Study1_Baseline_B_vs_Structured_fusion.*` | Revised Figure 2 in PNG, PDF, and SVG. |
| `publication_assets/manuscript/Table_2_main_effects.*` | Revised 20-row Table 2 in CSV, Markdown, and LaTeX. |

Figures 3–4, Figure S1, and Tables S2–S4 are regenerated unchanged by
`run_publication.py`; they are not duplicated in the repository asset folder.

## Audit views

- `COMPLETE_SOURCE_CODE.md` reproduces every canonical Python/YAML file
  verbatim with its SHA-256.
- `notebooks/EGMS_Drive_Publication_Reproduction_Colab.ipynb` presents the
  fast publication source and small inputs as readable `%%writefile` cells.
- `RELEASE_MANIFEST.json` is regenerated with the ZIP and verifies every
  released file.

No legacy `data/study1/`, obsolete exported dry-run input, earlier adverse
Study 1 publication input, generated checkpoint, or temporary file is included
in this repository package.
