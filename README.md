# EGMS-Drive Publication Reproducibility — Study 1 Revision

This v2.1 repository reproduces the figures and tables for the revised
EGMS-Drive manuscript. The principal change is Study 1; the controlled
synthetic Studies 2–3 source, frozen inputs, and numerical results are
unchanged from v1.1.

Reader-facing Study 1 artifacts use one comparison only:
**Baseline B versus Structured fusion**. The frozen evaluation source and
audit records retain an archived machine identifier for the modified
Structured fusion implementation so that source hashes and checkpoint
provenance remain verifiable. It is not a third method and never appears in
the manuscript figure or Table 2.

## What changed

- Replaced the unavailable exported dry-run Study 1 inputs with a complete,
  post-hoc exploratory controlled-synthetic evaluation.
- Used a method-blind generator, ten paired training-data/model-seed
  replicates, fixed validation/test/rollout tapes, 20 fitted checkpoints, and
  direction-neutral validation.
- Preserved frame labels, logits, probabilities, action/event records, split
  and seed manifests, interval algorithms, software versions, and SHA-256
  records in the companion complete-evidence archive.
- Rebuilt Figure 2 from validated numeric inputs and expanded Table 2 to all
  ten Study 1 endpoints.
- Removed the obsolete exported dry-run inputs and earlier Study 1 result
  snapshots from the public repository.

## Evidence boundary

Study 1 is post-hoc exploratory. Structured fusion was modified after the
earlier result, selected using development/validation data, and frozen before
the once-only final evaluation. Baseline B used 56 features, 4,012 fitted
parameters, and 50 epochs; Structured fusion used 145 features, 29,156 fitted
parameters, and 80 epochs. The comparison therefore evaluates the complete
implementations and does not isolate fusion from capacity or optimization.

Only Macro-F1, NLL, and Brier met the frozen support rule. ECE, collision,
near miss, critical event, route completion, TTC-P5, and jerk-P95 did not
establish a Structured fusion benefit. No CARLA server, public-dataset
benchmark, real-vehicle record, or empirical LLM output was used; rollout
endpoints are controlled-synthetic proxies, not deployment-safety evidence.

## Included

- Publication orchestration under `src/egms_publication/`.
- Complete frozen Study 1 implementation under its archived machine package,
  plus the public `run_study1.py` entry point.
- Reader-facing Study 1 inputs and provenance under `data/study1_frozen/`.
- Unchanged Studies 2–3 source and inputs under `src/egms_studies23/` and
  `data/studies23_frozen/`.
- Unchanged prospective power-analysis source and protocol.
- Tests, CI, Colab builder, deterministic release builder, license, and
  citation metadata.
- Ready-to-use revised Figure 2 and Table 2 under `publication_assets/`.

Bulky Study 1 raw outputs, checkpoints, split manifests, and bootstrap draws
are distributed separately in the companion archive named in
`data/study1_frozen/PROVENANCE.json`.

## Quick start

Python 3.11 or 3.12 is supported.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python run_publication.py --output outputs/reproduction
```

The command regenerates and validates:

| Manuscript item | Generated path |
|---|---|
| Figure 2 | `manuscript/figures/Figure_2_Study1_Baseline_B_vs_Structured_fusion.{png,pdf,svg}` |
| Figure 3 | `manuscript/figures/Figure_3_Study2.{png,pdf,svg}` |
| Figure 4 | `manuscript/figures/Figure_4_Study3.{png,pdf,svg}` |
| Table 2 | `manuscript/tables/Table_2_main_effects.{csv,md,tex}` |
| Figure S1 | `power_full/figures/figure1_unpaired_power_curve.{png,pdf,svg}` |
| Tables S2–S4 | `supplement/compact/Tables/` |

Every figure is drawn from structured numeric inputs. The pipeline never reads
a manuscript image, Word media, or screenshot.

## Study 1 audit and evidence replay

`run_study1.py` exposes the frozen implementation. The once-only final
evaluation is not rerun by the fast publication workflow. After extracting the
companion complete-evidence archive into a named directory under `outputs/`,
validate it with:

```bash
python run_study1.py validate --output outputs/COMPLETE_EVIDENCE_DIRECTORY
```

The validator checks completeness, source/input/checkpoint hashes, split and
seed separation, all 20 checkpoint replays, and raw-output recomputation. It
does not require Structured fusion to improve any endpoint.

## Optional Studies 2–3 refit

```bash
python run_studies.py run \
  --config configs/studies23.yaml \
  --output outputs/studies23_refit
```

These are lightweight controlled-synthetic mechanism surrogates, not the full
neural EGMS-Drive architecture.

## Validation

```bash
python -m compileall -q src tools tests run_publication.py run_study1.py run_study1r2.py run_studies.py
python tools/build_complete_source_document.py --check
python -m egms_power.cli validate --config configs/power_protocol.yaml
python run_publication.py --output outputs/qa
EGMS_PUBLICATION_OUTPUT="$PWD/outputs/qa" python -m unittest discover -s tests -v
```

The tests verify the ten Study 1 endpoints and public method names, exact
Figure 2 PNG hash, 20-row Table 2, Studies 2–3 consistency, output formats,
input boundaries, manifests, Colab reconstruction, and release packaging.

## GitHub package

The delivered ZIP is a source-and-publication-assets package. Extract it, then
commit and push the extracted files; GitHub does not expand a ZIP placed on the
Code page. See `GITHUB_UPLOAD_GUIDE.md`.

Rebuild the deterministic package with:

```bash
python tools/build_complete_source_document.py
python tools/build_colab.py
python tools/build_github_release.py \
  --output EGMS_Drive_Publication_Reproducibility_GitHub_v2.1.zip \
  --overwrite
```

`RELEASE_MANIFEST.json` records the byte count and SHA-256 of every archived
file.

## License and citation

Released under the MIT License. Cite `CITATION.cff` and preserve the evidence
boundary and post-hoc exploratory designation when reusing the artifacts.
