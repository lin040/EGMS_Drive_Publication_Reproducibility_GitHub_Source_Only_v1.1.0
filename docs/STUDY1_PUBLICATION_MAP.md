# Study 1 publication map

The manuscript and all reader-facing artifacts use exactly two method names:
**Baseline B** and **Structured fusion**. The archived machine identifier in
the frozen evaluation source denotes the same modified Structured fusion arm;
it is retained only so the source hashes, manifests, checkpoints, and raw
records remain auditable. It is not a third experimental method.

## Canonical manuscript outputs

| Manuscript item | Generated path |
|---|---|
| Figure 2 | `manuscript/figures/Figure_2_Study1_Baseline_B_vs_Structured_fusion.*` |
| Table 2 | `manuscript/tables/Table_2_main_effects.*` |

Figure 2 and the first ten rows of Table 2 are derived from
`data/study1_frozen/study1_metric_summary.csv` and
`data/study1_frozen/study1_paired_contrasts.csv`. The older exported dry-run
inputs and the earlier adverse Study 1 refit inputs are intentionally absent
from this repository.

## Evidence boundary

Study 1 is a post-hoc exploratory controlled-synthetic comparison. Structured
fusion was modified after the earlier result, selected on development and
validation data, and frozen before the once-only final evaluation. Baseline B
and Structured fusion were evaluated across ten paired training replicates on
fixed test and rollout tapes. The implementation is capacity- and
optimization-asymmetric, so the result does not isolate fusion alone.

The repository contains the complete frozen source and small validated
publication inputs. Bulky raw frame/action/event records, checkpoints, split
manifests, and bootstrap draws remain in the companion complete-evidence
archive named in `data/study1_frozen/PROVENANCE.json`.
