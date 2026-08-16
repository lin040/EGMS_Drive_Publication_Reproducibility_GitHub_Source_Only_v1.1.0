# Analysis plan frozen before the final full run

This is a local analysis-plan freeze, **not** a preregistration claim. It was
created after pilot/smoke QA and before executing the final training-replicate
seed block listed in `configs/studies23.yaml`.

## Evidence label

`CONTROLLED_SYNTHETIC_MECHANISM_VALIDATION_NOT_EMPIRICAL_DATASET`

CARLA, nuScenes, RADIATE, and Argoverse 2 rows must remain N/A unless their raw
files, manifests, checkpoints, predictions, and official evaluator outputs are
actually present in a separate run.

## Frozen decision rules

- Execute all five conditions in each study for all ten final training-sample
  replicates. Do not select replicates or alter DGM/model parameters by result.
- Within a replicate, every condition receives the identical training data;
  validation and test scenes remain fixed across replicates.
- Study 2 primary contrasts: adverse NLL for reliability weighting; SAS for
  alignment; adverse held-out one-step feature prediction error for temporal
  consistency; mean scene-level missing-modality macro-F1 drop for the
  dropout-distillation package.
- Study 3 primary contrasts: history-dependent minFDE_6 for graph ablations;
  Brier-minFDE_6 for predicted-intent gating; MR_1 for independently trained
  K=1 versus Full top-1.
- AV2-style aggregation selects the mode with minimum FDE and uses that same
  mode for minADE and Brier-minFDE. MR uses a 2 m endpoint threshold.
- Scene-decomposable primary intervals use 10,000 paired crossed-bootstrap
  draws over training replicate and test scene. SAS uses a paired replicate
  interval conditional on its fixed retrieval gallery. P values use all 1,024
  paired sign flips across ten replicate-level effects, followed by Holm
  correction within each study.
- `mechanism_support_rule_met` requires the directional CI rule, at least 8/10
  favorable replicate effects, and Holm-adjusted p < .05.
- Full is not required to win. Unsupported, null, or adverse effects remain in
  every exported table and narrative.
