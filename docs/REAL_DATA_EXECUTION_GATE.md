# Real-data execution gate

The controlled generator and its outputs are not substitutes for named-dataset
experiments. A row may change from `N/A` only after all items below are present
in a new, separately named run directory.

## Common gate

- Raw-file manifest with nonzero counts and SHA-256 hashes.
- Official split identifiers and a frozen preprocessing configuration.
- Independently trained checkpoints for every ablation under the same budget.
- Saved per-sample predictions and a model/config hash for every checkpoint.
- At least three independent training seeds for an empirical replication.
- Scene/log-cluster confidence intervals and all prespecified negative results.

## Study 2 source-specific gate

- **CARLA:** synchronous fixed-step collection; sensor callback payloads keyed by
  frame/timestamp. The recorder alone does not contain raw sensor payloads.
- **nuScenes:** timestamp and ego-motion alignment of raw streams. The dataset
  does not supply KEEP/SLOW/YIELD/STOP ground truth; any operational labels must
  be derived, disclosed, manually checked, and sensitivity-tested.
- **RADIATE:** use its object annotations for a declared object task. Do not
  present action calibration as RADIATE ground truth, and disclose any modality
  adapter because its scanning radar differs from sparse automotive radar.

## Study 3 source-specific gate

- **Argoverse 2 Motion:** use the official 5 s history/6 s future protocol and
  official evaluator output. AV2 Motion has tracks/maps but no intent labels;
  any maneuver class is a separately declared pseudo-label or manual label.
- **CARLA:** scripted actor intent may be used as simulator ground truth only
  when the script and event definitions are saved before evaluation.

Named-source results must remain separate because the sensors, labels, tasks,
and radar representations are not interchangeable.
