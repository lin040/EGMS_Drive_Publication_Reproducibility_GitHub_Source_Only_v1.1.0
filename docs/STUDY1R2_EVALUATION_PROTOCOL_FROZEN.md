# Study 1-R2 evaluation protocol

Protocol status after candidate selection:
`FROZEN_AFTER_EXPLORATORY_DEVELOPMENT_BEFORE_FINAL_EVALUATION`.

Study 1-R2 is post-hoc and not preregistered. It evaluates the single candidate
selected by the separately hashed development ledger. The original frozen
Study 1-R remains the primary structured-versus-baseline result.

## Final evaluation contract

- Generator: the unchanged method-blind controlled-synthetic generator.
- Comparator: Baseline B with the original 56-feature encoder, [48, 24] head,
  and original optimization budget.
- Candidate: r2_c03, the development-selected 145-feature Structured-R2
  architecture with hidden layers [128, 64, 32], 80 epochs, learning rate
  0.0010 and alpha 0.0010; no candidate substitution after freezing.
- Training: ten independent training-data and model-initialization replicates.
- Validation: one fixed set used for epoch selection and temperature scaling.
- Offline test: one new fixed 360-episode set shared by all replicates.
- Closed loop: one new fixed 360-episode exogenous scenario tape shared by all
  replicates and both methods.
- Raw records: frame labels, logits, calibrated probabilities, actions, state,
  events, and per-episode metrics.
- Statistics: paired crossed-cluster percentile 95% confidence intervals with
  10,000 draws; exact 1,024-pattern replicate sign-flip tests; Holm correction
  over the ten endpoints.
- Direction: all endpoints and all replicates are reported; no pass criterion
  requires Structured-R2 to improve.
- Integrity: split manifests, resolved configuration, software versions,
  deterministic JSON checkpoints, source hashes, artifact hashes, and replay
  checks are mandatory.

Final evaluation is once-only. If architecture, hyperparameters, policy, or
seeds are changed after final results are viewed, that work must receive a new
study identifier and entirely new final-evaluation seeds.
