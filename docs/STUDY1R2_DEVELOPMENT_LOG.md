# Study 1-R2 exploratory development log

Status: development candidates declared before final R2 evaluation.

The original Study 1-R source, protocol, outputs, and release artifacts remain
unchanged. Study 1-R2 changes the candidate model only; the controlled-synthetic
generator remains method-blind and both methods receive identical observations,
labels, training episodes, validation episodes, and paired rollout tapes.

## Seed firewall

| Stage | Purpose | Seeds |
|---|---|---|
| Development fit | five candidate-selection replicates | data 5129, 5171, 5227, 5273, 5323; model 6121, 6173, 6229, 6271, 6323 |
| Development validation | architecture and hyperparameter selection | 21013 |
| Development rollout | safety-constraint check | 21019 |
| Final fit | ten independent paired training replicates | data 8101, 8111, 8117, 8123, 8147, 8161, 8171, 8191, 8209, 8219; model 9103, 9127, 9133, 9151, 9173, 9181, 9199, 9209, 9221, 9239 |
| Final checkpoint validation | epoch selection and temperature calibration | 31013 |
| Final offline test | once-only fixed test | 31019 |
| Final rollout | once-only fixed tape | 31033 |

No final-fit or final-evaluation seed is permitted in the `develop` command.

## Declared candidates

All candidates use the same 145-feature dual-path encoder: the complete
Baseline-B raw quality-gated path, the original compact structured path,
inverse-observation-noise fusion, temporal residuals, TTC and stopping-margin
features, and observation-only distances to the published oracle thresholds.

| Candidate | Hidden layers | Epochs | Learning rate | Alpha |
|---|---:|---:|---:|---:|
| r2_c01 | 64, 32 | 60 | 0.0020 | 0.0001 |
| r2_c02 | 96, 48 | 70 | 0.0015 | 0.0005 |
| r2_c03 | 128, 64, 32 | 80 | 0.0010 | 0.0010 |
| r2_c04 | 96, 48 | 90 | 0.0010 | 0.0001 |

Selection is the lowest mean development-validation NLL among candidates whose
mean development-rollout critical-event rate is not above Baseline B. Ties are
resolved by higher mean Macro-F1, lower parameter count, then lexical candidate
ID. Every candidate and replicate is retained in the development ledger.

Selected candidate: **r2_c03** ([128, 64, 32], 80 epochs, learning rate
0.0010, alpha 0.0010; 29,156 fitted parameters). Across the five development
replicates, its mean validation NLL was 0.371958 versus 0.394760 for Baseline B,
its mean Macro-F1 was 0.809831 versus 0.782812, and its mean rollout
critical-event rate was 0.241111 versus 0.250556. The complete ledger, including
all non-selected candidates, has SHA-256
`c756ef260f1eb4b97a6a94c38e34301238eff0a2100578103b80abb7711bfc89`.
