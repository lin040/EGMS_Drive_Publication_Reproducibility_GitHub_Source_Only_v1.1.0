# Statistical Analysis Plan (Protocol v1.0)

> All results in this repository concern study design. They are not empirical EGMS-Drive performance estimates.

## 1. Aim and confirmatory question

The confirmatory planning question is whether a future implemented candidate system and one prespecified aligned/no-graph comparator differ in episode-level collision probability. One scenario block is evaluated under both methods, so the intended empirical analysis is paired. An independent-arm calculation is retained as a transparent reference because no empirical cross-method joint distribution currently exists.

The primary estimand is

\[
\Delta=p_{\mathrm{candidate}}-p_{\mathrm{control}}.
\]

The central planning assumptions are \(p_{\mathrm{control}}=0.06\) and \(p_{\mathrm{candidate}}=0.03\). They define a minimally relevant design scenario; they are not estimates from the former surrogate generator.

## 2. Primary endpoint and statistical unit

- Endpoint: whether at least one physical collision occurs during a prespecified evaluation episode.
- Statistical unit: a complete scenario–environment–density–seed episode block, not a frame or simulator tick.
- Pair ID: the same scenario, map, environment, density, actor initialization, route, and scenario seed evaluated under each compared method.
- Cluster ID: `scenario_family × scenario_seed`; the nine environment-density episodes within that ID may be correlated.
- Training seed: recorded separately from scenario and sensor seeds. Increasing rollouts under one trained checkpoint does not substitute for independent training seeds.

## 3. Independent reference calculation

For equal allocation, a two-sided test, and a pooled-null/unpooled-alternative normal approximation,

\[
n=\frac{\left[z_{1-\alpha/2}\sqrt{2\bar p(1-\bar p)}+z_{1-\beta}\sqrt{p_C(1-p_C)+p_E(1-p_E)}\right]^2}{(p_C-p_E)^2},
\quad \bar p=(p_C+p_E)/2.
\]

At \(\alpha=.05\), power \(=.80\), \(p_C=.06\), and \(p_E=.03\), the raw value is 748.388, or 749 episodes per arm. This is an independence-based benchmark, not a universally sufficient design.

## 4. Paired primary analysis

The paired design requires the four joint probabilities:

\[
\pi_{11}=p_Cp_E+r\sqrt{p_C(1-p_C)p_E(1-p_E)},
\]

\[
\pi_{10}=p_C-\pi_{11},\qquad
\pi_{01}=p_E-\pi_{11},\qquad
\pi_{00}=1-\pi_{11}-\pi_{10}-\pi_{01}.
\]

The program rejects infeasible correlations using the Fréchet bounds. The exact two-sided McNemar test conditions on \(D=N_{10}+N_{01}\); when \(D=0\), its p-value is one. Power is obtained both by exact unconditional enumeration and by repeated multinomial complete-trial simulation.

The final empirical sample size will be updated once a blinded pilot estimates \(\pi_{10}\), \(\pi_{01}\), and the cluster ICC. The update may use only pooled outcome structure, not method labels or the observed direction of effect.

## 5. Cluster and incomplete-episode allowance

The planning approximation for equal cluster size \(m\) is

\[
DE=1+(m-1)\rho_{\mathrm{ICC}}.
\]

For unequal cluster sizes with coefficient of variation \(CV\), the sensitivity approximation is

\[
DE\approx1+\left[(1+CV^2)m-1\right]\rho_{\mathrm{ICC}}.
\]

The canonical provisional allocation uses \(m=9\), ICC=.05, CV=0, a 5% invalid-episode allowance, and complete blocks of 45 episodes. These assumptions are varied in the output tables; they are not measured properties of CARLA or EGMS-Drive.

The formal empirical analysis must use a cluster-aware paired model or cluster bootstrap. A naive row-level bootstrap is prohibited.

## 6. Multiplicity

Only candidate versus aligned/no-graph collision risk is confirmatory. Comparisons with additional baselines are secondary and Holm-adjusted. Four ablation comparisons are not co-primary collision tests. If four collision contrasts were planned as co-primary, the conservative sample-size bound uses alpha=.0125.

## 7. Ablation endpoints

| Independently retrained ablation | Mechanism-specific primary endpoints | Collision role |
|---|---|---|
| No cross-modal alignment | macro-F1, ECE, NLL | Separately powered safety endpoint |
| No temporal consistency | temporal flip rate, minADE, minFDE, miss rate | Separately powered safety endpoint |
| No temporal scene graph | relation F1, decision error | Separately powered safety endpoint |
| No explanation verifier | entailment accuracy, false-accept rate, fallback rate | Negative-control/descriptive safety analysis |

Every learned condition is independently retrained with identical data and tuning budgets across multiple training seeds. A language-branch ablation that is programmed not to affect control is not evidence of empirical architectural decoupling.

## 8. Missingness and invalid episodes

Eligibility, trigger success, sensor synchronization, route completion, and log integrity are checked without reference to the result direction. Every excluded episode retains its pair ID, reason, method, and processing stage. A primary complete-pair analysis and a conservative sensitivity analysis treating method-specific technical failure as an adverse outcome will be reported.

## 9. Monte Carlo operating characteristics

Each canonical condition uses 50,000 complete simulated trials. If \(X\) of \(R\) trials reject,

\[
\widehat{Power}=X/R,\qquad MCSE=\sqrt{\widehat{Power}(1-\widehat{Power})/R}.
\]

The output reports MCSE and a Wilson 95% Monte Carlo interval. Null scenarios with equal marginal rates validate type-I error. The simulation seed is set once per labeled stream; it is not reset inside a repetition loop.

## 10. Reporting

The future empirical report will include absolute risk difference, risk ratio, paired discordant table, exact or cluster-aware p-value, confidence interval, event counts, method and training seeds, protocol deviations, and all prespecified secondary endpoints. A non-significant result will be reported as inconclusive unless the design supports an equivalence or non-inferiority conclusion.

## 11. Interpretation boundary

No synthetic planning output may be placed in an abstract as achieved method performance. Tables and figures must use “assumed,” “planning,” “prospective,” and “conditional” language. CARLA, public-dataset, real-LLM, human-annotation, and hardware profiling evidence will be reported only after those systems have actually been run.
