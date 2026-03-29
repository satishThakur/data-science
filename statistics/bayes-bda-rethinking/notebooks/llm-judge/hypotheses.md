# Hypotheses & Experiments — LLM Judge Bayesian Inference

Empirical questions to answer through the notebook experiments.
Each hypothesis is tested by changing `ACTIVE_SCENARIO` in the master config cell.

---

## Observation That Sparked These Experiments

**Baseline run** (`baseline` scenario, N_val=50, N_prod=200, flat priors):

```
θ  posterior mean = 0.835,  89% CI = [0.694, 0.980],  truth = 0.70
```

The posterior mean is 0.135 above truth. Truth is barely inside the CI at the lower edge.
Two sampling flukes coincided:
- Production: K=137 drawn, expected ~127 (K/N = 0.685 vs true p_judge = 0.637)
- Validation: observed TPR = 0.780 vs true TPR = 0.820 (underestimated sensitivity)

The underestimated TPR forced the model to inflate θ to explain the observed passes.

**Parked question**: is the bell-shaped induced prior on p_judge (centred at 0.5) pulling the
posterior upward? Analysis shows it is pulling *downward* (toward θ≈0.5) but is too weak
(Beta(1,1)) to resist the data. So the prior is not the cause — data flukes are.

---

## Hypothesis 1 — Sample Size Drives Posterior Accuracy

> **As N increases (validation and production), the posterior mean converges to the true θ
> and the credible interval narrows — regardless of prior choice.**

### Why we expect this

With large N:
- The likelihood becomes sharply peaked around the true parameter values
- The prior contribution becomes negligible relative to the likelihood
- Sampling flukes (like the ones in the baseline) average out

### Experiment design

Fix: flat priors Beta(1,1) for all parameters, same true θ=0.70, TPR=0.82, TNR=0.79.
Vary: N across a grid — e.g. N_val = N_prod/4 ∈ {10, 50, 200, 1000}.

| Scenario | N_val (+/-) | N_prod | Priors |
|----------|-------------|--------|--------|
| `small_flat` | 10 | 40 | Flat |
| `baseline` | 50 | 200 | Flat |
| `large_flat` | 200 | 1000 | Flat |

**What to look at:**
- Posterior mean of θ: does it converge toward 0.70?
- 89% CI width: does it shrink?
- Does truth stay inside the CI throughout?

**Expected result:**
Posterior mean drifts toward 0.70 and CI narrows as N grows. Even with flat priors,
large data overwhelms prior and sampling noise.

---

## Hypothesis 2 — Informative Priors Help When Data Is Scarce

> **With small N, informative priors on TPR and TNR produce a tighter, better-centred
> posterior on θ compared to flat priors.**

### Why we expect this

With small validation sets:
- TPR and TNR are poorly estimated (wide posteriors)
- Uncertainty in TPR/TNR propagates into θ via marginalisation
- An informative prior on TPR/TNR (e.g. Beta(8,2) encoding "judge is ~80% accurate")
  anchors their posteriors, reducing the uncertainty that flows into θ

### Experiment design

Fix: small data (N_val=10, N_prod=40), same true parameters.
Vary: prior informativeness.

| Scenario | N_val (+/-) | N_prod | Priors |
|----------|-------------|--------|--------|
| `small_flat` | 10 | 40 | Flat Beta(1,1) |
| `small_informative` | 10 | 40 | Beta(8,2) for TPR/TNR, Beta(3,2) for θ |

**What to look at:**
- 89% CI width for θ: informative should be narrower
- Posterior mean of θ: informative should be closer to truth
- Posterior of TPR/TNR: informative should be better anchored

**Expected result:**
Informative priors produce visibly tighter θ posterior. The prior knowledge that
"the judge is approximately 80% accurate" regularises the TPR/TNR estimates,
preventing the inflation of θ seen in the baseline.

---

## Hypothesis 3 — Large Data Washes Out the Prior

> **With large N, flat and informative priors produce nearly identical posteriors.
> The likelihood dominates and prior choice becomes irrelevant.**

### Why we expect this

This is the Bayesian consistency theorem: as N → ∞, the posterior concentrates on the
true value regardless of the prior (as long as the prior assigns non-zero probability there).

### Experiment design

Fix: large data (N_val=200, N_prod=1000), same true parameters.
Vary: prior informativeness.

| Scenario | N_val (+/-) | N_prod | Priors |
|----------|-------------|--------|--------|
| `large_flat` | 200 | 1000 | Flat Beta(1,1) |
| `large_informative` | 200 | 1000 | Beta(8,2) for TPR/TNR, Beta(3,2) for θ |

**What to look at:**
- Overlay posteriors from both scenarios: they should be nearly identical
- CI width: both should be narrow
- Posterior mean: both should be close to truth (0.70)

**Expected result:**
The two posteriors are visually indistinguishable. Prior choice becomes irrelevant.
This is the most important Bayesian consistency result.

---

## Combined Visualisation Plan

For each hypothesis, the notebook will produce:

1. **Marginal posterior of θ** — prior (dashed) + posterior (filled) + 89% CI + truth marker
2. **CI width vs N** — line plot showing 89% CI width shrinking as N grows (H1)
3. **Prior sensitivity plot** — flat vs informative posteriors overlaid at same N (H2, H3)
4. **Summary table** — posterior mean, CI, CI width, distance from truth across all scenarios

---

## Open Question (Parked)

Does the bell-shaped induced prior on p_judge (centred at 0.5 under flat parameter priors)
contribute to the high posterior mean of θ in small-data regimes?

**Approach to test**: compare the baseline posterior against a run where the prior on θ
is explicitly centred at 0.5 vs centred at 0.7. If the posterior shifts, the prior shape
is contributing. If not, it is purely data-driven.

---

*Created: 2026-03-29*
*Status: H1, H2, H3 to be implemented in notebook Phases 6–7*
