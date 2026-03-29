# CLAUDE.md — LLM-as-a-Judge Bayesian Inference

This file provides context for Claude Code sessions working in this directory.

## What This Is

A self-contained, pedagogical Bayesian inference project built around a real-world problem:
**inferring the true pass rate of an LLM-as-a-Judge evaluation system**, accounting for judge imperfection.

The goal is not just to solve the problem — it is to use this problem as a vehicle to teach
Bayesian thinking from first principles: priors, likelihoods, posteriors, marginalisation,
maximum entropy, and the role of data size.

## The Problem

An LLM judge evaluates production traces and classifies each as PASS or FAIL.
We observe K passes out of N traces. Naively, K/N is the pass rate — but the judge is imperfect.

We also have a **human-labelled validation set** that tells us:
- **TPR** (True Positive Rate): how often the judge correctly says PASS when the trace truly passes
- **TNR** (True Negative Rate): how often the judge correctly says FAIL when the trace truly fails

We assume TPR ≈ TNR ≈ 0.80 (a reasonably well-trained judge, not perfect).

We want to recover the **true underlying pass rate θ** — not the noisy judge-reported rate.

## The Model

### Variables

| Symbol   | Type          | Meaning                                      |
|----------|---------------|----------------------------------------------|
| θ        | Parameter     | True pass rate in production (~, has prior)  |
| TPR      | Parameter     | Judge sensitivity P(Judge=PASS \| True=PASS) (~, has prior) |
| TNR      | Parameter     | Judge specificity P(Judge=FAIL \| True=FAIL) (~, has prior) |
| p_judge  | Deterministic | θ·TPR + (1-θ)·(1-TNR)  (=, computed)        |
| N+, K_PP | Observed      | Human positives in validation; judge agrees  |
| N-, K_NN | Observed      | Human negatives in validation; judge agrees  |
| N, K     | Observed      | Production traces; judge says PASS           |

### Key Equation

    p_judge = θ · TPR + (1 - θ) · (1 - TNR)

### Generative Model

    θ   ~ Beta(α_θ, β_θ)
    TPR ~ Beta(α_TPR, β_TPR)
    TNR ~ Beta(α_TNR, β_TNR)

    K_PP  ~ Binomial(N+,  TPR)          ← validation: true positives
    K_NN  ~ Binomial(N-,  TNR)          ← validation: true negatives
    K     ~ Binomial(N,   p_judge)      ← production: observed passes

### Posterior

    P(θ, TPR, TNR | data) ∝ P(K | N, p_judge) · P(K_PP | N+, TPR) · P(K_NN | N-, TNR)
                            · P(θ) · P(TPR) · P(TNR)

### Identifiability

TPR + TNR > 1 is required. If the judge is no better than random, the production data
carries no information about θ. The validation set is what makes the model identifiable.

## Notebook Structure (llm_judge_bayesian.ipynb)

| # | Cell(s) | Content |
|---|---------|---------|
| 1 | `title-cell` (md), `7y7gergv9b` (md) | Problem statement + full mathematical framework (1.1–1.6) |
| 2 | `1645h6nl1o8` (code) | DAG with colour-coded nodes |
| 3 | `imports-cell` (code) | numpy, scipy, pymc, arviz, matplotlib |
| 4 | `config-cell` (code) | SCENARIOS dict + ACTIVE_SCENARIO selector |
| 5 | `atuoeo19i4w` (code) | ipywidgets dropdown — interactive scenario selector |
| 6 | `sim-cell` (code) | Simulate data from true parameters; confusion matrix printout |
| 7 | `wzybbs2rbo` (md), `yulhkdxs3i` (code) | Prior predictive check — 3×3 grid of plots |
| 8 | `89w14b1icyn` (md), `63omxxi5koy` (code), `63fnhvr9yyj` (code) | Grid approximation — 3D log-space computation, marginalisation, 2×3 visualisation |
| 9 | `244a2zv39fu` (md), `ca39fec2...` (code) | MCMC intro + PyMC model |
| 10 | `9n53kxsl7t` (code) | Convergence diagnostics (r_hat, ESS, trace plots) + 89% CI in plain English |
| 11 | `bb43vu5b9gu` (code) | Grid vs MCMC overlay comparison |
| 12 | `60f6fc77...` (md), `fm28urdhf2` (code) | Posterior predictive check |
| 13 | `30529179...` (md), `yrbh4mjf5s` (code) | Hypothesis experiments — helper + run 4 scenarios |
| 14 | `ijy4joy1c7i` (code) | Hypothesis A plot (KDE overlay, H_A verdict) |
| 15 | `que23yvwlq` (code) | Hypothesis B plot (KDE overlay) + full summary table |

**Pending:** Teaching summary cell (Part 7)

## Scenarios (Master Config)

| Scenario             | N_val+/- | N_prod | Priors         | Purpose       |
|----------------------|----------|--------|----------------|---------------|
| `baseline`           | 50/50    | 200    | Flat           | Default run   |
| `small_flat`         | 10/10    | 40     | Flat           | H_A reference |
| `small_informative`  | 10/10    | 40     | Informative    | H_A experiment|
| `large_flat`         | 200/200  | 1000   | Flat           | H_B reference |
| `large_informative`  | 200/200  | 1000   | Informative    | H_B experiment|

Informative priors: `theta_prior=(3,2)`, `tpr_prior=(8,2)`, `tnr_prior=(8,2)`

## Key Implementation Details

### Grid Approximation
- Grid size: 50 per dimension (50³ = 125,000 points)
- All computation in log-space; normalised via `logsumexp` to prevent underflow
- Axes: `axis=0` → θ, `axis=1` → TPR, `axis=2` → TNR
- Marginalisation: `.sum(axis=(1,2))` for θ, `.sum(axis=(0,2))` for TPR, etc.
- Shape annotation in code: every array annotated with `# shape (50, 50, 50)`

### MCMC
- NUTS sampler: 2000 draws, 1000 tune steps, `target_accept=0.9`, `random_seed=42`
- 4 chains by default → 8,000 total posterior samples
- Convergence check: r_hat < 1.01 and ess_bulk > 400
- `p_judge` declared as `pm.Deterministic` so ArviZ tracks it in the trace

### Visualisation Conventions
- **Overlapping posteriors**: always use KDE curves (`scipy.stats.gaussian_kde`) with light
  `fill_between` (alpha=0.15) — never overlapping histograms (colors blend confusingly)
- **Discrete count histograms**: `np.arange(0, n+2) - 0.5` for integer-aligned bins
- **CI shading**: `axvspan` with alpha=0.12–0.15
- **Truth marker**: crimson vertical line, lw=2
- **Credible intervals**: 89% (McElreath convention) via `np.percentile(s, [5.5, 94.5])`

### Plain-English CI Output Pattern
```python
print(f'Given the data, there is an 89% probability the {label}')
print(f'lies between {lo:.3f} and {hi:.3f}.')
```

### ipywidgets Selector
- Dropdown + Output widget after config cell
- `on_change` callback updates `ACTIVE_SCENARIO` and `cfg` globals
- Widget only sets config — user re-runs cells below (MCMC is too slow to auto-trigger)

### Posterior Predictive Check
- Uses `pm.sample_posterior_predictive(idata)` inside `with judge_model:` block
- Checks all three observables: K_prod, k_pp, k_nn
- Prints percentile of observed value in the predictive distribution

### Hypothesis Experiments
- `run_scenario(name, seed=42)` helper: simulate data + run MCMC, returns `(cfg, data, idata)`
- All 4 scenarios run in a single cell; stored in `results` dict
- H_A: compare `small_flat` vs `small_informative` — shows CI width reduction
- H_B: compare `large_flat` vs `large_informative` — shows convergence
- Final summary table: mean, CI, width, truth, inside ✓/✗ for all 4 scenarios

## Key Teaching Points

1. **The naive estimate K/N is biased** — it conflates judge errors with true signal
2. **Parameter taxonomy** — Observed vs Parameter (~) vs Deterministic (=)
3. **Flat priors are not non-informative on derived quantities** — Beta(1,1)³ → bell-shaped p_judge
4. **Marginalisation is honest** — integrate out nuisance parameters, don't fix them
5. **Priors matter when data is scarce** — informative priors regularise the posterior (H_A)
6. **Data overwhelms priors** — with enough data, prior choice becomes irrelevant (H_B)
7. **Grid vs MCMC** — same posterior, different computation; grid shows the glass box, MCMC scales
8. **Posterior predictive check** — sanity check before trusting any conclusions
9. **Maximum entropy** — Beta-Binomial is not arbitrary; least-assumption model for this structure
10. **Identifiability** — TPR + TNR > 1; validation set is what makes θ recoverable

## Conventions

- All data simulated from known true parameters (teaching — real use would substitute actual data)
- Seed: 42 for all random number generation
- Credible intervals: 89% (5.5th to 94.5th percentile)

## Dependencies

PyMC, ArviZ, NumPy, SciPy, Matplotlib, ipywidgets. See pyproject.toml in project root.

---

*Directory created: 2026-03-28*
*Last updated: 2026-03-29*
*Notebook status: Parts 1–6 complete; Part 7 (teaching summary) pending*
