# Developer PR Rate — Hierarchical Inference

End-to-end Bayesian hierarchical model for developer PR rates across designations and teams.

---

## Problem Statement

We have **250 developers** across **10 teams** and **5 designations** (aSDE → SDE-4).
Each developer's 6-month PR count is observed.

**Questions we want to answer:**
1. How does PR rate vary by designation?
2. How much variation is due to team vs. individual?
3. Where does a specific developer rank?
4. What PR rate should we expect from a **new, unobserved developer** in a given designation+team?

---

## Key Design Decisions

### 1. Crossed vs Nested Hierarchy
Designation and team are **crossed** (not nested). A developer belongs to exactly one team AND one designation, but any designation can appear in any team. So we model them as independent random effects:

```
log(λ_dev) = μ_org + δ_desig + γ_team + ε_dev
```

Not nested like: team → designation → developer.

### 2. Poisson Likelihood with Log Link
PR counts are non-negative integers with no fixed upper bound → Poisson is natural.

```
PR_count_i ~ Poisson(λ_i)
log(λ_i) = α_{dev[i]}
```

### 3. Non-Centred Parameterisation
Instead of `δ ~ Normal(0, σ_desig)` (centred), we use:
```
z_desig ~ Normal(0, 1)
δ = z_desig * σ_desig
```
This avoids funnel geometry in the posterior when σ is small, giving MCMC better mixing.

### 4. Why simulate first?
We simulate data with **known ground truth** to verify the model can recover parameters.
Once we trust the model, we plug in real data — same code, different `pr_obs`.

---

## Model Structure

```
PR_count_i ~ Poisson(λ_i)                          # likelihood
log(λ_i) = α_{dev[i]}                              # log link

α_dev  = μ_dev + z_dev * σ_dev                     # developer noise
μ_dev  = μ_org + δ_{desig[dev]} + γ_{team[dev]}    # crossed effects

δ      = z_desig * σ_desig                         # designation offset
γ      = z_team  * σ_team                          # team offset

z_dev, z_desig, z_team ~ Normal(0, 1)              # non-centred

μ_org       ~ Normal(log(8), 0.5)                  # grand mean ≈ 8 PRs/6mo
σ_desig     ~ Exponential(1)
σ_team      ~ Exponential(1)
σ_dev       ~ Exponential(1)
```

---

## Ground Truth Parameters (Simulation)

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| μ_org | log(8) ≈ 2.08 | Grand mean: ~8 PRs per 6 months |
| σ_desig | 0.4 | Designation-level spread |
| σ_team | 0.3 | Team-level spread |
| σ_dev | 0.4 | Individual developer spread |

### Designation Distribution (pyramid)
| Designation | Count | True offset (log scale) | Expected PRs/6mo |
|-------------|-------|------------------------|-----------------|
| aSDE  | 75  | −0.2 | 6.6  |
| SDE-1 | 63  | +0.3 | 10.8 |
| SDE-2 | 50  | +0.2 | 9.8  |
| SDE-3 | 37  | −0.3 | 5.9  |
| SDE-4 | 25  | −0.7 | 4.0  |

**Rationale for offsets:**
- SDE-1 peaks: fast learners, high output, feature work
- SDE-2 still high: productive but more design involved
- aSDE slightly below mean: still ramping up
- SDE-3/4 lowest: architectural decisions, mentoring, code review — less volume

---

## Notebook Plan

| Notebook | Content |
|----------|---------|
| `01_simulate.ipynb` | Simulate data with known ground truth. Save to `data/pr_simulated.csv` |
| `02_eda.ipynb` | Exploratory analysis of simulated data |
| `03_prior_predictive.ipynb` | Prior predictive check — do priors produce sensible counts? |
| `04_fit_model.ipynb` | Fit hierarchical model. Check convergence. |
| `05_recovery.ipynb` | Did the model recover the true parameters? |
| `06_inference.ipynb` | Designation distributions, developer ranking, new developer prediction |

---

## Key Statistical Concepts

### Partial Pooling (Shrinkage)
- **No pooling**: estimate each developer independently → overfit, extreme values stay extreme
- **Complete pooling**: one estimate for all → underfit, ignores individual differences
- **Partial pooling** (hierarchical): developers share information via the hyperprior → extreme values shrink toward group mean, especially for small groups

### Predicting for New Developers
For a new developer (not in the data), we:
1. Use posterior of μ_org, δ_desig, γ_team
2. Sample fresh `z_dev ~ Normal(0, 1)` scaled by posterior σ_dev
3. This gives wider, more honest intervals than using the group mean alone

### Why log scale for offsets?
On log scale, effects are multiplicative. An offset of +0.3 means `exp(0.3) ≈ 1.35x` the baseline rate.
This is more interpretable than additive counts for Poisson data.

---

*Last updated: 2026-03-15*
