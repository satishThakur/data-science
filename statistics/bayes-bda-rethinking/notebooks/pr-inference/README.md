# Developer PR Rate — Hierarchical Bayesian Inference

End-to-end Bayesian hierarchical model for developer PR rates across designations and teams.
Uses **simulated data with known ground truth** to validate the model before applying to real data.

---

## Problem Statement

We have **316 developers** across **15 teams** and **5 designations** (aSDE → SDE-4).
Each developer's monthly PR count is observed over **6 months** (1,896 total observations).

**Questions we want to answer:**
1. How does PR rate vary by designation and team?
2. How much variation is due to team vs. individual noise?
3. Where does a specific developer rank within their peers?
4. What PR rate should we expect from a **new, unobserved developer** in a given designation+team?

---

## Models

Three models with increasing complexity:

| Model | What it estimates | Key property |
|-------|-------------------|--------------|
| `m_simple` | Per-developer rate only | Baseline, no structure |
| `m1` | Designation + team + developer | No pooling (σ fixed) |
| `m2` | Designation + team + developer | **Partial pooling (σ estimated)** ← main model |

### m_simple

Minimal baseline — estimates one rate per developer, no designation/team effects:

```
log(λ_dev) = log(8) + z_dev * σ_dev

z_dev ~ ZeroSumNormal(1)          # forces mean = 0 → anchors intercept
σ_dev ~ HalfNormal(0.5)
```

### m2 (main model)

Hierarchical crossed model with **partial pooling**:

```
PR_count_i ~ Poisson(λ_i)
log(λ_i) = log(exposure_i) + α_{dev[i]}

α_dev  = μ_org + δ_{desig[dev]} + γ_{team[dev]} + z_dev * σ_dev

δ      = z_desig * σ_desig        # designation offset
γ      = z_team  * σ_team         # team offset

z_dev, z_desig, z_team ~ Normal(0, 1)    # non-centred

μ_org       ~ Normal(log(8), 0.5)
σ_desig     ~ HalfNormal(0.5)
σ_team      ~ HalfNormal(0.5)
σ_dev       ~ HalfNormal(0.5)
```

---

## Key Design Decisions

### 1. Crossed vs Nested Hierarchy

Designation and team are **crossed** (not nested). A developer belongs to exactly one team AND one designation, but any designation can appear in any team:

```
log(λ_dev) = μ_org + δ_desig + γ_team + ε_dev
```

Not nested like: team → designation → developer.

### 2. Poisson Likelihood with Log Link

PR counts are non-negative integers with no fixed upper bound → Poisson is natural.
Log link keeps λ > 0 and makes effects **multiplicative**: an offset of +0.3 means `exp(0.3) ≈ 1.35×` the baseline rate.

### 3. Non-Centred Parameterisation

Instead of `δ ~ Normal(0, σ_desig)` (centred), we use:
```
z_desig ~ Normal(0, 1)
δ = z_desig * σ_desig
```
This avoids **funnel geometry** in the posterior when σ is small, giving MCMC better mixing.

### 4. HalfNormal(0.5) for σ priors

- Same mode at 0 as Exponential(1), but **lighter tails**
- 95% of mass below ~1.0, concentrated in [0.1, 0.8]
- Consistent with true simulation values (0.3–0.4)
- Reduces funnel geometry → fewer divergences during sampling

### 5. ZeroSumNormal in m_simple

Forces `sum(z_dev) = 0`, preventing a ridge between the hardcoded intercept and the individual offsets. Without this constraint, MCMC wanders along a flat direction → R-hat > 1.01.

### 6. Why simulate first?

Simulated data with **known ground truth** lets us verify the model can recover parameters.
Once we trust the model, we plug in real data — same pipeline, different `pr_obs`.

---

## Ground Truth Parameters (Simulation)

| Parameter | True value | Interpretation |
|-----------|------------|----------------|
| μ_org | log(8) ≈ 2.08 | Grand mean: 8 PRs/month |
| σ_desig | 0.4 | Designation-level spread (log scale) |
| σ_team | 0.3 | Team-level spread (log scale) |
| σ_dev | 0.4 | Individual developer spread (log scale) |

### Designation Distribution

| Designation | Developers | Offset (log scale) | Expected PRs/month |
|-------------|-----------|-------------------|-------------------|
| aSDE  | 108 | −0.2 | 6.5  |
| SDE-1 | 82  | +0.3 | 10.8 |
| SDE-2 | 63  | +0.2 | 9.8  |
| SDE-3 | 42  | −0.3 | 5.9  |
| SDE-4 | 21  | −0.7 | 4.0  |

**Rationale for offsets:**
- SDE-1 peaks: fast learners, high output, feature work
- SDE-2 still high: productive but more design involved
- aSDE slightly below mean: ramping up
- SDE-3/4 lowest: architectural decisions, mentoring, code review — less volume

### Team Tiers (3 tiers × 5 teams each)

| Tier | Teams | Seniority mix | Design purpose |
|------|-------|---------------|----------------|
| Balanced | T1–T5 | Even distribution | Baseline; all models estimate well |
| Junior-heavy | T6–T10 | Mostly SDE-1/2/aSDE | SDE-3/4 data sparse → m2 borrows |
| Mostly-junior | T11–T15 | Almost no SDE-3/4 | Near-absent seniority → maximum shrinkage |

This 3-tier design makes **shrinkage visible**: Tier 3 developers with sparse data are pulled strongly toward the group mean in m2 but not in m1.

---

## Notebook Pipeline

| Notebook | Purpose | Key outputs |
|----------|---------|-------------|
| `01_simulate.ipynb` | Generate data with known ground truth | `data/pr_simulated.csv` |
| `02_eda.ipynb` | Verify simulation matches design | Plots, sanity checks |
| `03_models_prior_check.ipynb` | Define models + prior predictive checks | Model objects, prior samples |
| `04_fit_models.ipynb` | Fit m_simple, m1, m2; check convergence | `data/idata_*.nc` |
| `05_inference.ipynb` | Answer business questions from posteriors | Rankings, predictions, standing |
| `app.py` | Interactive Streamlit dashboard | Web UI |

**Dependency order**: 01 → 02 → 03 → 04 → 05 → app.py

Run the app: `streamlit run notebooks/pr-inference/app.py` from project root.

---

## Known Issues

| Issue | Severity | Status |
|-------|----------|--------|
| m1 convergence: R-hat = 1.10, ESS = 40 | High | Identifiability bug — four additive terms without sufficient constraints; don't use m1 for inference |
| m2 divergences: 8 out of 4000 draws | Low | Residual funnel geometry near σ → 0; results reliable but worth monitoring |

---

## Key Statistical Concepts

### Partial Pooling (Shrinkage)
- **No pooling** (m1): estimate each group independently → overfit, extreme values stay extreme
- **Complete pooling**: one estimate for all → underfit, ignores group differences
- **Partial pooling** (m2): groups share information via estimated σ hyperprior → extremes shrink toward group mean, more for small groups

### Predicting for New Developers
For a new developer (unseen), we:
1. Use posterior of μ_org, δ_desig, γ_team
2. Sample fresh `z_dev ~ Normal(0, 1)` scaled by posterior σ_dev
3. This gives **wider, more honest intervals** than using the group mean alone

### Peer Comparison
`P(dev beats random peer)` is more honest than a percentile rank because it accounts for both:
- The developer's own posterior uncertainty
- The peer's individual noise (sampled fresh from σ_dev)

---

*Last updated: 2026-03-24*
