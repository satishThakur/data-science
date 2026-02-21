# Chapter 7: Overfitting & Model Comparison

**Focus**: Information criteria, cross-validation, and regularization

## Key Concepts

### 1. Information Theory Basics
- Entropy and uncertainty
- KL divergence (information distance)
- Deviance as distance from perfect prediction

### 2. WAIC (Widely Applicable Information Criterion)
- Log pointwise predictive density (lppd)
- Effective number of parameters (p_WAIC)
- WAIC = -2(lppd - p_WAIC)
- When to use vs AIC

### 3. Cross-Validation
- Leave-One-Out (LOO-CV)
- PSIS-LOO (Pareto Smoothed Importance Sampling)
- K-fold cross-validation
- When CV is better than WAIC

### 4. Regularization
- Regularizing priors (skeptical priors)
- Ridge-like priors (small variance)
- Preventing overfitting
- When to regularize vs model selection

## Planned Notebooks

- [ ] `information_theory_basics.ipynb` - Entropy, KL divergence, deviance
- [ ] `waic_model_comparison.ipynb` - Computing and using WAIC
- [ ] `cross_validation.ipynb` - LOO-CV and K-fold
- [ ] `regularization.ipynb` - Regularizing priors
- [ ] `chapter7_homework.ipynb` - Practice problems

## Required Infrastructure

### quap.py Enhancements Needed
- `waic()` method - compute WAIC for a model
- `compare_models()` function - compare multiple models
- `loo()` method - LOO cross-validation (or use ArviZ)
- Log-likelihood computation utilities

## Datasets

- Polynomial overfitting examples (simulated)
- Car data for model comparison
- Other datasets from Statistical Rethinking

## Next Steps

1. Enhance quap.py with WAIC support (do together)
2. Session 1: Information theory and WAIC
3. Session 2: Cross-validation and regularization
4. Session 3 (optional): Practice and review

---

*Created: 2026-02-21*
*Status: Planning phase - ready to begin implementation*
