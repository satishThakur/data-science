# Chapters 7 & 8 Implementation Plan

**Goal**: Complete Chapters 7 and 8 comprehensively in 2-3 sessions each

---

## Chapter 7: Overfitting & Model Comparison

### Learning Objectives
1. Understand overfitting and the bias-variance tradeoff
2. Use information criteria (WAIC, PSIS-LOO) for model comparison
3. Apply cross-validation
4. Use regularizing priors to prevent overfitting

### Session Structure (2-3 sessions)

#### **Session 1: Information Theory & WAIC**
**Duration**: ~2 hours

**Topics**:
- Information theory basics (entropy, KL divergence, deviance)
- Why AIC doesn't work for multilevel models
- WAIC (Widely Applicable Information Criterion)
- Effective number of parameters (p_WAIC)
- Comparing models with WAIC

**Notebooks**:
1. `information_theory_basics.ipynb`
   - Entropy and uncertainty
   - KL divergence
   - Deviance as distance from truth
   - Visualizations of overfitting

2. `waic_model_comparison.ipynb`
   - Compute WAIC manually
   - Compare multiple models
   - WAIC weights and model averaging
   - Example: polynomial regression overfitting

**Deliverables**:
- Enhanced `quap.py` with WAIC methods
- Two notebooks with clear examples
- Understanding of model comparison

#### **Session 2: Cross-Validation & Regularization**
**Duration**: ~2 hours

**Topics**:
- Leave-one-out cross-validation (LOO-CV)
- PSIS-LOO (Pareto Smoothed Importance Sampling)
- K-fold cross-validation
- Regularizing priors (skeptical priors)
- When to use what approach

**Notebooks**:
1. `cross_validation.ipynb`
   - LOO-CV from scratch
   - Using PSIS-LOO (via ArviZ if complex)
   - K-fold CV
   - Compare CV with WAIC

2. `regularization.ipynb`
   - Ridge-like priors (small σ)
   - Lasso-like priors (Laplace)
   - Comparing regularized vs non-regularized
   - Optimal regularization strength

**Deliverables**:
- CV utilities added to quap.py or separate module
- Two notebooks
- Clear intuition for regularization

#### **Session 3 (Optional): Practice & Review**
**Duration**: ~1-2 hours

**Topics**:
- Homework problems
- Applied examples with real datasets
- Review and consolidate

**Notebooks**:
1. `chapter7_homework.ipynb`
2. `chapter7_applications.ipynb`

---

## Chapter 8: Interactions

### Learning Objectives
1. Understand when and why to include interactions
2. Model categorical × categorical interactions
3. Model continuous × continuous interactions
4. Model categorical × continuous interactions
5. Interpret interaction effects correctly

### Session Structure (2-3 sessions)

#### **Session 1: Categorical × Categorical Interactions**
**Duration**: ~2 hours

**Topics**:
- What are interactions?
- Why main effects + interactions?
- Categorical × categorical = separate effect for each combination
- Visualization strategies (interaction plots)
- Interpretation with index variables

**Notebooks**:
1. `interactions_intro.ipynb`
   - Concept introduction with simple examples
   - Additive vs multiplicative effects
   - When interactions matter

2. `categorical_categorical_interactions.ipynb`
   - Example: Gender × Department effects on salary
   - Index variable approach
   - Dummy variable approach
   - Visualizing interaction effects
   - Statistical significance of interactions

**Deliverables**:
- Clear mental model of interactions
- One comprehensive notebook
- Visualization templates

#### **Session 2: Continuous Interactions**
**Duration**: ~2 hours

**Topics**:
- Continuous × continuous interactions (surfaces, slopes)
- Categorical × continuous (different slopes per group)
- Centering and interpretation
- 3D visualization and contour plots

**Notebooks**:
1. `continuous_continuous_interactions.ipynb`
   - Example: Temperature × Humidity on plant growth
   - Interaction as "slope depends on other variable"
   - 3D surface plots
   - Marginal effects at different values

2. `categorical_continuous_interactions.ipynb`
   - Example: Treatment × dosage (different dose-response curves)
   - Separate slopes per category
   - Visualizing with multiple regression lines
   - Testing if slopes differ

**Deliverables**:
- Two notebooks with rich visualizations
- Interpretation guidelines
- Reusable plotting functions

#### **Session 3: Practice & Complex Interactions**
**Duration**: ~1-2 hours

**Topics**:
- Three-way interactions (if time permits)
- Homework problems
- Real-world applications
- When NOT to include interactions

**Notebooks**:
1. `chapter8_homework.ipynb`
2. `chapter8_applications.ipynb`

---

## Required `quap.py` Enhancements

### Priority 1: WAIC Support (Chapter 7 Session 1)

Add methods to `QuapResult` class:

```python
class QuapResult:
    # ... existing code ...

    def log_likelihood(self,
                      data: np.ndarray,
                      log_lik_fn: Callable) -> np.ndarray:
        """
        Compute log-likelihood for each observation at each posterior sample.

        Parameters:
            data: Observed data (n_obs x n_features)
            log_lik_fn: Function that takes (params, observation) -> log_lik

        Returns:
            np.ndarray: Log-likelihoods (n_samples x n_obs)
        """
        pass

    def waic(self,
             data: np.ndarray,
             log_lik_fn: Callable,
             pointwise: bool = False) -> Union[float, Dict]:
        """
        Compute WAIC (Widely Applicable Information Criterion).

        WAIC = -2 * (lppd - p_WAIC)
        where:
            lppd = log pointwise predictive density
            p_WAIC = effective number of parameters

        Parameters:
            data: Observed data
            log_lik_fn: Function computing log P(y_i | θ)
            pointwise: If True, return pointwise components

        Returns:
            If pointwise=False: WAIC value (scalar)
            If pointwise=True: Dict with components
        """
        pass

    def compare_waic(self, other_model: 'QuapResult', ...) -> pd.DataFrame:
        """Compare this model with another using WAIC."""
        pass
```

### Priority 2: Cross-Validation (Chapter 7 Session 2)

```python
def loo_cv(self,
          data: np.ndarray,
          log_lik_fn: Callable) -> Dict:
    """
    Leave-one-out cross-validation using importance sampling.

    Returns:
        Dict with LOO score, standard error, and diagnostics
    """
    pass

def k_fold_cv(self,
             data: np.ndarray,
             neg_log_posterior_fn: Callable,
             k: int = 10) -> Dict:
    """
    K-fold cross-validation (requires re-fitting).

    Returns:
        Dict with mean deviance and standard error
    """
    pass
```

### Priority 3: Model Comparison Utilities

```python
def compare_models(models: List[QuapResult],
                  data: np.ndarray,
                  log_lik_fn: Callable,
                  criteria: str = 'waic') -> pd.DataFrame:
    """
    Compare multiple models using information criteria.

    Parameters:
        models: List of fitted QuapResult objects
        data: Observed data
        log_lik_fn: Log-likelihood function
        criteria: 'waic' or 'loo'

    Returns:
        DataFrame with WAIC/LOO, SE, dWAIC, weights, etc.
    """
    pass
```

---

## Implementation Checklist

### Chapter 7 Prep (Before Session 1)
- [ ] Add `log_likelihood()` method to QuapResult
- [ ] Add `waic()` method to QuapResult
- [ ] Add standalone `compare_models()` function
- [ ] Test WAIC implementation with simple example
- [ ] Create `notebooks/chapter7/` directory
- [ ] Create `notebooks/chapter7/CLAUDE.md`

### Chapter 7 Session 1
- [ ] Notebook: Information theory basics
- [ ] Notebook: WAIC model comparison
- [ ] Verify WAIC calculations match R/ArviZ
- [ ] Update chapter 7 CLAUDE.md

### Chapter 7 Session 2
- [ ] Add `loo_cv()` or use ArviZ for PSIS-LOO
- [ ] Add `k_fold_cv()` method
- [ ] Notebook: Cross-validation
- [ ] Notebook: Regularization
- [ ] Update chapter 7 CLAUDE.md

### Chapter 7 Session 3
- [ ] Notebook: Homework
- [ ] Chapter 7 summary and review
- [ ] Update TODO.md and MEMORY.md

### Chapter 8 Prep (Before Session 1)
- [ ] Create `notebooks/chapter8/` directory
- [ ] Create `notebooks/chapter8/CLAUDE.md`
- [ ] Review interaction plotting libraries (optional)

### Chapter 8 Session 1
- [ ] Notebook: Interactions intro
- [ ] Notebook: Categorical × categorical
- [ ] Update chapter 8 CLAUDE.md

### Chapter 8 Session 2
- [ ] Notebook: Continuous × continuous
- [ ] Notebook: Categorical × continuous
- [ ] Create reusable interaction plotting utilities
- [ ] Update chapter 8 CLAUDE.md

### Chapter 8 Session 3
- [ ] Notebook: Homework
- [ ] Chapter 8 summary and review
- [ ] Update TODO.md and MEMORY.md

---

## Key Datasets

### Chapter 7
- **Polynomial overfitting**: Simulated data with known generating process
- **Car mileage**: Compare linear, polynomial, spline models
- **Plant growth**: Model selection example

### Chapter 8
- **Rugged terrain & GDP**: Continuous × continuous (Africa interaction)
- **Tulips**: Multiple continuous predictors with interactions
- **UCB admissions**: Categorical × categorical
- **Custom simulated data**: To illustrate each interaction type clearly

---

## Success Criteria

### Chapter 7
- ✅ Can compute and interpret WAIC
- ✅ Can compare models with information criteria
- ✅ Understand effective number of parameters
- ✅ Can apply cross-validation
- ✅ Know when and how to regularize
- ✅ `quap.py` has robust model comparison tools

### Chapter 8
- ✅ Can specify interaction models correctly
- ✅ Can visualize all three interaction types
- ✅ Can interpret interaction coefficients
- ✅ Know when interactions are necessary
- ✅ Can center variables appropriately

---

## Timeline Estimate

| Chapter | Sessions | Hours | Completion Date |
|---------|----------|-------|-----------------|
| Chapter 7 | 2-3 | 4-6 hours | Week 1-2 |
| Chapter 8 | 2-3 | 4-6 hours | Week 2-3 |

**Total**: 8-12 hours of focused work over 2-3 weeks

---

*Created: 2026-02-21*
*Status: Planning complete, ready to begin*
