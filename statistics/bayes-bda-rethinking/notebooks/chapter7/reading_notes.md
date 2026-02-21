# Chapter 7 Reading Notes: Overfitting & Model Comparison

**Date**: 2026-02-22
**Status**: In progress - conceptual understanding phase

---

## Core Framework (Pages 1-2)

### The Central Problem

Both **overfitting** and **underfitting** harm our models:

| Problem | Description | Impact |
|---------|-------------|--------|
| **Underfitting** | Model too simple | Misses real patterns (high bias) |
| **Overfitting** | Model too complex | Fits noise as signal (high variance) |

**Why we care**: These problems affect BOTH:
1. **Inference** - Understanding causal relationships, which variables matter
2. **Prediction** - Forecasting accuracy on new data

### Key Insight: Regular Features vs Noise

**Goal of modeling**: Learn the **regular features** from data
- **Regular features** = True underlying patterns/signal that generalize
- **Noise** = Random variation that doesn't generalize

**Overfitting intuition**:
- Learns regular features + noise
- Model becomes too flexible, treats noise as signal
- Example: Fitting a wiggly curve through every data point

**Underfitting intuition**:
- Fails to learn even the regular features
- Model too rigid, misses real patterns
- Example: Fitting a flat line when relationship is curved

**The sweet spot**: Learn signal, ignore noise

---

## Three Tools to Combat Overfitting

Chapter 7 presents three complementary approaches:

### 1. Regularizing Priors
- **What**: Skeptical priors that resist complexity
- **How**: Use priors that penalize large parameter values
- **Best for**: Both inference and prediction
- **Tradeoff**: Need to choose appropriate prior strength

### 2. Information Criteria
- **What**: WAIC, AIC, DIC, etc. - mathematical measures of model quality
- **How**: Balance fit vs complexity using a penalty term
- **Best for**: Model comparison and selection
- **Tradeoff**: Fast but approximate (relies on assumptions)

### 3. Cross-Validation
- **What**: Test model on held-out data
- **How**: LOO-CV, K-fold CV - evaluate out-of-sample performance
- **Best for**: Pure predictive accuracy
- **Tradeoff**: Computationally expensive, requires refitting

### When to Use What?

**Important**: You might need one, two, or all three tools depending on your goals!

- **For inference** (understanding)
  → Regularizing priors + information criteria

- **For prediction** (forecasting)
  → Cross-validation (gold standard)
  → Information criteria (faster approximation)

- **For model comparison**
  → Information criteria (WAIC preferred)
  → Cross-validation (if computational budget allows)

---

## The Bias-Variance Tradeoff

```
Simple Model                     Complex Model
    ↓                                 ↓
Underfitting  ←  SWEET SPOT  →  Overfitting
 (high bias)    (balanced)       (high variance)
 Misses signal                    Learns noise
```

**Bias**: Error from overly simple assumptions
**Variance**: Error from sensitivity to training data

**The fundamental challenge**: We can't just minimize error on training data!
- Training error always decreases as model gets complex
- But out-of-sample error starts increasing (overfitting)

**Chapter 7's central question**:
> "How do we know if our model is too complex?"

**Answer**: We need out-of-sample validation or approximations thereof

---

## Key Principles

### 1. Training Fit ≠ Predictive Accuracy
- A model can fit the training data perfectly (R² = 1)
- But still make terrible predictions on new data
- **Overfitting paradox**: Better training fit can mean worse predictions!

### 2. More Parameters ≠ Better Model
- Adding parameters always improves fit to training data
- But may hurt generalization by learning noise
- Need to penalize complexity

### 3. Regularization is About Information
- Regularizing priors = being skeptical about extreme values
- Information criteria = measuring information content
- Cross-validation = direct test of information transfer to new data

---

## What's Coming Next

Based on this foundation, Chapter 7 will cover:

1. **Information theory basics**
   - Entropy (uncertainty)
   - KL divergence (distance between distributions)
   - Deviance (distance from truth)

2. **WAIC (Widely Applicable Information Criterion)**
   - What it measures
   - How to compute it
   - How to use it for model comparison

3. **Cross-validation**
   - Leave-one-out (LOO)
   - K-fold
   - PSIS-LOO (Pareto Smoothed Importance Sampling)

4. **Regularization in practice**
   - How skeptical should priors be?
   - Ridge-like vs Lasso-like regularization
   - Adaptive regularization

---

## Questions to Explore

- How much regularization is enough?
- When does WAIC work well vs poorly?
- How does cross-validation compare to information criteria?
- ✅ Can we visualize the bias-variance tradeoff? **YES - see polynomial notebook!**

---

## Practical Lessons from Polynomial Example

### 1. Scaling Matters for Priors

**Key insight**: Scale outcomes thoughtfully, not automatically!

- **Predictors**: Z-score standardization (standard practice)
- **Outcomes**: Scale to make priors interpretable

For brain volume:
- **Book approach**: Scale to [0, 1] by dividing by max
  - Zero = no brain (meaningful!)
  - One = largest brain (meaningful!)
  - Priors: α ~ Normal(0.5, 1) makes immediate sense

- **Not**: Z-score standardization
  - Less interpretable for bounded outcomes
  - Harder to set meaningful priors

**Principle**: Choose scaling that helps you set informative priors!

### 2. Overparameterization is Real

With **N data points** and **N+ parameters**, bad things happen:

**Example**: Degree 6 polynomial with 7 observations
- Parameters: α + β₁...β₆ + σ = **8 parameters**
- Data: **7 observations**
- Problem: Hessian is singular, can't compute covariance

**Solution** (from book): Fix σ = 0.001 instead of estimating it
- Reduces to 7 parameters for 7 data points
- Just barely identifiable
- Still a terrible model, but at least it fits!

**Lesson**: Models can be too complex to even fit properly!

### 3. Visualizing Bias-Variance Tradeoff

**Three complementary visualizations**:

1. **R² progression**: Always increases (misleading!)

2. **Credible intervals**:
   - Underfit (degree 1): Narrow but wrong
   - Good fit (degree 2-3): Reasonable width
   - Overfit (degree 5-6): HUGE uncertainty!
   - **Insight**: Overfit models are very uncertain about their bad predictions

3. **Sensitivity to observations**:
   - **Underfit**: Drop any point → barely changes (stable but rigid)
   - **Overfit**: Drop any point → drastically changes (memorizing, not learning)
   - **Good model**: Intermediate stability

### 4. The Fragility of Overfitting

**Key demonstration**: Leave-one-out sensitivity

- **Degree 1 (underfit)**:
  - All leave-one-out fits nearly identical
  - High bias, low variance
  - Too stable (rigid)

- **Degree 6 (overfit)**:
  - Each leave-one-out fit wildly different
  - Low bias, high variance
  - Too fragile (memorizing)

- **Degree 2-3 (just right)**:
  - Moderate sensitivity
  - Balanced bias-variance
  - Generalizable

**Quote from visualization**:
> "Overfit models are not just wrong—they're unstable!"

---

*These notes will be updated as we progress through Chapter 7*
*Last updated: After completing polynomial overfitting notebook*
