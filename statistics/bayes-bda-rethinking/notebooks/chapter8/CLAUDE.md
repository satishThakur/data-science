# Chapter 8: Interactions

**Focus**: Modeling interaction effects between predictors

## Key Concepts

### 1. What Are Interactions?
- When the effect of X on Y depends on Z
- Additive vs multiplicative relationships
- Why "main effects + interaction" matters

### 2. Categorical × Categorical Interactions
- Separate effect for each combination
- Index variable approach
- Dummy variable approach
- Interpretation and visualization

### 3. Continuous × Continuous Interactions
- Interaction as "slope depends on other variable"
- 3D surfaces and contour plots
- Marginal effects at different values
- Centering for interpretation

### 4. Categorical × Continuous Interactions
- Different slopes per category
- Parallel vs non-parallel regression lines
- Testing if slopes differ significantly

## Planned Notebooks

- [ ] `interactions_intro.ipynb` - Concept introduction
- [ ] `categorical_categorical_interactions.ipynb` - Category × category
- [ ] `continuous_continuous_interactions.ipynb` - Continuous × continuous
- [ ] `categorical_continuous_interactions.ipynb` - Category × continuous
- [ ] `chapter8_homework.ipynb` - Practice problems

## Key Patterns

### Model Specification
```python
# Categorical × Categorical
mu = alpha[category1, category2]  # Separate mean for each combo

# Continuous × Continuous
mu = alpha + beta1*x1 + beta2*x2 + beta3*x1*x2  # Interaction term

# Categorical × Continuous
mu = alpha[category] + beta[category]*x  # Different slope per category
```

### Visualization
- Interaction plots for categorical
- 3D surface plots for continuous
- Multiple regression lines for mixed

## Datasets

- **Rugged terrain & GDP**: Continuous × continuous (Africa interaction)
- **Tulips**: Multiple continuous predictors
- **UCB admissions**: Categorical examples
- **Custom simulations**: Clear illustration of each type

## Next Steps

1. Session 1: Categorical × categorical interactions
2. Session 2: Continuous interactions (both types)
3. Session 3 (optional): Practice and complex cases

---

*Created: 2026-02-21*
*Status: Planning phase - waiting for Chapter 7 completion*
