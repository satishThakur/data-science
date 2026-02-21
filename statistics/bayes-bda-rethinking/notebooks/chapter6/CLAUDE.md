# Chapter 6: Causal Inference

This chapter focuses on **causal reasoning** using Directed Acyclic Graphs (DAGs) and understanding confounding, mediation, and selection bias.

## Key Concepts

### 1. Multicollinearity
**Problem**: When predictors are highly correlated, individual coefficients become uncertain but their sum remains precise.

**Notebooks**:
- `multicollinearity.ipynb` - Legs example (perfect correlation) and milk example (negative correlation)

**Key Insights**:
- Individual βs: Very wide, unstable posteriors (37x wider!)
- Sum of βs: Remains precise
- Posterior correlation: Opposite sign to data correlation
- Solution: Drop variables, combine them, or accept it for prediction

### 2. Directed Acyclic Graphs (DAGs)
**Purpose**: Represent causal assumptions explicitly before modeling.

**DAG Elements**:
- Nodes: Variables
- Arrows: Direct causal effects
- Paths: Sequences of arrows (any direction)
- Backdoor paths: Non-causal paths from treatment to outcome

**Key Questions**:
1. What paths exist from X to Y?
2. Which paths are causal (forward)?
3. Which paths are backdoor (confounding)?
4. What should we condition on?

### 3. Confounding
**Pattern**: X ← Z → Y (Z is a confounder)
- Z causes both X and Y
- Creates spurious association between X and Y
- **Solution**: Condition on Z to block the backdoor path

### 4. Collider Bias
**Pattern**: X → C ← Y (C is a collider)
- X and Y both cause C
- Without conditioning: X ⊥ Y (independent)
- Conditioning on C: X and Y become associated (spurious!)
- **Solution**: NEVER condition on colliders

**Notebooks**:
- `collider_bias.ipynb` - Examples with trustworthiness/funding, age/happiness

### 5. Post-Treatment Bias
**Pattern**: X → M → Y (M is a mediator/post-treatment)
- M is caused by treatment X and causes outcome Y
- Conditioning on M blocks the indirect path X → M → Y
- Only estimates direct effect, misses total causal effect
- **Solution**: Don't condition on consequences of treatment

**Notebooks**:
- `post_treatment_bias.ipynb` - Plant growth example

## Datasets Used

### Foxes (`foxes.csv`)
Urban fox data for causal inference practice.

**Variables**:
- `weight`: Body weight (outcome)
- `area`: Territory size
- `avgfood`: Average food availability
- `groupsize`: Number of foxes in group

**DAG**:
```
F (Food) → A (Area) → W (Weight) ← G (Group) ← F
```

**Causal Stories**:
- More food → larger territories → heavier foxes
- More food → larger groups → more competition → lighter foxes
- These effects can cancel out in naive regressions!

### Primate Milk (`milk.csv`)
**Variables**: `kcal.per.g`, `perc.fat`, `perc.lactose`, `mass`, `neocortex.perc`

**Used for**: Multicollinearity examples (fat and lactose highly negatively correlated)

## Homework Problems (6H3-6H7)

**Status**: In progress

**Files**:
- `homework_6H3_6H7.ipynb`

**Problems**:
1. **6H3**: Causal effect W → A (wrong direction - pedagogical)
2. **6H4**: Causal effect A → W (condition on F)
3. **6H5**: Causal effect G → W (condition on F)
4. **6H6**: Design your own research DAG
5. **6H7**: Complex DAG analysis with conditional independence

## Key Principles for This Chapter

### The DAG Workflow
1. **Think causally**: What causes what? Draw the DAG first
2. **Identify paths**: List all paths from treatment to outcome
3. **Classify paths**:
   - Direct (causal effect you want)
   - Backdoor (confounding - must block)
   - Mediated (part of effect - don't block)
4. **Find conditioning set**: What to adjust for?
5. **Fit the model**: Only then write code

### What to Condition On

| Pattern | Name | Condition? | Why |
|---------|------|------------|-----|
| X ← Z → Y | Confounder | ✓ Yes | Blocks backdoor path |
| X → M → Y | Mediator | ✗ No | Would block the effect |
| X → C ← Y | Collider | ✗ No | Creates spurious association |
| X ← Z → M → Y | Pipe | ✓ On Z only | Z sufficient to block |

### Common Mistakes

1. **Conditioning on everything**: "Control for all variables" is wrong!
   - Can block the effect (mediators)
   - Can create bias (colliders)

2. **Confusing association with causation**:
   - Association: Statistical pattern in data
   - Causation: What happens when you intervene

3. **Ignoring causal direction**:
   - A → B vs B → A have different implications
   - DAG forces you to commit to causal assumptions

## Statistical Models from This Chapter

### Model 1: Naive (No Conditioning)
```python
# W ~ A (bivariate)
# Includes both direct effect AND confounding
```

### Model 2: Conditioning on Confounder
```python
# W ~ A + F (blocks backdoor: A ← F → G → W)
# Estimates causal effect of A → W
```

### Model 3: Over-Conditioning (Wrong!)
```python
# W ~ A + F + G
# Conditioning on G (descendant on path) can create bias
```

## Visualization Patterns

1. **DAG diagrams**: Box-and-arrow graphs (draw manually or use dagitty)
2. **Scatter plots by group**: Show confounding visually
3. **Coefficient comparisons**: Compare β with/without conditioning
4. **Posterior distributions**: Show uncertainty in causal effects

## Next Steps

After Chapter 6:
- Complete all homework problems
- Review DAG reasoning with additional examples
- Practice identifying backdoor paths
- Move to Chapter 7: Model comparison and regularization

## References

- Statistical Rethinking Chapter 6
- Pearl's *Book of Why* for deeper causal inference
- dagitty.net for interactive DAG analysis

---

*Last updated: 2026-02-21*
*Status: Homework problems 6H3-6H7 in progress*
