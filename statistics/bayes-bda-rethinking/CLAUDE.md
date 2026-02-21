# CLAUDE.md - Bayesian Data Analysis Project

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**Statistical Rethinking & Bayesian Data Analysis**

This is a learning/research repository working through **Statistical Rethinking (2nd Edition)** by Richard McElreath, implementing concepts using Python instead of R. The focus is on understanding Bayesian inference, causal reasoning, and building statistical models from first principles.

## Development Environment

- **Package Manager**: `uv` (Python 3.11+)
- **Primary Tools**: PyMC, ArviZ, NumPy, Pandas, Matplotlib, SciPy
- **Development**: JupyterLab for interactive notebooks
- **Custom Code**: `src/quap.py` - Quadratic Approximation implementation

### Key Dependencies
- **PyMC 5.26.1+**: Probabilistic programming framework
- **ArviZ 0.22.0+**: Bayesian visualization and diagnostics
- **JupyterLab 4.5.0+**: Interactive notebooks
- **SciPy, NumPy, Pandas, Matplotlib**: Core data science stack

## Common Commands

```bash
# Install dependencies
uv sync

# Start Jupyter Lab
jupyter lab

# Add new packages
uv add <package-name>
```

## Project Structure

```
├── CLAUDE.md           # This file - project overview
├── TODO.md             # Progress tracking and next steps
├── main.py             # Entry point (minimal usage)
├── pyproject.toml      # Dependencies and configuration
├── src/                # Custom Python modules
│   └── quap.py        # Quadratic Approximation (replaces rethinking::quap)
├── notebooks/          # Jupyter notebooks organized by chapter
│   ├── foundations/    # Basic probability distributions
│   ├── chapter2/       # Bayesian inference fundamentals
│   ├── chapter3/       # Sampling and posterior distributions
│   ├── chapter4/       # Linear models, splines, polynomials
│   ├── chapter5/       # Multiple regression, confounding
│   ├── chapter6/       # Causal inference, DAGs, multicollinearity
│   └── [chapter]/CLAUDE.md  # Chapter-specific context (see below)
└── data/              # Datasets (downloaded as needed)
```

## Development Workflow

1. **Notebooks First**: All analysis in Jupyter notebooks organized by chapter
2. **Custom Tools**: Utilities in `src/` (e.g., `quap.py` for MAP estimation)
3. **Chapter-by-Chapter**: Follow Statistical Rethinking structure
4. **Python Translation**: Convert R code to Python/PyMC equivalents

## Book Progress & Chapter Status

See chapter-specific `CLAUDE.md` files for detailed context:

- ✅ **Chapter 2**: Bayesian inference basics (grid approximation, sequential updating)
- ✅ **Chapter 3**: Sampling from posterior (HPDI, compatibility intervals)
- ✅ **Chapter 4**: Linear models (Gaussian regression, splines, polynomials)
- ✅ **Chapter 5**: Multiple regression (confounding, masked relationships, categorical variables)
- 🔄 **Chapter 6**: Causal inference (multicollinearity, collider bias, post-treatment bias, DAGs)
  - Currently working on: Homework problems 6H3-6H7
- ⏳ **Chapter 7+**: Not started

## Key Conventions & Patterns

### Notebook Organization
- **One concept per notebook** (e.g., `multicollinearity.ipynb`, `collider_bias.ipynb`)
- **Descriptive markdown**: Explain concepts, not just code
- **Visual emphasis**: Plots to illustrate statistical concepts
- **Standardize variables**: Use `(x - x.mean()) / x.std()` for regression

### Code Style
- **Use `quap`** from `src/quap.py` for quadratic approximation (mimics R's `rethinking::quap`)
- **Negative log posterior**: Define as `neg_log_posterior` functions
- **Transform parameters**: Use `model.transform_param()` for constrained parameters (e.g., log_sigma → sigma)
- **Sample from posterior**: `model.sample(n=5000)` for posterior draws
- **Visualizations**: Use matplotlib with clear titles, labels, and grid

### Statistical Modeling
- **Always standardize** continuous predictors before regression
- **Use informative priors**: Based on standardized scale (e.g., `Normal(0, 0.5)` for slopes)
- **Check convergence**: Look at `Converged: True` in quap output
- **Interpret on original scale**: Transform back when presenting results

## Causal Inference (Chapter 6 Focus)

When working with DAGs and causal models:

1. **Draw the DAG first** - understand causal structure before modeling
2. **Identify backdoor paths** - find confounding paths from treatment to outcome
3. **Condition appropriately**:
   - Close backdoor paths by conditioning on confounders
   - Do NOT condition on mediators (blocks the effect)
   - Do NOT condition on colliders (creates spurious associations)
4. **Distinguish correlation from causation**

## Custom Tools

### `src/quap.py` - Quadratic Approximation

Python implementation of `rethinking::quap` for maximum a posteriori (MAP) estimation:

```python
from src.quap import quap

def neg_log_posterior(params):
    # Define negative log posterior
    return -(log_likelihood + log_prior)

model = quap(neg_log_posterior,
             initial_params=[0, 0, np.log(1)],
             param_names=['alpha', 'beta', 'log_sigma'])

model.summary()  # View results
model.sample(n=5000)  # Sample from posterior
model.transform_param('log_sigma', 'sigma', np.exp)  # Transform
```

## Data Sources

Primary datasets from Statistical Rethinking repository:
- Base URL: `https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/`
- Common datasets: `WaffleDivorce.csv`, `Howell1.csv`, `milk.csv`, `foxes.csv`

## Git Workflow

When making significant progress:
1. Update chapter-specific `CLAUDE.md` with new concepts covered
2. Update `TODO.md` with completed items and next steps
3. Commit with descriptive messages (see `.git/hooks/pre-commit` template)
4. Update this file if project structure changes

## References

- **Book**: Statistical Rethinking (2nd Ed) by Richard McElreath
- **R Package**: [rethinking](https://github.com/rmcelreath/rethinking)
- **PyMC Docs**: https://www.pymc.io/
- **ArviZ Docs**: https://python.arviz.org/

## For Claude Code

When starting a session:
1. Check `TODO.md` for current focus
2. Read relevant chapter `CLAUDE.md` for context
3. Check `MEMORY.md` for key learnings and patterns
4. Follow conventions above for consistency

---

*Last updated: 2026-02-21*
*Current focus: Chapter 6 Homework (6H3-6H7)*
