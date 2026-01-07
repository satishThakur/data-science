# Bayesian Data Analysis & Statistical Rethinking

A journey through Bayesian statistics and data analysis, following Richard McElreath's **Statistical Rethinking**.

## Learning Path

Follow this sequence for the best learning experience:

### 1. Foundation: Theory & Concepts
**Start here:** [`bayesian_notes.md`](./bayesian_notes.md)

Core concepts covered:
- Probability as degree of plausibility (Laplace-Jeffreys-Cox-Jaynes interpretation)
- Prior, Likelihood, Posterior, and Prediction
- Discrete and continuous parameter cases
- Marginalisation and joint distributions
- Two views of likelihood (inference vs prediction)
- Predictive distributions (prior and posterior)
- Complete worked example: Panda Problem with genetic test

### 2. Chapter 2: Complete Implementation
**Next:** [`notebooks/chapter2_bayesian_inference_updated.ipynb`](./notebooks/chapter2_bayesian_inference_updated.ipynb)

Interactive notebook with:
- All concepts from bayesian_notes.md with code and visualizations
- Competing conjectures demonstration
- Discrete and continuous Bayes theorem
- The Panda Problem: Sequential updating with genetic test
  - Prior (0.5/0.5) → After twins (0.33/0.67) → After test (0.533/0.467)
- Complete visualizations of parameter and prediction evolution

### 3. Sequential Updating Exploration
**Then:** [`notebooks/sequential_updates.ipynb`](./notebooks/sequential_updates.ipynb)

Deep dive into:
- How beliefs evolve with each observation
- Sequential Bayesian updating in action
- Real-time belief refinement

### 4. Bernoulli Distribution
**Next:** [`notebooks/bernoulli.ipynb`](./notebooks/bernoulli.ipynb)

Understanding:
- Binary outcomes (success/failure)
- Bernoulli likelihood
- Conjugate priors (Beta distribution)
- Analytical solutions

### 5. Sampling from the Posterior
**Then:** [`notebooks/sampling_exercise_1.ipynb`](./notebooks/sampling_exercise_1.ipynb)

Chapter 3 exercises:
- Grid approximation sampling
- Understanding posterior distributions through samples
- Percentile intervals and HPDIs
- Loss functions and point estimates

### 6. Univariate Gaussian Models
**Next:** [`notebooks/univariate_gaussian.ipynb`](./notebooks/univariate_gaussian.ipynb)

Chapter 4 Part 1:
- Gaussian/Normal distribution as likelihood
- Height modeling examples
- Prior predictive checks
- Posterior inference for μ and σ
- Introduction to quap (quadratic approximation)

### 7. Polynomial Regression
**Then:** [`notebooks/polynomial_regression.ipynb`](./notebooks/polynomial_regression.ipynb)

Chapter 4 Part 2 - Curved lines:
- When linear models aren't enough
- Polynomial models (degree 1-6)
- Critical importance of standardization
- Model comparison across degrees
- Overfitting dangers and extrapolation problems
- Why high-degree polynomials are unstable

### 8. Splines
**Next:** [`notebooks/splines.ipynb`](./notebooks/splines.ipynb)

Chapter 4 Part 3 - Better curves:
- Problems with polynomials (global influence, poor extrapolation)
- B-spline basis functions
- Knot placement strategies
- Local influence and stability
- Splines vs polynomials comparison
- Why splines are superior for curved relationships

### 9. Chapter 4 Exercises
**Finally:** [`notebooks/chapter4_exercise.ipynb`](./notebooks/chapter4_exercise.ipynb)

Practice problems covering:
- Linear models and prior specification
- Model fitting with quap
- Posterior analysis
- Model checking and validation
- Easy, Medium, and Hard problems from the book

---

## Repository Structure

```
.
├── bayesian_notes.md                    # Core theory notes
├── notebooks/
│   ├── chapter2_bayesian_inference_updated.ipynb  # Chapter 2 complete
│   ├── sequential_updates.ipynb         # Sequential updating
│   ├── bernoulli.ipynb                  # Bernoulli distribution
│   ├── sampling_exercise_1.ipynb        # Chapter 3 exercises
│   ├── univariate_gaussian.ipynb        # Chapter 4 Part 1 - Linear models
│   ├── polynomial_regression.ipynb      # Chapter 4 Part 2 - Polynomials
│   ├── splines.ipynb                    # Chapter 4 Part 3 - Splines
│   ├── chapter4_exercise.ipynb          # Chapter 4 exercises
│   └── test.ipynb                       # Environment test
├── src/
│   └── quap.py                          # Quadratic approximation implementation
├── data/                                # Datasets
└── pyproject.toml                       # Dependencies
```

---

## Getting Started

### Prerequisites

This project uses `uv` for dependency management with Python 3.11+.

### Installation

```bash
# Install dependencies
uv sync

# Start Jupyter Lab
jupyter lab
```

### Running Notebooks

```bash
# Navigate to notebooks directory
cd notebooks

# Open specific notebook
jupyter notebook chapter2_bayesian_inference_updated.ipynb
```

---

## Key Dependencies

- **PyMC 5.26.1+**: Probabilistic programming framework
- **ArviZ 0.22.0+**: Bayesian data analysis and visualization
- **NumPy, Pandas, Matplotlib**: Core data science stack
- **SciPy**: Statistical functions
- **JupyterLab 4.5.0+**: Interactive notebooks

---

## What You'll Learn

### Chapter 2: Small Worlds and Large Worlds
- Bayesian inference fundamentals
- Prior, likelihood, and posterior
- Marginalisation and joint distributions
- Predictive distributions
- Sequential updating

### Chapter 3: Sampling the Imaginary
- Grid approximation
- Sampling from posterior
- Summarizing posterior distributions
- Posterior predictive checks

### Chapter 4: Geocentric Models (Linear Regression)
- Gaussian models
- Linear regression basics
- Prior specification
- Quadratic approximation (quap)
- Polynomial regression and standardization
- B-splines for curved relationships
- Model comparison and overfitting
- Posterior analysis and visualization

---

## Philosophy

This repository follows a **hands-on, build-from-scratch** approach:

1. **Understand concepts deeply** - Start with theory (bayesian_notes.md)
2. **Implement from scratch** - Build your own inference tools (src/quap.py)
3. **Practice extensively** - Work through all exercises
4. **Visualize everything** - See what the math means
5. **Learn sequentially** - Each notebook builds on previous ones

---

## Notes

- **bayesian_notes.md** is the theoretical foundation - read this first!
- All notebooks include detailed explanations and visualizations
- The `src/quap.py` module is reusable across chapters
- Notebooks are meant to be run in order for best understanding

---

## Resources

- **Book**: [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/) by Richard McElreath
- **Lectures**: [2023 Lecture Series](https://github.com/rmcelreath/stat_rethinking_2023)
- **Python Port**: Working through concepts using Python instead of R

---

## License

Educational project for learning Bayesian data analysis.

---

**Happy Learning!**

Start with [`bayesian_notes.md`](./bayesian_notes.md) and work your way through the notebooks sequentially.
