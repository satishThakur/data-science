# Bayesian Inference of LLM-as-a-Judge True Pass Rate

> *"The goal of Bayesian analysis is not to find the right answer,
> but to honestly quantify uncertainty — then let data update our beliefs."*

---

## Why This Problem Exists

Modern AI systems rely on **LLM-as-a-Judge** pipelines: an LLM automatically evaluates
whether a model response passes quality criteria. This is essential at scale — human
reviewers cannot manually inspect every production trace.

But here is the uncomfortable truth: **the judge is not perfect**. It has a false positive
rate (it passes traces that should fail) and a false negative rate (it fails traces that
should pass). When you report "our system passes X% of traces", you are actually reporting
the judge's noisy signal — not the true underlying quality of the system.

**The question we want to answer:**

> Given that the judge is imperfect — and we have measured its accuracy on a labelled
> validation set — what is the **true pass rate** of the system, along with honest
> uncertainty quantification?

This is exactly the kind of question Bayesian inference was designed for.

---

## Why Bayesian?

Classical (frequentist) statistics would give you a point estimate and a confidence interval.
But those answers come with hidden assumptions and are often misinterpreted.

Bayesian inference gives you something richer: a **full posterior distribution** over every
unknown quantity. It tells you not just "the true pass rate is probably around 70%" but
*exactly how uncertain* you should be, given your data and prior knowledge.

More importantly, Bayesian inference forces you to be explicit about:
- What you knew **before** seeing data (the prior)
- How the data updates that knowledge (the likelihood)
- What you now believe (the posterior)

This transparency is rare and valuable in production AI systems.

### Maximum Entropy: The Philosophical Foundation

Why do we use specific distributions (Beta, Binomial) rather than others?
The answer comes from the **Maximum Entropy principle** (E.T. Jaynes, 1957):

> Among all probability distributions consistent with your constraints,
> choose the one with the **maximum entropy** — the one that makes the
> fewest additional assumptions.

Applied here:
- **Binomial**: Given $n$ binary trials with fixed success probability $p$, the Binomial is
  the MaxEnt distribution. It assumes nothing beyond the trial structure.
- **Beta**: Given a probability $p \in [0,1]$ with known mean (and possibly variance),
  the Beta is the MaxEnt distribution. It is also the conjugate prior for the Binomial —
  meaning the posterior is also Beta, which is mathematically elegant.

These are not arbitrary choices. They are the **least informative** models consistent with
the problem structure. Bayesian inference under MaxEnt is epistemically honest.

---

## The Mathematical Model

### Setup

We have three unknown quantities we want to infer:

| Symbol | Meaning |
|--------|---------|
| **θ** | True pass rate — the probability that any given production trace *truly* passes |
| **TPR** | True Positive Rate — P(Judge=PASS given trace truly passes) |
| **TNR** | True Negative Rate — P(Judge=FAIL given trace truly fails) |

We observe:
- Validation set: $N_+$ human-labelled positives, $K_{PP}$ of which the judge agreed
- Validation set: $N_-$ human-labelled negatives, $K_{NN}$ of which the judge agreed
- Production: $N$ traces evaluated, $K$ of which the judge said PASS

### Parameter Taxonomy

| Type | Symbol | Description |
|------|--------|-------------|
| **Observed** | $K_{PP}, K_{NN}, K$ | Data we collected — fixed numbers |
| **Parameter** | $\theta, \text{TPR}, \text{TNR}$ | Latent — have a prior (`~`), inferred from data |
| **Deterministic** | $p_\text{judge}$ | Computed from parameters (`=`), no prior needed |

### The Bridge Equation

By the **Law of Total Probability**, the probability that the judge says PASS on any trace:

$$p_{\text{judge}} = P(\hat{Y}=+) = \underbrace{\theta \cdot \text{TPR}}_{\text{true pass, correctly identified}} + \underbrace{(1-\theta)\cdot(1-\text{TNR})}_{\text{true fail, incorrectly passed}}$$

This is the core insight: the observed judge pass rate is a **mixture** of two processes.
Without accounting for TPR and TNR, $K/N$ is a biased estimate of $\theta$.

### Generative Model

```
θ   ~ Beta(α_θ,   β_θ)       ← prior belief about true pass rate
TPR ~ Beta(α_TPR, β_TPR)     ← prior belief about judge sensitivity
TNR ~ Beta(α_TNR, β_TNR)     ← prior belief about judge specificity

K_PP ~ Binomial(N+,  TPR)                           ← validation: true positives
K_NN ~ Binomial(N-,  TNR)                           ← validation: true negatives
K    ~ Binomial(N,   θ·TPR + (1-θ)·(1-TNR))        ← production: judge passes
```

### Bayes' Theorem

$$P(\theta, \text{TPR}, \text{TNR} \mid \text{data}) \;\propto\;
\underbrace{P(K \mid N,\; p_{\text{judge}})}_{\text{production likelihood}}
\cdot
\underbrace{P(K_{PP} \mid N_+, \text{TPR}) \cdot P(K_{NN} \mid N_-, \text{TNR})}_{\text{validation likelihoods}}
\cdot
\underbrace{P(\theta)\cdot P(\text{TPR})\cdot P(\text{TNR})}_{\text{priors}}$$

### Marginalisation

The result of Bayes' theorem is a **joint posterior** over all three unknowns simultaneously.
To reason about any one parameter, we **marginalise out** the others:

$$P(\theta \mid \text{data}) = \iint P(\theta, \text{TPR}, \text{TNR} \mid \text{data})\; d\,\text{TPR}\; d\,\text{TNR}$$

$$P(\text{TPR} \mid \text{data}) = \iint P(\theta, \text{TPR}, \text{TNR} \mid \text{data})\; d\,\theta\; d\,\text{TNR}$$

$$P(\text{TNR} \mid \text{data}) = \iint P(\theta, \text{TPR}, \text{TNR} \mid \text{data})\; d\,\theta\; d\,\text{TPR}$$

Each marginal posterior answers: "what do we know about *this* parameter, having integrated
over all uncertainty about the other two?" We do not fix TPR and TNR at point estimates —
we carry their full uncertainty forward into our estimate of θ.

In grid approximation, these integrals become literal array summations over the
corresponding dimensions of the 3D posterior array. Think of it as looking at the same
mass of probability from different directions.

### Identifiability Condition

There is a subtle but critical condition for the model to be informative:

$$\text{TPR} + \text{TNR} > 1$$

The judge must be **better than random**. If TPR + TNR = 1, every value of θ produces
the same $p_\text{judge}$ — the production data carries no information about θ.
The validation set is what makes this model identifiable.

---

## What the Notebook Covers

### Two Inference Methods

**1. Grid Approximation** — The Glass Box

We discretise the parameter space into a 3D grid (θ × TPR × TNR) and compute the
posterior probability at every grid point. Marginalisation becomes literal array summation.
This method is slow for large grids but **completely transparent** — you can see every step.

It teaches:
- What a joint posterior looks like
- How marginalisation works (sum over dimensions)
- Why log-space computation matters (numerical stability via logsumexp)
- The role of each likelihood term

**2. MCMC with PyMC** — The Fast Lane

The No-U-Turn Sampler (NUTS) explores the posterior efficiently via Hamiltonian Monte Carlo.
It scales to high dimensions where grid approximation breaks down. The model definition
mirrors the mathematical notation almost one-to-one.

Includes:
- Convergence diagnostics: r_hat, ESS, trace plots
- 89% credible intervals printed in plain English
- Grid vs MCMC overlay comparison (both should agree)

### Predictive Checks

**Prior Predictive Check** — Before seeing data, what counts K does the prior imply?
- Flat Beta(1,1)³ priors induce a bell-shaped distribution on $p_\text{judge}$ centred at 0.5
  (not flat!) — this is a non-obvious consequence worth understanding
- All three observables plotted: K_prod, K_PP, K_NN

**Posterior Predictive Check** — After fitting, simulate new K values from the posterior.
They should cluster around the observed K. If the observed value falls in the tail, the
model is misspecified. Includes plain-English verdicts for all three observables.

### Two Hypotheses

**Hypothesis A — Does Prior Choice Matter with Small Data?**

With only N_val=10 and N_prod=40:
- Flat prior Beta(1,1): TPR and TNR poorly estimated → uncertainty propagates into θ
- Informative prior Beta(8,2): anchors TPR/TNR → tighter posterior on θ

Prediction: informative prior produces a narrower, better-centred θ posterior.
The TPR posterior is shown alongside to illustrate *why* — the prior anchors the nuisance.

**Hypothesis B — Does Data Size Make Priors Irrelevant?**

With N_val=200 and N_prod=1000:
- Flat and informative priors converge to the same posterior
- CI widths narrow; both means land near truth

Prediction: the two posteriors are visually indistinguishable — data overwhelms the prior.
This is the Bayesian consistency theorem made concrete.

### Summary Table

After both hypotheses, a full comparison table shows posterior mean, 89% CI, CI width,
truth value, and whether truth falls inside the CI — across all four scenarios.

---

## Interactive Scenario Selector

The notebook includes an **ipywidgets dropdown** immediately after the master config cell.
Select a scenario and all subsequent cells update when re-run:

| Scenario | N_val | N_prod | Priors | Purpose |
|---|---|---|---|---|
| `baseline` | 50/50 | 200 | Flat | Default starting point |
| `small_flat` | 10/10 | 40 | Flat | H_A reference |
| `small_informative` | 10/10 | 40 | Informative | H_A experiment |
| `large_flat` | 200/200 | 1000 | Flat | H_B reference |
| `large_informative` | 200/200 | 1000 | Informative | H_B experiment |

---

## Key Concepts Reference

### Credible Intervals vs Confidence Intervals

A **89% Bayesian credible interval** [a, b] means:
> "Given the data and prior, there is an 89% probability that the true parameter lies in [a, b]."

This is the natural interpretation most people *want* from a "confidence interval" — but
frequentist confidence intervals do not actually say this. Bayesian credible intervals do.
The notebook uses 89% following McElreath's convention in Statistical Rethinking.

### Induced Prior on p_judge

An important non-obvious result: even if all three parameter priors are flat Beta(1,1),
the **induced prior on $p_\text{judge}$** is bell-shaped and centred at 0.5 — not flat.
This arises from the non-linear mixing in the bridge equation. The prior predictive check
visualises this directly.

### Prior Sensitivity

The degree to which the posterior depends on the prior is called **prior sensitivity**.
- High sensitivity: small data, model poorly identified → priors matter a lot
- Low sensitivity: large data, model well-identified → posteriors are prior-robust

Hypothesis A and B bracket this behaviour at the two extremes.

---

## References

### Books
- **Statistical Rethinking** (2nd Ed.) — Richard McElreath. Primary reference. Chapters 2–6.
- **Bayesian Data Analysis** (3rd Ed.) — Gelman, Carlin, Stern, Dunson, Vehtari, Rubin.
- **Probability Theory: The Logic of Science** — E.T. Jaynes. MaxEnt foundation.
- **Think Bayes** — Allen Downey. Accessible Python-first introduction.

### Papers & Articles
- Vehtari et al. (2017) — *Practical Bayesian model evaluation using LOO and WAIC*.
- Gelman & Shalizi (2013) — *Philosophy and the practice of Bayesian statistics*.

### Software
- [PyMC](https://www.pymc.io/) — Probabilistic programming in Python
- [ArviZ](https://python.arviz.org/) — Bayesian diagnostics and visualization
- [ipywidgets](https://ipywidgets.readthedocs.io/) — Interactive Jupyter widgets

---

## Quick Start

```bash
# From project root
uv sync
uv run jupyter lab notebooks/llm-judge/llm_judge_bayesian.ipynb
```

Change scenario via the **dropdown widget** in the notebook (no code editing needed),
or set `ACTIVE_SCENARIO` directly in the master config cell:

```python
ACTIVE_SCENARIO = "baseline"   # or: small_flat, small_informative, large_flat, large_informative
```

---

*Created: 2026-03-28*
*Last updated: 2026-03-29*
*Part of: Statistical Rethinking / Bayesian Data Analysis learning project*
