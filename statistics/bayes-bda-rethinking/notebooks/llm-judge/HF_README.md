---
title: LLM Judge Bayesian Inference
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: mit
short_description: Infer the true pass rate of an LLM-as-a-Judge using Bayesian inference
---

# Bayesian Inference of LLM-as-a-Judge True Pass Rate

Interactive Streamlit app for inferring the **true pass rate θ** of an LLM-as-a-Judge
evaluation system, accounting for judge imperfection.

## What This Does

LLM judges are imperfect — they have false positive and false negative rates.
Naively reporting K/N (observed passes / total traces) gives a **biased estimate**
of the true system quality. This app uses Bayesian inference to recover the true
underlying pass rate, along with honest uncertainty quantification.

**Model**: Beta priors on θ, TPR, TNR → Binomial likelihoods → MCMC via PyMC NUTS

## Pages

- **🏠 Home** — Problem statement, bridge equation, generative model DAG
- **🔬 Explore** — Run any of 5 preset scenarios (instant) or define a custom scenario
- **⚖️ Compare** — Overlay two scenarios side-by-side to test hypotheses

## Key Features

- 5 pre-computed preset scenarios (no waiting)
- Custom inference with live MCMC progress indicator
- 89% credible intervals with plain-English interpretation
- Prior predictive and posterior predictive checks
- KDE-based posterior overlays for clean visual comparison

## Local Development

```bash
git clone https://huggingface.co/spaces/<your-username>/llm-judge-bayes
cd llm-judge-bayes
pip install -r requirements.txt
streamlit run app.py
```

## References

- [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/) — Richard McElreath
- [PyMC](https://www.pymc.io/) — Probabilistic programming in Python
- [ArviZ](https://python.arviz.org/) — Bayesian diagnostics and visualization
