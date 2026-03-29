"""
LLM-as-a-Judge Bayesian Inference — Home

Run from the llm-judge directory:
    uv run streamlit run app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from llm_judge.plots import plot_dag
from llm_judge.config import SCENARIOS

st.set_page_config(
    page_title="LLM-Judge Bayesian Inference",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    code { background: #f0f2f6; padding: 2px 6px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚖️ LLM-Judge Inference")
st.sidebar.markdown("""
**Pages**
- 🏠 Home — you are here
- 🔬 Explore — run a single scenario
- ⚖️ Compare — A vs B side-by-side
""")

# ── Hero ───────────────────────────────────────────────────────────────────────
st.title("Bayesian Inference of LLM-as-a-Judge True Pass Rate")
st.caption("From First Principles — Grid Approximation · MCMC · Prior Sensitivity · Hypothesis Testing")

st.divider()

# ── Two-column intro ───────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("The Problem")
    st.markdown("""
Modern AI systems rely on **LLM-as-a-Judge** pipelines: an LLM automatically
evaluates whether a model response passes quality criteria.

But here is the uncomfortable truth: **the judge is not perfect**.
It has a false positive rate (passes traces that should fail)
and a false negative rate (fails traces that should pass).

When you report *"our system passes 70% of traces"*, you are reporting the
judge's noisy signal — not the true underlying quality of the system.

**The question we want to answer:**
> Given that the judge is imperfect — and we have measured its accuracy
> on a labelled validation set — what is the **true pass rate θ**,
> along with honest uncertainty quantification?
""")

    st.subheader("The Bridge Equation")
    st.markdown("By the Law of Total Probability:")
    st.latex(r"""
        p_{\text{judge}} \;=\; \theta \cdot \text{TPR}
        \;+\; (1-\theta)\cdot(1-\text{TNR})
    """)
    st.markdown("""
| Term | Meaning |
|------|---------|
| **θ** | True pass rate — what we want to recover |
| **TPR** | P(Judge=PASS \| trace truly passes) — from validation set |
| **TNR** | P(Judge=FAIL \| trace truly fails) — from validation set |
| **p_judge** | Observable judge pass rate — what we actually measure |

The naive estimate $K/N$ conflates judge errors with true signal.
Bayesian inference untangles them.
""")

with right:
    st.subheader("Generative Model (DAG)")
    fig = plot_dag()
    st.pyplot(fig, width="stretch")

st.divider()

# ── Scenarios table ────────────────────────────────────────────────────────────
st.subheader("Available Scenarios")
st.markdown("Use the **Explore** page to run any scenario, or **Compare** to overlay two.")

import pandas as pd
rows = []
for name, s in SCENARIOS.items():
    prior = "Flat Beta(1,1)" if s["theta_prior"] == (1, 1) else "Informative"
    rows.append({
        "Scenario":     name,
        "Description":  s["description"],
        "N_val (±)":    s["n_val_pos"],
        "N_prod":       s["N_prod"],
        "Priors":       prior,
        "True θ":       s["true_theta"],
    })

st.dataframe(pd.DataFrame(rows).set_index("Scenario"), width="stretch")

st.divider()
st.subheader("Key Concepts")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
**89% Credible Interval**

A Bayesian credible interval [a, b] means:
> "There is an 89% probability the true
> parameter lies between a and b."

This is what people *want* confidence
intervals to mean — but frequentist CIs
don't say this. Bayesian ones do.
""")
with c2:
    st.markdown("""
**Prior Sensitivity**

With *small data*, priors matter a lot.
An informative prior on TPR/TNR anchors
the nuisance parameters and produces
a tighter posterior on θ.

With *large data*, the likelihood
dominates — prior choice becomes
irrelevant (Bayesian consistency theorem).
""")
with c3:
    st.markdown("""
**Marginalisation**

We don't fix TPR and TNR at point
estimates. We carry their full
uncertainty forward and *integrate
over* all plausible values.

This is what makes the inference
honest. The posterior on θ reflects
every source of uncertainty.
""")
