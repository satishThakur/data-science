"""
Compare — overlay two preset scenarios side-by-side.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from llm_judge.config import SCENARIOS, PRESET_NAMES, PARAM_LABELS
from llm_judge.inference import load_preset, convergence_summary
from llm_judge.plots import plot_comparison, comparison_verdict

st.set_page_config(page_title="Compare — LLM-Judge", page_icon="⚖️",
                   layout="wide", initial_sidebar_state="expanded")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚖️ Compare")
st.sidebar.markdown("Select two preset scenarios to compare their posteriors side-by-side.")

name_a = st.sidebar.selectbox(
    "Scenario A", PRESET_NAMES, index=1,           # default: small_flat
    format_func=lambda n: f"{n}  —  {SCENARIOS[n]['description']}",
    key="sel_a",
)
name_b = st.sidebar.selectbox(
    "Scenario B", PRESET_NAMES, index=2,           # default: small_informative
    format_func=lambda n: f"{n}  —  {SCENARIOS[n]['description']}",
    key="sel_b",
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: try **small_flat** vs **small_informative** to see Hypothesis A, "
                   "or **large_flat** vs **large_informative** for Hypothesis B.")

# ── Load presets ───────────────────────────────────────────────────────────────
@st.cache_resource
def _load(name):
    return load_preset(name)

with st.spinner("Loading scenarios..."):
    cfg_a, data_a, idata_a = _load(name_a)
    cfg_b, data_b, idata_b = _load(name_b)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("⚖️ Compare Two Scenarios")

if name_a == name_b:
    st.warning("Both panels show the same scenario. Select different scenarios to compare.")

# ── Scenario metadata cards ────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

def _meta_card(col, name, cfg, color):
    with col:
        st.markdown(f"### {name}")
        prior = "Flat  Beta(1,1)" if cfg["theta_prior"] == (1, 1) else "Informative  Beta(8,2)"
        m1, m2, m3 = st.columns(3)
        m1.metric("N_val (±)", cfg["n_val_pos"])
        m2.metric("N_prod", cfg["N_prod"])
        m3.metric("Prior", prior)

_meta_card(col_a, name_a, cfg_a, "#2980B9")
_meta_card(col_b, name_b, cfg_b, "#E67E22")

st.divider()

# ── Overlay plot ───────────────────────────────────────────────────────────────
st.subheader("Posterior Overlay")
fig = plot_comparison(cfg_a, idata_a, name_a, cfg_b, idata_b, name_b)
st.pyplot(fig, width="stretch")

# ── Verdict ────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Verdict")

v = comparison_verdict(idata_a, name_a, idata_b, name_b, cfg_a["true_theta"])

col1, col2 = st.columns(2)
with col1:
    delta = f"{v['b']['width'] - v['a']['width']:+.3f}"
    st.metric(f"{name_a} — CI width (θ)", f"{v['a']['width']:.3f}",
              help="89% credible interval width")
    inside = "✅ truth inside CI" if v["inside_a"] else "⚠️ truth outside CI"
    st.caption(f"mean={v['a']['mean']:.3f}   [{v['a']['lo']:.3f}, {v['a']['hi']:.3f}]   {inside}")

with col2:
    delta_b = f"{v['a']['width'] - v['b']['width']:+.3f}"
    st.metric(f"{name_b} — CI width (θ)", f"{v['b']['width']:.3f}",
              delta=delta_b, delta_color="inverse",
              help="Positive delta = narrower than A = better")
    inside_b = "✅ truth inside CI" if v["inside_b"] else "⚠️ truth outside CI"
    st.caption(f"mean={v['b']['mean']:.3f}   [{v['b']['lo']:.3f}, {v['b']['hi']:.3f}]   {inside_b}")

# Verdict box
if "narrower" in v["verdict"]:
    st.success(v["verdict"])
else:
    st.info(v["verdict"])

st.divider()

# ── Full summary table ─────────────────────────────────────────────────────────
st.subheader("Full Summary Table")

rows = []
for name, cfg, idata in [(name_a, cfg_a, idata_a), (name_b, cfg_b, idata_b)]:
    diag = convergence_summary(idata)
    prior = "flat" if cfg["theta_prior"] == (1, 1) else "informative"
    for var in ["theta", "tpr", "tnr"]:
        d = diag[var]
        rows.append({
            "Scenario":  name,
            "Parameter": var,
            "Prior":     prior,
            "N_prod":    cfg["N_prod"],
            "Mean":      f"{d['mean']:.4f}",
            "89% CI":    f"[{d['lo']:.3f}, {d['hi']:.3f}]",
            "CI width":  f"{d['hi'] - d['lo']:.3f}",
            "Truth":     {"theta": cfg["true_theta"],
                          "tpr":   cfg["true_tpr"],
                          "tnr":   cfg["true_tnr"]}[var],
            "Inside":    "✅" if d["lo"] <= {"theta": cfg["true_theta"],
                                              "tpr":   cfg["true_tpr"],
                                              "tnr":   cfg["true_tnr"]}[var] <= d["hi"] else "⚠️",
        })

st.dataframe(pd.DataFrame(rows).set_index(["Scenario", "Parameter"]),
             width="stretch")
