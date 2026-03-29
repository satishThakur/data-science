"""
Explore — run a single scenario (preset or custom) and inspect the posterior.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from llm_judge.config import SCENARIOS, PRESET_NAMES, PARAM_LABELS
from llm_judge.inference import load_preset, run_custom_scenario, convergence_summary
from llm_judge.plots import (plot_prior_predictive, plot_posteriors,
                              posterior_ci_text, plot_ppc)

st.set_page_config(page_title="Explore — LLM-Judge", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

# ── Session state guards ───────────────────────────────────────────────────────
for key in ["custom_cfg", "custom_data", "custom_idata", "custom_ran"]:
    if key not in st.session_state:
        st.session_state[key] = None
st.session_state.setdefault("custom_ran", False)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🔬 Explore")
mode = st.sidebar.radio("Mode", ["Preset", "Custom"], horizontal=True)

if mode == "Preset":
    scenario_name = st.sidebar.selectbox(
        "Scenario",
        PRESET_NAMES,
        format_func=lambda n: f"{n}  —  {SCENARIOS[n]['description']}",
    )
else:
    st.sidebar.markdown("**Simulation parameters**")
    n_val  = st.sidebar.slider("Validation set size (each class)", 5, 500, 50, step=5)
    n_prod = st.sidebar.slider("Production traces N", 20, 2000, 200, step=20)
    st.sidebar.markdown("**True parameters** (ground truth, known because we simulate)")
    true_theta = st.sidebar.slider("True θ", 0.30, 0.99, 0.70, step=0.01)
    true_tpr   = st.sidebar.slider("True TPR", 0.50, 0.99, 0.82, step=0.01)
    true_tnr   = st.sidebar.slider("True TNR", 0.50, 0.99, 0.79, step=0.01)
    st.sidebar.markdown("**Priors**")
    theta_prior = st.sidebar.radio("θ prior", ["Flat  Beta(1,1)", "Informative  Beta(3,2)"])
    judge_prior = st.sidebar.radio("TPR/TNR prior", ["Flat  Beta(1,1)", "Informative  Beta(8,2)"])

    t_pr = (1, 1) if "Flat" in theta_prior else (3, 2)
    j_pr = (1, 1) if "Flat" in judge_prior else (8, 2)

    custom_cfg = {
        "description": "Custom scenario",
        "true_theta":  true_theta, "true_tpr": true_tpr, "true_tnr": true_tnr,
        "n_val_pos":   n_val,      "n_val_neg": n_val,   "N_prod":   n_prod,
        "theta_prior": t_pr,       "tpr_prior": j_pr,    "tnr_prior": j_pr,
    }

# ── Main header ────────────────────────────────────────────────────────────────
st.title("🔬 Explore a Scenario")

# ── Load / run ─────────────────────────────────────────────────────────────────
if mode == "Preset":
    # Load cached preset — instant
    @st.cache_resource
    def _load(name):
        return load_preset(name)

    with st.spinner("Loading pre-computed results..."):
        cfg, data, idata = _load(scenario_name)

    st.success(f"**{scenario_name}** — {cfg['description']}  "
               f"(N_val={cfg['n_val_pos']}, N_prod={cfg['N_prod']})")

else:
    # Custom — needs MCMC on demand
    run_btn = st.button("▶  Run Bayesian Inference", type="primary", width="content")

    if run_btn:
        t0 = time.time()
        phases = {
            "building":  "🔧  Building PyMC model...",
            "sampling":  f"⛓  Sampling MCMC — 2,000 draws × 4 chains...",
            "ppc":       "🔁  Computing posterior predictive...",
        }

        with st.status("Running Bayesian inference...", expanded=True) as status:
            def cb(phase):
                st.write(phases[phase])

            cfg, data, idata = run_custom_scenario(custom_cfg, seed=42, progress_callback=cb)
            elapsed = time.time() - t0
            status.update(label=f"✅  Done in {elapsed:.1f}s — 8,000 posterior samples",
                          state="complete", expanded=False)

        st.session_state["custom_cfg"]   = cfg
        st.session_state["custom_data"]  = data
        st.session_state["custom_idata"] = idata
        st.session_state["custom_ran"]   = True

    if not st.session_state["custom_ran"]:
        st.info("Configure the scenario in the sidebar and click **Run Bayesian Inference**.")
        st.stop()

    cfg   = st.session_state["custom_cfg"]
    data  = st.session_state["custom_data"]
    idata = st.session_state["custom_idata"]

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊  Posteriors & CI", "🔮  Prior Predictive", "✅  PPC"])

with tab1:
    st.subheader("Posterior Distributions")
    st.pyplot(plot_posteriors(cfg, idata, scenario_name if mode == "Preset" else "custom"),
              width="stretch")

    st.divider()
    st.subheader("89% Credible Intervals — Plain English")
    ci_rows = posterior_ci_text(cfg, idata)
    for row in ci_rows:
        icon = "✅" if row["inside"] else "⚠️"
        st.markdown(f"{icon}  {row['text']}")
        st.caption(f"  mean = {row['mean']:.3f}   truth = {row['truth']}   "
                   f"{'inside CI' if row['inside'] else 'OUTSIDE CI'}")

    st.divider()
    with st.expander("🔍 Convergence diagnostics"):
        diag = convergence_summary(idata)
        diag_rows = []
        for var, d in diag.items():
            diag_rows.append({
                "Parameter": PARAM_LABELS[var],
                "Mean":      f"{d['mean']:.4f}",
                "89% CI":    f"[{d['lo']:.4f}, {d['hi']:.4f}]",
                "r̂ (r_hat)": f"{d['rhat']:.4f}",
                "ESS bulk":  f"{d['ess']:.0f}",
                "Status":    "✅ OK" if d["ok"] else "⚠️ Check",
            })
        st.dataframe(pd.DataFrame(diag_rows).set_index("Parameter"), width="stretch")

        all_ok = all(d["ok"] for d in diag.values())
        if all_ok:
            st.success("All chains converged. r̂ < 1.01 and ESS > 400 for all parameters.")
        else:
            st.warning("Some convergence issues detected. Results may be unreliable.")

with tab2:
    st.subheader("Prior Predictive Check")
    st.markdown("""
Before seeing any data, what do the priors imply about the observable counts?
A flat Beta(1,1) prior on all parameters induces a **bell-shaped** distribution
on p_judge centred at 0.5 — not flat. This is a non-obvious consequence of the
non-linear bridge equation.
""")
    st.pyplot(plot_prior_predictive(cfg), width="stretch")

with tab3:
    st.subheader("Posterior Predictive Check")
    st.markdown("""
After fitting, simulate new data from the posterior. If the observed counts fall
comfortably within the simulated distribution, the model is well-specified.
""")
    st.pyplot(plot_ppc(cfg, data, idata), width="stretch")
