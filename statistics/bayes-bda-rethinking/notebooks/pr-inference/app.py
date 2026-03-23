"""
PR Inference Dashboard
======================
Streamlit app for exploring developer PR rates using Bayesian models.

Run from project root:
    streamlit run notebooks/pr-inference/app.py
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent.parent.parent / 'data'
DESIGNATIONS = ['aSDE', 'SDE-1', 'SDE-2', 'SDE-3', 'SDE-4']
GRAND_MEAN   = np.log(8)

# ── Load data (cached — only runs once) ───────────────────────────────────────
@st.cache_resource
def load_all():
    df           = pd.read_csv(DATA_DIR / 'pr_simulated.csv')
    idata_m2     = az.from_netcdf(DATA_DIR / 'idata_m2.nc')
    idata_simple = az.from_netcdf(DATA_DIR / 'idata_simple.nc')
    return df, idata_m2, idata_simple

df, idata_m2, idata_simple = load_all()

# ── Precompute posteriors (cached) ────────────────────────────────────────────
@st.cache_resource
def precompute(_df, _idata_m2, _idata_simple):
    dev_df  = _df.drop_duplicates('developer_id').sort_values('developer_id').reset_index(drop=True)
    N_DEVS  = len(dev_df)
    N_TEAMS = _df['team_id'].nunique()

    # Empirical: total PRs / total exposure per developer
    emp = (_df.groupby('developer_id')['pr_count'].sum()
           / _df.groupby('developer_id')['exposure'].sum()).values

    # m_simple posterior — alpha shape: (S, N_DEVS)
    alpha_s      = _idata_simple.posterior['alpha'].values.reshape(-1, N_DEVS)
    lam_s_med    = np.median(np.exp(alpha_s), axis=0)
    lam_s_lo     = np.percentile(np.exp(alpha_s), 5,  axis=0)
    lam_s_hi     = np.percentile(np.exp(alpha_s), 95, axis=0)

    # m2 posterior — developer level
    alpha_m2     = _idata_m2.posterior['alpha'].values.reshape(-1, N_DEVS)
    lam_m2_med   = np.median(np.exp(alpha_m2), axis=0)
    lam_m2_lo    = np.percentile(np.exp(alpha_m2), 5,  axis=0)
    lam_m2_hi    = np.percentile(np.exp(alpha_m2), 95, axis=0)

    # m2 posterior — designation and team level
    mu_org_s     = _idata_m2.posterior['mu_org'].values.flatten()
    delta_s      = _idata_m2.posterior['delta'].values.reshape(-1, len(DESIGNATIONS))
    gamma_s      = _idata_m2.posterior['gamma'].values.reshape(-1, N_TEAMS)
    sigma_dev_s  = _idata_m2.posterior['sigma_dev'].values.flatten()
    lam_desig    = np.exp(mu_org_s[:, None] + delta_s)   # (S, 5)
    lam_team     = np.exp(mu_org_s[:, None] + gamma_s)   # (S, 15)

    return (dev_df, N_DEVS, N_TEAMS, emp,
            alpha_s, lam_s_med, lam_s_lo, lam_s_hi,
            alpha_m2, lam_m2_med, lam_m2_lo, lam_m2_hi,
            mu_org_s, delta_s, gamma_s, sigma_dev_s,
            lam_desig, lam_team)

(dev_df, N_DEVS, N_TEAMS, emp,
 alpha_s, lam_s_med, lam_s_lo, lam_s_hi,
 alpha_m2, lam_m2_med, lam_m2_lo, lam_m2_hi,
 mu_org_s, delta_s, gamma_s, sigma_dev_s,
 lam_desig, lam_team) = precompute(df, idata_m2, idata_simple)

# ── Page layout ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PR Inference", layout="wide")
st.title("PR Inference Dashboard")
st.markdown(
    "Bayesian analysis of developer PR rates. "
    "**Empirical** = raw MLE | **m_simple** = per-developer Bayesian | "
    "**m2** = full hierarchy (org + designation + team + developer)"
)

tab1, tab2 = st.tabs(["Org Overview", "Developer Lookup"])

# ── Tab 1: Org Overview ───────────────────────────────────────────────────────
with tab1:

    # ── Designation rates ─────────────────────────────────────────────────────
    st.subheader("Designation Rates")
    st.markdown(
        "**m2** (coloured): posterior median + 90% CI. "
        "**Empirical** (grey diamonds): raw mean per designation."
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, desig in enumerate(DESIGNATIONS):
        mask       = dev_df['designation'] == desig
        emp_desig  = emp[mask]

        # Empirical scatter + mean
        ax.scatter([i - 0.2] * len(emp_desig), emp_desig,
                   alpha=0.25, color='grey', s=15, zorder=2)
        ax.scatter(i - 0.2, emp_desig.mean(),
                   color='grey', s=100, marker='D', zorder=4, label='Empirical' if i == 0 else '')

        # m2 posterior
        med     = np.median(lam_desig[:, i])
        lo, hi  = np.percentile(lam_desig[:, i], [5, 95])
        ax.errorbar(i + 0.2, med, yerr=[[med - lo], [hi - med]],
                    fmt='o', color=f'C{i}', capsize=5, markersize=9,
                    label=f'{desig} m2' if i == 0 else desig, zorder=5)

    ax.axhline(np.exp(GRAND_MEAN), color='black', lw=1.2,
               linestyle='--', label='Grand mean (8/mo)')
    ax.set_xticks(range(len(DESIGNATIONS)))
    ax.set_xticklabels(DESIGNATIONS, fontsize=11)
    ax.set_ylabel("Expected PRs / month")
    ax.set_title("Designation rates: Empirical (grey) vs m2 posterior 90% CI")
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Team rates ────────────────────────────────────────────────────────────
    st.subheader("Team Rates (m2 posterior)")
    st.markdown("Colour = tier: **blue** Balanced | **orange** Junior-heavy | **green** Mostly-junior")

    TIER_COLOR = {'balanced': 'C0', 'junior_heavy': 'C1', 'mostly_junior': 'C2'}
    team_names = [f'Team-{i+1}' for i in range(N_TEAMS)]

    fig, ax = plt.subplots(figsize=(13, 4))

    for i, tname in enumerate(team_names):
        tier_val = dev_df[dev_df['team'] == tname]['tier'].iloc[0]
        color    = TIER_COLOR[tier_val]
        med      = np.median(lam_team[:, i])
        lo, hi   = np.percentile(lam_team[:, i], [5, 95])
        ax.errorbar(i, med, yerr=[[med - lo], [hi - med]],
                    fmt='o', color=color, capsize=4, markersize=8, zorder=3)
        # Empirical team mean
        emp_team = emp[dev_df['team'] == tname]
        ax.scatter(i, emp_team.mean(), marker='D', color='grey',
                   s=60, alpha=0.6, zorder=2)

    ax.axhline(np.exp(GRAND_MEAN), color='black', lw=1.2, linestyle='--', label='Grand mean')
    ax.axvline(4.5,  color='black', lw=1, linestyle=':')
    ax.axvline(9.5,  color='black', lw=1, linestyle=':')
    ax.set_xticks(range(N_TEAMS))
    ax.set_xticklabels(team_names, rotation=45, fontsize=9)
    ax.set_ylabel("Expected PRs / month")
    ax.set_title("Team rates: m2 posterior 90% CI (grey diamond = empirical team mean)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Sigma comparison ──────────────────────────────────────────────────────
    st.subheader("Model Comparison: sigma_dev")
    st.markdown(
        "m_simple absorbs all developer variation into one `sigma_dev`. "
        "m2 partitions it into designation + team + individual. "
        "If m_simple sigma >> m2 sigma, team/designation effects are real."
    )

    sigma_s_simple = idata_simple.posterior['sigma_dev'].values.flatten()
    sigma_s_m2     = idata_m2.posterior['sigma_dev'].values.flatten()

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(sigma_s_simple, bins=50, alpha=0.6, color='steelblue',
            density=True, label=f'm_simple sigma_dev (mean={sigma_s_simple.mean():.2f})')
    ax.hist(sigma_s_m2,     bins=50, alpha=0.6, color='darkorange',
            density=True, label=f'm2 sigma_dev (mean={sigma_s_m2.mean():.2f})')
    ax.set_xlabel("sigma_dev")
    ax.set_title("sigma_dev: m_simple vs m2")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Tab 2: Developer Lookup ───────────────────────────────────────────────────
with tab2:
    st.subheader("Developer Lookup")

    dev_id = int(st.number_input(
        "Developer ID", min_value=0, max_value=N_DEVS - 1, value=155, step=1
    ))

    dev_info  = dev_df.iloc[dev_id]
    desig     = dev_info['designation']
    team      = dev_info['team']
    tier      = dev_info['tier']
    true_lam  = dev_info['true_lam']
    desig_idx = DESIGNATIONS.index(desig)

    # ── Developer info ────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Designation", desig)
    col2.metric("Team", team)
    col3.metric("Tier", tier.replace('_', ' ').title())
    col4.metric("True λ (ground truth)", f"{true_lam:.2f} PRs/mo")

    # ── Rate comparison ───────────────────────────────────────────────────────
    st.subheader("Rate Estimates — Three Models Side by Side")
    st.markdown(
        "Anomaly check: if **m2** is very different from **Empirical**, "
        "the team/designation context is pulling hard — worth investigating."
    )

    fig, ax = plt.subplots(figsize=(9, 3))

    models = ['Empirical',         'm_simple',         'm2 (hierarchical)']
    meds   = [emp[dev_id],         lam_s_med[dev_id],  lam_m2_med[dev_id]]
    los    = [emp[dev_id],         lam_s_lo[dev_id],   lam_m2_lo[dev_id]]
    his    = [emp[dev_id],         lam_s_hi[dev_id],   lam_m2_hi[dev_id]]
    colors = ['grey',              'steelblue',         'darkorange']

    for i, (name, med, lo, hi, color) in enumerate(zip(models, meds, los, his, colors)):
        ax.barh(i, med, color=color, alpha=0.6, height=0.5)
        if name != 'Empirical':
            ax.plot([lo, hi], [i, i], color=color, lw=4, alpha=0.5)
        ax.text(max(his) * 1.02, i, f'{med:.1f}', va='center', fontsize=11)

    ax.axvline(true_lam, color='red', lw=2, linestyle='--',
               label=f'True λ = {true_lam:.1f}')
    ax.set_yticks(range(3))
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel("Estimated λ (PRs / month)")
    ax.set_title(f"Dev {dev_id} ({desig}, {team}) — rate comparison")
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Percentile analysis ───────────────────────────────────────────────────
    st.subheader(f"Percentile within {desig} (m2)")
    st.markdown(
        "**P(beats random peer)** = probability this developer outperforms a randomly chosen "
        f"{desig} colleague (accounts for individual noise). "
        "This is the primary ranking metric."
    )

    rng             = np.random.default_rng(42)
    S               = len(mu_org_s)
    lam_dev_post    = np.exp(alpha_m2[:, dev_id])
    z_peer          = rng.normal(0, 1, S)
    lam_peer        = np.exp(mu_org_s + delta_s[:, desig_idx] + z_peer * sigma_dev_s)

    # Primary metric: P(dev > random peer) — honest peer comparison
    prob_beats_peer = (lam_dev_post > lam_peer).mean() * 100

    # Secondary: dev posterior median vs designation mean
    lam_desig_pop   = np.exp(mu_org_s + delta_s[:, desig_idx])
    prob_above_mean = (lam_dev_post > lam_desig_pop).mean() * 100

    # Outlier status based on prob_beats_peer
    if prob_beats_peer < 10:
        outlier_status = "Bottom 10%"
    elif prob_beats_peer < 25:
        outlier_status = "Bottom 25%"
    elif prob_beats_peer > 90:
        outlier_status = "Top 10%"
    elif prob_beats_peer > 75:
        outlier_status = "Top 25%"
    else:
        outlier_status = "Within normal range"

    col1, col2, col3 = st.columns(3)
    col1.metric(
        f"P(beats random {desig} peer)",
        f"{prob_beats_peer:.1f}%",
        f"≈ {prob_beats_peer:.0f}th percentile"
    )
    col2.metric(
        f"P(above {desig} mean)",
        f"{prob_above_mean:.1f}%"
    )
    col3.metric("Outlier status", outlier_status)

    # ── Monthly trend ─────────────────────────────────────────────────────────
    st.subheader("Monthly PR trend")

    dev_monthly = (df[df['developer_id'] == dev_id]
                   [['month', 'pr_count']].sort_values('month'))

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(dev_monthly['month'], dev_monthly['pr_count'],
           color='steelblue', alpha=0.7, label='Observed count')
    ax.axhline(emp[dev_id],       color='grey',       lw=1.5,
               linestyle='--', label=f'Empirical mean = {emp[dev_id]:.1f}')
    ax.axhline(lam_s_med[dev_id], color='steelblue',  lw=1.5,
               linestyle='--', label=f'm_simple median = {lam_s_med[dev_id]:.1f}')
    ax.axhline(lam_m2_med[dev_id],color='darkorange',  lw=1.5,
               linestyle='--', label=f'm2 median = {lam_m2_med[dev_id]:.1f}')
    ax.axhline(true_lam,          color='red',         lw=1.5,
               linestyle=':',  label=f'True λ = {true_lam:.1f}')
    ax.set_xticks(dev_monthly['month'])
    ax.set_xlabel("Month")
    ax.set_ylabel("PR count")
    ax.set_title(f"Dev {dev_id} monthly PR counts")
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
