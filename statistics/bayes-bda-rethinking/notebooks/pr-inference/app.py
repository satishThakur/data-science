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
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PR Inference",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom style ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .metric-label { font-size: 0.8rem !important; }
    h1 { text-align: center; }
    h3 { color: #555; font-weight: 500; }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 0.5rem 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent.parent.parent / 'data'
DESIGNATIONS = ['aSDE', 'SDE-1', 'SDE-2', 'SDE-3', 'SDE-4']
DESIG_COLORS = px.colors.qualitative.Plotly[:5]
GRAND_MEAN   = np.log(8)

# ── Load data (cached) ────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    df           = pd.read_csv(DATA_DIR / 'pr_simulated.csv')
    idata_m2     = az.from_netcdf(DATA_DIR / 'idata_m2.nc')
    idata_simple = az.from_netcdf(DATA_DIR / 'idata_simple.nc')
    return df, idata_m2, idata_simple

df, idata_m2, idata_simple = load_all()

@st.cache_resource
def precompute(_df, _idata_m2, _idata_simple):
    dev_df  = _df.drop_duplicates('developer_id').sort_values('developer_id').reset_index(drop=True)
    N_DEVS  = len(dev_df)
    N_TEAMS = _df['team_id'].nunique()

    emp = (_df.groupby('developer_id')['pr_count'].sum()
           / _df.groupby('developer_id')['exposure'].sum()).values

    alpha_s      = _idata_simple.posterior['alpha'].values.reshape(-1, N_DEVS)
    lam_s_med    = np.median(np.exp(alpha_s), axis=0)
    lam_s_lo     = np.percentile(np.exp(alpha_s), 5,  axis=0)
    lam_s_hi     = np.percentile(np.exp(alpha_s), 95, axis=0)

    alpha_m2     = _idata_m2.posterior['alpha'].values.reshape(-1, N_DEVS)
    lam_m2_med   = np.median(np.exp(alpha_m2), axis=0)
    lam_m2_lo    = np.percentile(np.exp(alpha_m2), 5,  axis=0)
    lam_m2_hi    = np.percentile(np.exp(alpha_m2), 95, axis=0)

    mu_org_s     = _idata_m2.posterior['mu_org'].values.flatten()
    delta_s      = _idata_m2.posterior['delta'].values.reshape(-1, len(DESIGNATIONS))
    gamma_s      = _idata_m2.posterior['gamma'].values.reshape(-1, N_TEAMS)
    sigma_dev_s  = _idata_m2.posterior['sigma_dev'].values.flatten()
    lam_desig    = np.exp(mu_org_s[:, None] + delta_s)
    lam_team     = np.exp(mu_org_s[:, None] + gamma_s)

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

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Org Overview", "Developer Lookup"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Models**")
st.sidebar.markdown("- 🔵 **Empirical** — raw MLE")
st.sidebar.markdown("- 🟠 **m_simple** — per-developer Bayes")
st.sidebar.markdown("- 🟢 **m2** — full hierarchy")
st.sidebar.markdown("---")
st.sidebar.caption(f"{N_DEVS} developers · {N_TEAMS} teams · 6 months")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1>📊 PR Inference Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#888; margin-bottom:2rem;'>"
    "Bayesian analysis of developer PR rates across teams and designations"
    "</p>",
    unsafe_allow_html=True
)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ORG OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "Org Overview":

    # ── Top KPIs ──────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Developers", N_DEVS)
    col2.metric("Teams", N_TEAMS)
    col3.metric("Org mean (m2)", f"{np.exp(mu_org_s.mean()):.1f} PRs/mo")
    col4.metric("Avg individual noise", f"±{np.exp(sigma_dev_s.mean()):.2f}x")

    st.markdown("---")

    # ── Designation rates ─────────────────────────────────────────────────────
    st.subheader("Designation Rates")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        fig = go.Figure()

        for i, desig in enumerate(DESIGNATIONS):
            med    = np.median(lam_desig[:, i])
            lo, hi = np.percentile(lam_desig[:, i], [5, 95])

            # m2 posterior CI
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[desig, desig],
                mode='lines',
                line=dict(color=DESIG_COLORS[i], width=6),
                opacity=0.4,
                showlegend=False,
                hovertemplate=f"{desig} 90% CI: [{lo:.1f}, {hi:.1f}]<extra></extra>"
            ))
            # m2 median dot
            fig.add_trace(go.Scatter(
                x=[med], y=[desig],
                mode='markers',
                marker=dict(color=DESIG_COLORS[i], size=12, symbol='circle'),
                name=f"{desig} (m2)",
                hovertemplate=f"{desig} m2 median: {med:.1f} PRs/mo<extra></extra>"
            ))
            # Empirical mean diamond
            emp_mean = emp[dev_df['designation'] == desig].mean()
            fig.add_trace(go.Scatter(
                x=[emp_mean], y=[desig],
                mode='markers',
                marker=dict(color=DESIG_COLORS[i], size=10, symbol='diamond',
                            line=dict(color='white', width=1)),
                showlegend=False,
                hovertemplate=f"{desig} empirical mean: {emp_mean:.1f} PRs/mo<extra></extra>"
            ))

        fig.add_vline(x=np.exp(GRAND_MEAN), line_dash="dash",
                      line_color="grey", opacity=0.6,
                      annotation_text="Grand mean", annotation_position="top right")

        fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="PRs / month",
            yaxis_title="",
            showlegend=False,
            plot_bgcolor='white',
            xaxis=dict(gridcolor='#eee'),
            yaxis=dict(gridcolor='#eee'),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("**Designation summary**")
        rows = []
        for i, desig in enumerate(DESIGNATIONS):
            med    = np.median(lam_desig[:, i])
            lo, hi = np.percentile(lam_desig[:, i], [5, 95])
            n_devs = (dev_df['designation'] == desig).sum()
            rows.append({
                "Designation": desig,
                "m2 median": f"{med:.1f}",
                "90% CI": f"[{lo:.1f}, {hi:.1f}]",
                "N devs": n_devs,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("---")

    # ── Team rates ────────────────────────────────────────────────────────────
    st.subheader("Team Rates")

    TIER_COLOR = {'balanced': '#636EFA', 'junior_heavy': '#EF553B', 'mostly_junior': '#00CC96'}
    team_names = [f'Team-{i+1}' for i in range(N_TEAMS)]

    fig = go.Figure()

    for i, tname in enumerate(team_names):
        tier_val = dev_df[dev_df['team'] == tname]['tier'].iloc[0]
        color    = TIER_COLOR[tier_val]
        med      = np.median(lam_team[:, i])
        lo, hi   = np.percentile(lam_team[:, i], [5, 95])
        emp_mean = emp[dev_df['team'] == tname].mean()

        fig.add_trace(go.Scatter(
            x=[tname, tname], y=[lo, hi],
            mode='lines',
            line=dict(color=color, width=3),
            opacity=0.4,
            showlegend=False,
            hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=[tname], y=[med],
            mode='markers',
            marker=dict(color=color, size=10),
            name=tier_val.replace('_', ' ').title() if i in [0, 5, 10] else '',
            showlegend=i in [0, 5, 10],
            hovertemplate=f"{tname} m2 median: {med:.1f}<br>Empirical: {emp_mean:.1f}<extra></extra>"
        ))

    fig.add_hline(y=np.exp(GRAND_MEAN), line_dash="dash",
                  line_color="grey", opacity=0.6)
    fig.add_vline(x=4.5, line_dash="dot", line_color="black", opacity=0.3)
    fig.add_vline(x=9.5, line_dash="dot", line_color="black", opacity=0.3)

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="PRs / month",
        xaxis_title="",
        plot_bgcolor='white',
        xaxis=dict(gridcolor='#eee'),
        yaxis=dict(gridcolor='#eee'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Vertical dotted lines separate tiers. Blue = Balanced | Red = Junior-heavy | Green = Mostly-junior")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DEVELOPER LOOKUP
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Developer Lookup":

    col_input, _ = st.columns([1, 3])
    with col_input:
        dev_id = int(st.number_input(
            "Developer ID", min_value=0, max_value=N_DEVS - 1, value=155, step=1
        ))

    dev_info  = dev_df.iloc[dev_id]
    desig     = dev_info['designation']
    team      = dev_info['team']
    tier      = dev_info['tier'].replace('_', ' ').title()
    true_lam  = dev_info['true_lam']
    desig_idx = DESIGNATIONS.index(desig)

    # ── Developer info strip ──────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Developer", f"#{dev_id}")
    col2.metric("Designation", desig)
    col3.metric("Team", team)
    col4.metric("Tier", tier)
    col5.metric("True λ", f"{true_lam:.1f} PRs/mo")

    st.markdown("---")

    # ── Rate comparison ───────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Rate Estimates")

        fig = go.Figure()

        models = ['m2 (hierarchical)', 'm_simple', 'Empirical']
        meds   = [lam_m2_med[dev_id], lam_s_med[dev_id],  emp[dev_id]]
        los    = [lam_m2_lo[dev_id],  lam_s_lo[dev_id],   emp[dev_id]]
        his    = [lam_m2_hi[dev_id],  lam_s_hi[dev_id],   emp[dev_id]]
        colors = ['#00CC96',           '#636EFA',            '#aaa']

        for i, (name, med, lo, hi, color) in enumerate(zip(models, meds, los, his, colors)):
            # CI bar
            if name != 'Empirical':
                fig.add_trace(go.Scatter(
                    x=[lo, hi], y=[name, name],
                    mode='lines',
                    line=dict(color=color, width=8),
                    opacity=0.3,
                    showlegend=False,
                    hovertemplate=f"90% CI: [{lo:.1f}, {hi:.1f}]<extra></extra>"
                ))
            # Point estimate
            fig.add_trace(go.Scatter(
                x=[med], y=[name],
                mode='markers+text',
                marker=dict(color=color, size=14),
                text=[f"  {med:.1f}"],
                textposition='middle right',
                textfont=dict(size=13),
                name=name,
                hovertemplate=f"{name}: {med:.1f} PRs/mo<extra></extra>"
            ))

        fig.add_vline(x=true_lam, line_dash="dash", line_color="red",
                      annotation_text=f"True λ={true_lam:.1f}", annotation_position="top right")

        fig.update_layout(
            height=220,
            margin=dict(l=20, r=80, t=20, b=20),
            xaxis_title="PRs / month",
            yaxis_title="",
            showlegend=False,
            plot_bgcolor='white',
            xaxis=dict(gridcolor='#eee'),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Standing")

        rng             = np.random.default_rng(42)
        S               = len(mu_org_s)
        lam_dev_post    = np.exp(alpha_m2[:, dev_id])
        z_peer          = rng.normal(0, 1, S)
        lam_peer        = np.exp(mu_org_s + delta_s[:, desig_idx] + z_peer * sigma_dev_s)
        lam_desig_pop   = np.exp(mu_org_s + delta_s[:, desig_idx])

        prob_beats_peer = (lam_dev_post > lam_peer).mean() * 100
        prob_above_mean = (lam_dev_post > lam_desig_pop).mean() * 100

        if prob_beats_peer < 10:
            outlier_status = "⚠️ Bottom 10%"
        elif prob_beats_peer < 25:
            outlier_status = "📉 Bottom 25%"
        elif prob_beats_peer > 90:
            outlier_status = "🏆 Top 10%"
        elif prob_beats_peer > 75:
            outlier_status = "📈 Top 25%"
        else:
            outlier_status = "✅ Normal range"

        st.metric(f"P(beats random {desig})", f"{prob_beats_peer:.0f}%",
                  f"≈ {prob_beats_peer:.0f}th percentile")
        st.metric(f"P(above {desig} mean)", f"{prob_above_mean:.0f}%")
        st.metric("Outlier status", outlier_status)

    st.markdown("---")

    # ── Monthly trend ─────────────────────────────────────────────────────────
    st.subheader("Monthly Trend")

    dev_monthly = (df[df['developer_id'] == dev_id]
                   [['month', 'pr_count']].sort_values('month'))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=dev_monthly['month'],
        y=dev_monthly['pr_count'],
        name='Observed',
        marker_color='#636EFA',
        opacity=0.7,
    ))

    for val, name, color, dash in [
        (emp[dev_id],        'Empirical',  '#aaa',     'dash'),
        (lam_s_med[dev_id],  'm_simple',   '#636EFA',  'dot'),
        (lam_m2_med[dev_id], 'm2',         '#00CC96',  'dashdot'),
        (true_lam,           'True λ',     'red',      'dash'),
    ]:
        fig.add_hline(y=val, line_dash=dash, line_color=color,
                      annotation_text=f"{name}={val:.1f}",
                      annotation_position="right")

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=120, t=20, b=20),
        xaxis_title="Month",
        yaxis_title="PR count",
        xaxis=dict(tickmode='linear', dtick=1),
        plot_bgcolor='white',
        xaxis_gridcolor='#eee',
        yaxis_gridcolor='#eee',
        showlegend=False,
    )

    col_chart, _ = st.columns([3, 1])
    with col_chart:
        st.plotly_chart(fig, use_container_width=True)
