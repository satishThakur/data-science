"""
plots.py — all matplotlib visualisations. No Streamlit imports.
Every function returns a matplotlib Figure. Never calls plt.show().
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.stats import beta as beta_dist, binom, gaussian_kde

from .config import COLORS, PARAM_LABELS, CI_LO, CI_HI

# ── Shared helpers ─────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)


def _ci89(samples):
    return np.percentile(samples, [CI_LO, CI_HI])


def _kde_plot(ax, samples, color, label, lw=2):
    """KDE curve + light fill. Returns (lo, hi, mean)."""
    kde    = gaussian_kde(samples, bw_method="scott")
    x      = np.linspace(max(0, samples.min() - 0.05),
                         min(1, samples.max() + 0.05), 500)
    y      = kde(x)
    lo, hi = _ci89(samples)
    mean   = samples.mean()
    ax.plot(x, y, color=color, lw=lw,
            label=f"{label}  mean={mean:.3f}  CI=[{lo:.3f},{hi:.3f}]")
    ax.fill_between(x, y, alpha=0.15, color=color)
    ax.axvline(mean, color=color, lw=1.2, ls="--", alpha=0.7)
    return lo, hi, mean


def _int_bins(n):
    return np.arange(0, n + 2) - 0.5


# ── DAG ────────────────────────────────────────────────────────────────────────

def plot_dag() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    nodes = {
        "θ":         (2, 4.5, "#AED6F1",  "Latent"),
        "TPR":       (5, 5.5, "#AED6F1",  "Latent"),
        "TNR":       (5, 3.5, "#AED6F1",  "Latent"),
        "p_judge":   (5, 1.5, "#A9DFBF",  "Deterministic"),
        "K_PP":      (8, 5.5, "#FAD7A0",  "Observed"),
        "K_NN":      (8, 3.5, "#FAD7A0",  "Observed"),
        "K_prod":    (8, 1.5, "#FAD7A0",  "Observed"),
    }

    edges = [
        ("θ",   "p_judge"),
        ("TPR", "p_judge"),
        ("TNR", "p_judge"),
        ("TPR", "K_PP"),
        ("TNR", "K_NN"),
        ("p_judge", "K_prod"),
    ]

    for (src, dst) in edges:
        x0, y0 = nodes[src][:2]
        x1, y1 = nodes[dst][:2]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    for name, (x, y, color, kind) in nodes.items():
        circ = plt.Circle((x, y), 0.6, color=color, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, name, ha="center", va="center", fontsize=10, fontweight="bold", zorder=4)

    legend_patches = [
        mpatches.Patch(color="#AED6F1", label="Latent parameter"),
        mpatches.Patch(color="#A9DFBF", label="Deterministic"),
        mpatches.Patch(color="#FAD7A0", label="Observed"),
    ]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=9)
    ax.set_title("Generative Model — Directed Acyclic Graph", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Prior predictive ───────────────────────────────────────────────────────────

def plot_prior_predictive(cfg: dict, seed: int = 42) -> plt.Figure:
    rng = np.random.default_rng(seed)
    N   = 20_000

    a_th, b_th   = cfg["theta_prior"]
    a_tp, b_tp   = cfg["tpr_prior"]
    a_tn, b_tn   = cfg["tnr_prior"]

    theta_s = rng.beta(a_th, b_th, N)
    tpr_s   = rng.beta(a_tp, b_tp, N)
    tnr_s   = rng.beta(a_tn, b_tn, N)
    p_j_s   = theta_s * tpr_s + (1 - theta_s) * (1 - tnr_s)

    k_pp_s   = rng.binomial(cfg["n_val_pos"],  tpr_s)
    k_nn_s   = rng.binomial(cfg["n_val_neg"],  tnr_s)
    k_prod_s = rng.binomial(cfg["N_prod"],     p_j_s)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 1: Beta PDFs ──────────────────────────────────────────────────────
    for col, (name, a, b, color) in enumerate([
        ("θ",    a_th, b_th, COLORS["theta"]),
        ("TPR",  a_tp, b_tp, COLORS["tpr"]),
        ("TNR",  a_tn, b_tn, COLORS["tnr"]),
    ]):
        ax = fig.add_subplot(gs[0, col])
        x  = np.linspace(0.001, 0.999, 300)
        y  = beta_dist.pdf(x, a, b)
        lo, hi = np.percentile(rng.beta(a, b, 50_000), [CI_LO, CI_HI])
        ax.plot(x, y, color=color, lw=2)
        ax.fill_between(x, y, where=(x >= lo) & (x <= hi), alpha=0.3, color=color,
                        label=f"89% CI [{lo:.2f},{hi:.2f}]")
        ax.set_title(f"Prior: {name} ~ Beta({a},{b})", fontsize=10)
        ax.set_xlabel(name); ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        _style_ax(ax)

    # ── Row 2: Implied count histograms ──────────────────────────────────────
    for col, (name, samples, n_tot, color) in enumerate([
        ("K_PP (val pos)",  k_pp_s,   cfg["n_val_pos"], COLORS["tpr"]),
        ("K_NN (val neg)",  k_nn_s,   cfg["n_val_neg"], COLORS["tnr"]),
        ("K_prod",          k_prod_s, cfg["N_prod"],    COLORS["theta"]),
    ]):
        ax = fig.add_subplot(gs[1, col])
        ax.hist(samples, bins=_int_bins(n_tot), color=color, alpha=0.7, density=False)
        ax.set_title(f"Implied {name}", fontsize=10)
        ax.set_xlabel("Count"); ax.set_ylabel("Frequency")
        _style_ax(ax)

    # ── Row 3: Sampled parameter distributions ────────────────────────────────
    for col, (name, samples, color) in enumerate([
        ("TPR (sampled)",     tpr_s, COLORS["tpr"]),
        ("TNR (sampled)",     tnr_s, COLORS["tnr"]),
        ("p_judge (implied)", p_j_s, COLORS["p_judge"]),
    ]):
        ax = fig.add_subplot(gs[2, col])
        lo, hi = np.percentile(samples, [CI_LO, CI_HI])
        ax.hist(samples, bins=80, color=color, alpha=0.7, density=True)
        ax.axvspan(lo, hi, alpha=0.2, color=color, label=f"89% CI [{lo:.2f},{hi:.2f}]")
        ax.set_title(f"Sampled {name}", fontsize=10)
        ax.set_xlabel(name); ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        _style_ax(ax)

    fig.suptitle("Prior Predictive Check", fontsize=13, fontweight="bold")
    return fig


# ── Posteriors ─────────────────────────────────────────────────────────────────

def plot_posteriors(cfg: dict, idata, scenario_name: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Posterior Distributions — {scenario_name}", fontsize=13, fontweight="bold")

    params = [
        ("theta", cfg["true_theta"]),
        ("tpr",   cfg["true_tpr"]),
        ("tnr",   cfg["true_tnr"]),
    ]

    for ax, (var, truth) in zip(axes, params):
        samples    = idata.posterior[var].values.ravel()
        color      = COLORS[var]
        lo, hi, mn = _kde_plot(ax, samples, color, PARAM_LABELS[var])
        ax.axvline(truth, color=COLORS["truth"], lw=2, ls="--", label=f"Truth = {truth}")
        ax.axvspan(lo, hi, alpha=0.10, color=color)
        inside = "✓" if lo <= truth <= hi else "✗"
        ax.set_title(f"{var}  |  mean={mn:.3f}  {inside} truth={truth}", fontsize=10)
        ax.set_xlabel(PARAM_LABELS[var])
        ax.set_ylabel("Posterior density")
        ax.legend(fontsize=8)
        _style_ax(ax)

    fig.tight_layout()
    return fig


def posterior_ci_text(cfg: dict, idata) -> list[dict]:
    """Return plain-English CI sentences for all three parameters."""
    labels = {
        "theta": "true pass rate",
        "tpr":   "judge sensitivity (TPR)",
        "tnr":   "judge specificity (TNR)",
    }
    truths = {"theta": cfg["true_theta"], "tpr": cfg["true_tpr"], "tnr": cfg["true_tnr"]}
    rows   = []
    for var, label in labels.items():
        s      = idata.posterior[var].values.ravel()
        lo, hi = _ci89(s)
        truth  = truths[var]
        inside = lo <= truth <= hi
        rows.append({
            "var":    var,
            "mean":   s.mean(),
            "lo":     lo,
            "hi":     hi,
            "truth":  truth,
            "inside": inside,
            "text":   (
                f"There is an 89% probability the **{label}** "
                f"lies between **{lo:.3f}** and **{hi:.3f}**."
            ),
        })
    return rows


# ── Posterior predictive check ─────────────────────────────────────────────────

def plot_ppc(cfg: dict, data: dict, idata) -> plt.Figure:
    checks = [
        ("K_prod", "Production passes",         cfg["N_prod"],    COLORS["theta"]),
        ("k_pp",   "Validation true positives",  cfg["n_val_pos"], COLORS["tpr"]),
        ("k_nn",   "Validation true negatives",  cfg["n_val_neg"], COLORS["tnr"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Posterior Predictive Check", fontsize=13, fontweight="bold")

    for ax, (key, label, n_total, color) in zip(axes, checks):
        pp       = idata.posterior_predictive[key].values.ravel()
        observed = data[key]
        lo, hi   = np.percentile(pp, [CI_LO, CI_HI])
        inside   = lo <= observed <= hi
        pct      = (pp < observed).mean() * 100

        ax.hist(pp, bins=_int_bins(n_total), color=color, alpha=0.6,
                label="Posterior predictive")
        ax.axvline(observed, color=COLORS["truth"], lw=2.5, ls="--",
                   label=f"Observed = {observed}")
        ax.axvspan(lo, hi, alpha=0.15, color=color,
                   label=f"89% CI [{lo:.0f}, {hi:.0f}]")

        marker = "✓" if inside else "✗"
        ax.set_title(f"{label}\n{marker} observed={observed}  percentile={pct:.0f}%", fontsize=10)
        ax.set_xlabel(f"Simulated count  (N={n_total})")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        _style_ax(ax)

    fig.tight_layout()
    return fig


# ── Comparison plot ────────────────────────────────────────────────────────────

def plot_comparison(cfg_a, idata_a, name_a,
                    cfg_b, idata_b, name_b) -> plt.Figure:
    """
    Overlay posteriors of θ, TPR, TNR for two scenarios.
    Uses KDE curves — no histogram colour blending.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Comparison: {name_a}  vs  {name_b}",
                 fontsize=13, fontweight="bold")

    params = [
        ("theta", cfg_a["true_theta"]),
        ("tpr",   cfg_a["true_tpr"]),
        ("tnr",   cfg_a["true_tnr"]),
    ]

    for ax, (var, truth) in zip(axes, params):
        for cfg, idata, name, color in [
            (cfg_a, idata_a, name_a, COLORS["flat"]),
            (cfg_b, idata_b, name_b, COLORS["info"]),
        ]:
            s = idata.posterior[var].values.ravel()
            _kde_plot(ax, s, color, name)

        ax.axvline(truth, color=COLORS["truth"], lw=2, ls="--", label=f"Truth = {truth}")
        ax.set_xlabel(PARAM_LABELS[var])
        ax.set_ylabel("Posterior density")
        ax.set_title(f"Posterior of {var}", fontsize=10)
        ax.legend(fontsize=8)
        _style_ax(ax)

    fig.tight_layout()
    return fig


def comparison_verdict(idata_a, name_a, idata_b, name_b, true_theta) -> dict:
    """
    Compute summary stats and a plain-English verdict for a comparison.
    Returns dict with per-scenario stats and verdict string.
    """
    def stats(idata):
        s      = idata.posterior["theta"].values.ravel()
        lo, hi = _ci89(s)
        return {"mean": s.mean(), "lo": lo, "hi": hi, "width": hi - lo}

    sa = stats(idata_a)
    sb = stats(idata_b)

    # Which is tighter?
    if sa["width"] < sb["width"] * 0.95:
        verdict = f"**{name_a}** has a {(1-sa['width']/sb['width'])*100:.0f}% narrower CI."
    elif sb["width"] < sa["width"] * 0.95:
        verdict = f"**{name_b}** has a {(1-sb['width']/sa['width'])*100:.0f}% narrower CI."
    else:
        verdict = "Both posteriors are nearly identical — the data dominates the prior."

    # Are both covering truth?
    inside_a = sa["lo"] <= true_theta <= sa["hi"]
    inside_b = sb["lo"] <= true_theta <= sb["hi"]

    return {
        "a": sa, "b": sb,
        "name_a": name_a, "name_b": name_b,
        "verdict": verdict,
        "inside_a": inside_a, "inside_b": inside_b,
        "true_theta": true_theta,
    }
