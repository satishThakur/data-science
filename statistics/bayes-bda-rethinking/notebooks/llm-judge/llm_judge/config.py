"""
config.py — single source of truth for scenarios, constants, colours.
No imports beyond stdlib / pathlib.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "llm_judge"

# ── MCMC settings ─────────────────────────────────────────────────────────────
MCMC_PARAMS = dict(draws=2000, tune=1000, target_accept=0.9, random_seed=42)
CI_LO, CI_HI = 5.5, 94.5   # 89% credible interval (McElreath convention)

# ── Colour palette ─────────────────────────────────────────────────────────────
COLORS = {
    "theta":   "#2980B9",   # blue
    "tpr":     "#E67E22",   # orange
    "tnr":     "#27AE60",   # green
    "p_judge": "#8E44AD",   # purple
    "truth":   "#E74C3C",   # crimson
    "flat":    "#2980B9",   # scenario A
    "info":    "#E67E22",   # scenario B
}

# ── Scenarios ──────────────────────────────────────────────────────────────────
SCENARIOS = {
    "baseline": {
        "description":  "Baseline — moderate data, flat priors",
        "true_theta":   0.70,
        "true_tpr":     0.82,
        "true_tnr":     0.79,
        "n_val_pos":    50,
        "n_val_neg":    50,
        "N_prod":       200,
        "theta_prior":  (1, 1),
        "tpr_prior":    (1, 1),
        "tnr_prior":    (1, 1),
    },
    "small_flat": {
        "description":  "Small data + flat priors",
        "true_theta":   0.70,
        "true_tpr":     0.82,
        "true_tnr":     0.79,
        "n_val_pos":    10,
        "n_val_neg":    10,
        "N_prod":       40,
        "theta_prior":  (1, 1),
        "tpr_prior":    (1, 1),
        "tnr_prior":    (1, 1),
    },
    "small_informative": {
        "description":  "Small data + informative priors",
        "true_theta":   0.70,
        "true_tpr":     0.82,
        "true_tnr":     0.79,
        "n_val_pos":    10,
        "n_val_neg":    10,
        "N_prod":       40,
        "theta_prior":  (3, 2),
        "tpr_prior":    (8, 2),
        "tnr_prior":    (8, 2),
    },
    "large_flat": {
        "description":  "Large data + flat priors",
        "true_theta":   0.70,
        "true_tpr":     0.82,
        "true_tnr":     0.79,
        "n_val_pos":    200,
        "n_val_neg":    200,
        "N_prod":       1000,
        "theta_prior":  (1, 1),
        "tpr_prior":    (1, 1),
        "tnr_prior":    (1, 1),
    },
    "large_informative": {
        "description":  "Large data + informative priors",
        "true_theta":   0.70,
        "true_tpr":     0.82,
        "true_tnr":     0.79,
        "n_val_pos":    200,
        "n_val_neg":    200,
        "N_prod":       1000,
        "theta_prior":  (3, 2),
        "tpr_prior":    (8, 2),
        "tnr_prior":    (8, 2),
    },
}

PRESET_NAMES = list(SCENARIOS.keys())

PARAM_LABELS = {
    "theta": "θ — true pass rate",
    "tpr":   "TPR — judge sensitivity",
    "tnr":   "TNR — judge specificity",
}
