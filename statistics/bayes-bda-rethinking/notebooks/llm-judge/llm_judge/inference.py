"""
inference.py — all PyMC logic. Only file that imports pymc.
"""
import logging
import warnings
import time

import numpy as np
import arviz as az

logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("numba").setLevel(logging.ERROR)
logging.getLogger("pytensor").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from .config import SCENARIOS, DATA_DIR, MCMC_PARAMS, CI_LO, CI_HI


# ── Data simulation ────────────────────────────────────────────────────────────

def simulate_data(cfg: dict, seed: int = 42) -> dict:
    """Simulate observed counts from true parameters. Deterministic given seed."""
    rng    = np.random.default_rng(seed=seed)
    k_pp   = rng.binomial(cfg["n_val_pos"], cfg["true_tpr"])
    k_nn   = rng.binomial(cfg["n_val_neg"], cfg["true_tnr"])
    p_j    = cfg["true_theta"] * cfg["true_tpr"] + (1 - cfg["true_theta"]) * (1 - cfg["true_tnr"])
    K_prod = rng.binomial(cfg["N_prod"], p_j)
    return {"k_pp": k_pp, "k_nn": k_nn, "K_prod": K_prod, "p_judge_true": p_j}


# ── Model building ─────────────────────────────────────────────────────────────

def build_pymc_model(cfg: dict, data: dict):
    """Build and return the PyMC model (not yet sampled)."""
    import pymc as pm

    with pm.Model() as model:
        theta   = pm.Beta("theta", alpha=cfg["theta_prior"][0], beta=cfg["theta_prior"][1])
        tpr     = pm.Beta("tpr",   alpha=cfg["tpr_prior"][0],   beta=cfg["tpr_prior"][1])
        tnr     = pm.Beta("tnr",   alpha=cfg["tnr_prior"][0],   beta=cfg["tnr_prior"][1])
        p_judge = pm.Deterministic("p_judge", theta * tpr + (1 - theta) * (1 - tnr))
        pm.Binomial("k_pp",   n=cfg["n_val_pos"], p=tpr,     observed=data["k_pp"])
        pm.Binomial("k_nn",   n=cfg["n_val_neg"], p=tnr,     observed=data["k_nn"])
        pm.Binomial("K_prod", n=cfg["N_prod"],    p=p_judge, observed=data["K_prod"])

    return model


def run_mcmc(model, seed: int = 42) -> az.InferenceData:
    """Run NUTS on an already-built model. Returns idata (no PPC yet)."""
    import pymc as pm

    with model:
        idata = pm.sample(
            draws=MCMC_PARAMS["draws"],
            tune=MCMC_PARAMS["tune"],
            target_accept=MCMC_PARAMS["target_accept"],
            random_seed=seed,
            progressbar=False,
        )
    return idata


def add_ppc(model, idata: az.InferenceData, seed: int = 42) -> az.InferenceData:
    """Append posterior predictive samples to idata in place. Returns idata."""
    import pymc as pm

    with model:
        pm.sample_posterior_predictive(idata, random_seed=seed, extend_inferencedata=True,
                                       progressbar=False)
    return idata


# ── High-level helpers ─────────────────────────────────────────────────────────

def run_scenario(scenario_name: str, seed: int = 42,
                 progress_callback=None) -> tuple:
    """
    Simulate data + run full inference for a named scenario.
    progress_callback(phase: str) is called at each phase for UI updates.
    Returns (cfg, data, idata).
    """
    cfg  = SCENARIOS[scenario_name]
    data = simulate_data(cfg, seed=seed)

    if progress_callback:
        progress_callback("building")
    model = build_pymc_model(cfg, data)

    if progress_callback:
        progress_callback("sampling")
    idata = run_mcmc(model, seed=seed)

    if progress_callback:
        progress_callback("ppc")
    idata = add_ppc(model, idata, seed=seed)

    return cfg, data, idata


def run_custom_scenario(cfg: dict, seed: int = 42,
                        progress_callback=None) -> tuple:
    """
    Run inference for a custom (user-defined) config dict.
    Same structure as run_scenario but cfg is passed directly.
    Returns (cfg, data, idata).
    """
    data = simulate_data(cfg, seed=seed)

    if progress_callback:
        progress_callback("building")
    model = build_pymc_model(cfg, data)

    if progress_callback:
        progress_callback("sampling")
    idata = run_mcmc(model, seed=seed)

    if progress_callback:
        progress_callback("ppc")
    idata = add_ppc(model, idata, seed=seed)

    return cfg, data, idata


def load_preset(scenario_name: str) -> tuple:
    """
    Load pre-computed idata from .nc file. Fast — no PyMC needed.
    Returns (cfg, data, idata).
    """
    path  = DATA_DIR / f"{scenario_name}.nc"
    idata = az.from_netcdf(str(path))
    cfg   = SCENARIOS[scenario_name]
    data  = simulate_data(cfg, seed=42)   # deterministic — same seed as precompute
    return cfg, data, idata


# ── Diagnostics ────────────────────────────────────────────────────────────────

def compute_ci89(samples: np.ndarray) -> tuple:
    """Return (lo, hi) 89% credible interval."""
    return tuple(np.percentile(samples, [CI_LO, CI_HI]))


def convergence_summary(idata: az.InferenceData) -> dict:
    """
    Returns dict of {var: {"mean": float, "lo": float, "hi": float,
                            "rhat": float, "ess": float, "ok": bool}}
    for theta, tpr, tnr.
    """
    summary = az.summary(idata, var_names=["theta", "tpr", "tnr"], hdi_prob=0.89)
    result  = {}
    for var in ["theta", "tpr", "tnr"]:
        samples    = idata.posterior[var].values.ravel()
        lo, hi     = compute_ci89(samples)
        rhat       = float(summary.loc[var, "r_hat"])
        ess        = float(summary.loc[var, "ess_bulk"])
        result[var] = {
            "mean": float(samples.mean()),
            "lo":   lo,
            "hi":   hi,
            "rhat": rhat,
            "ess":  ess,
            "ok":   rhat < 1.01 and ess > 400,
        }
    return result
