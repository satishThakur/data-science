# CLAUDE.md — PR Inference Subproject

Hierarchical Bayesian model for developer PR rates. See `README.md` for full design rationale.

## Pipeline Order

Run notebooks in sequence — each depends on the previous:

```
01_simulate.ipynb       → data/pr_simulated.csv
02_eda.ipynb            → (validation only, no outputs)
03_models_prior_check.ipynb  → (model definitions + prior checks, no saved files)
04_fit_models.ipynb     → data/idata_m_simple.nc, idata_m1.nc, idata_m2.nc
05_inference.ipynb      → (reads idata_m2.nc + idata_simple.nc)
app.py                  → reads all .nc files + pr_simulated.csv
```

Run app: `streamlit run notebooks/pr-inference/app.py` from project root.

## Known Issues

**m1 convergence (do not use for inference)**
- R-hat = 1.102, ESS = 40
- Root cause: `log(λ) = μ_org + α_desig + α_team + α_dev` — four additive terms create a flat ridge even with ZeroSumNormal on α_desig and α_team
- m1 is kept for comparison but its posteriors are unreliable

**m2 divergences (8 out of 4000)**
- Small but non-zero (0.2%)
- Root cause: residual funnel geometry when σ parameters approach 0
- Results are reliable but run with `target_accept=0.99` and `tune=5000` if you want cleaner diagnostics

## Model Conventions

**Naming**: m_simple / m1 / m2 (increasing complexity)

**m_simple**: one parameter per developer, no designation/team structure
- No `mu_org` — hardcoded `np.log(8)` to prevent ridge with `z_dev`
- ZeroSumNormal on z_dev to anchor the intercept

**m2 (main model)**: full crossed hierarchy with partial pooling
- HalfNormal(0.5) for all σ priors (not Exponential(1)) — lighter tails, fewer divergences
- Non-centred for all group effects (z × σ pattern)
- `dev_desig` / `dev_team` from `drop_duplicates('developer_id')` — never `desig_ids[::6]`

## Data Shapes

| Variable | Shape | Source |
|----------|-------|--------|
| `dev_ids` | (1896,) | One entry per observation |
| `desig_ids` | (1896,) | One entry per observation |
| `team_ids` | (1896,) | One entry per observation |
| `pr_obs` | (1896,) | Observed counts |
| `dev_desig` | (316,) | One entry per developer |
| `dev_team` | (316,) | One entry per developer |
| `alpha` (posterior) | (4000, 316) | Chains × draws × developers |
| `delta` (posterior) | (4000, 5) | Chains × draws × designations |
| `gamma` (posterior) | (4000, 15) | Chains × draws × teams |

## Adapting to Real Data

To plug in real org data:
1. Replace `data/pr_simulated.csv` with real CSV in the same column format:
   `developer_id, month, designation, desig_id, team, team_id, pr_count, exposure`
2. Set correct `exposure` values (fraction of month worked, e.g., 0.5 for a new joiner)
3. Re-run `04_fit_models.ipynb` — everything else is unchanged
4. Remove `true_lam` references from `app.py` (only exists for simulated data)

## Posterior Access Pattern

```python
import arviz as az
idata = az.from_netcdf('data/idata_m2.nc')
post = idata.posterior
alpha_s = post['alpha'].values.reshape(-1, post.sizes['developer'])  # (4000, 316)
sigma_dev_s = post['sigma_dev'].values.flatten()                      # (4000,)
```

## Common PyMC Gotchas

- `pm.compute_log_likelihood()` must be inside the `with model:` block
- `os.remove(path)` before `idata.to_netcdf(path)` to avoid stale file lock
- Use `post.sizes['chain']` not `post.dims['chain']` (ArviZ FutureWarning)
- `drop_duplicates('developer_id').sort_values('developer_id')` for per-dev lookups

---

*Last updated: 2026-03-24*
