"""
precompute_presets.py — run once to generate .nc cache files for all 5 preset scenarios.

Run from the llm-judge directory:
    uv run python scripts/precompute_presets.py
"""
import sys
import time
from pathlib import Path

# Allow imports from llm_judge package
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_judge.config import SCENARIOS, DATA_DIR
from llm_judge.inference import run_scenario
import arviz as az
import os

DATA_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {DATA_DIR}\n")

for name in SCENARIOS:
    out_path = DATA_DIR / f"{name}.nc"
    if out_path.exists():
        print(f"  {name}: already exists — skipping (delete to re-run)")
        continue

    print(f"  {name}: running MCMC...", end=" ", flush=True)
    t0 = time.time()
    cfg, data, idata = run_scenario(name, seed=42)
    elapsed = time.time() - t0

    # Remove stale file if exists (avoids ArviZ lock issue)
    if out_path.exists():
        os.remove(out_path)

    idata.to_netcdf(str(out_path))
    size_kb = out_path.stat().st_size // 1024
    print(f"done in {elapsed:.1f}s  →  {out_path.name}  ({size_kb} KB)")

print("\nAll presets ready.")
