# Bayesian Statistics

This project explores Bayesian statistics concepts, focusing on:
- Binomial distributions
- Beta distributions
- Bayesian inference
- Conjugate priors

## Setup

Using uv for environment and package management:

```bash
# Create virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate   # On Windows

# Install dependencies (including JupyterLab and ipykernel)
uv pip install -e .

# Register Jupyter kernel
python -m ipykernel install --user --name bayesian-stats --display-name "Python (Bayesian Stats)"

# Start Jupyter Lab
uv run jupyter lab
```

## Project Structure

- `notebooks/`: Jupyter notebooks with examples and explorations
- `src/`: Python modules for reusable functions
- `data/`: Data files used in examples