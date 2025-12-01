# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Bayesian data analysis project focused on statistical rethinking using PyMC, ArviZ, and other Python data science libraries. The project appears to be a learning/research repository for Bayesian statistics and probabilistic programming.

## Development Environment

The project uses `uv` as the package manager with Python 3.11+. Dependencies are managed through `pyproject.toml`.

### Key Dependencies
- **PyMC 5.26.1+**: Probabilistic programming framework for Bayesian modeling
- **ArviZ 0.22.0+**: Bayesian data analysis and visualization
- **JupyterLab 4.5.0+**: Interactive development environment
- **Standard data science stack**: NumPy, Pandas, Matplotlib, SciPy

## Common Development Commands

```bash
# Install dependencies
uv sync

# Run the main script
python main.py

# Start Jupyter Lab for notebook development
jupyter lab

# Install additional packages
uv add <package-name>
```

## Project Structure

```
├── main.py              # Simple entry point script
├── notebooks/           # Jupyter notebooks for analysis and experiments
│   ├── test.ipynb      # Environment verification notebook
│   └── bern.ipynb      # Bernoulli distribution examples
├── src/                 # Empty source directory (ready for future modules)
├── data/               # Empty data directory (for datasets)
├── pyproject.toml      # Project configuration and dependencies
└── uv.lock            # Dependency lock file
```

## Development Workflow

- **Notebooks First**: Primary development happens in Jupyter notebooks in the `notebooks/` directory
- **Minimal Source Code**: Currently only has a basic `main.py` entry point
- **Ready for Expansion**: Empty `src/` and `data/` directories suggest the project is set up for future growth

## Bayesian Analysis Context

This project focuses on Bayesian data analysis using modern Python tools. When working with this codebase:
- PyMC is the primary tool for probabilistic modeling
- ArviZ handles posterior analysis and visualization
- Notebooks serve as the main development interface for iterative analysis
- The project follows typical data science notebook-driven development patterns