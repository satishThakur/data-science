# Exploratory Data Analysis (EDA) Learning Resources

This directory contains learning materials and examples for Exploratory Data Analysis techniques.

## Project Setup

This project uses [uv](https://github.com/astral-sh/uv) for Python environment management and package installation.

### Prerequisites

- Make sure you have `uv` installed. If not, install it following the [official instructions](https://github.com/astral-sh/uv#installation).

### Getting Started

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd statistics/exploratory-data-analysis
   ```

2. **Create a virtual environment**:
   ```bash
   uv venv
   ```
   This creates a virtual environment in the `.venv` directory.

3. **Install dependencies**:
   ```bash
   uv add numpy pandas matplotlib seaborn jupyter scikit-learn statsmodels plotly
   ```

### Running Jupyter Notebooks

To start a Jupyter Notebook server:

```bash
uv run jupyter notebook
```

This will start the server and open a browser window with the Jupyter interface. From there, you can navigate to any of the notebook files (`.ipynb`) and open them.

### Project Structure

- `notes.md` - Theoretical explanations and concepts for EDA
- `eda_basics.ipynb` - Basic EDA techniques demonstrated with sample datasets
- Additional notebooks for specific EDA topics will be added over time

## Learning Path

1. Start with the `notes.md` file to understand the theoretical concepts
2. Work through the `eda_basics.ipynb` notebook to see practical examples
3. Experiment with the techniques on your own datasets

## Additional Resources

- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)