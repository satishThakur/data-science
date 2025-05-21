# fast.ai Deep Learning

This directory contains code and notes for working with the fast.ai library, following the [fastai course](https://course.fast.ai/) and the "Deep Learning for Coders with fastai & PyTorch" book.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for Python environment management and package installation.

### Prerequisites

- Make sure you have `uv` installed. If not, install it following the [official instructions](https://github.com/astral-sh/uv#installation).

### Getting Started

1. Create a virtual environment:
   ```bash
   uv venv
   ```

2. Install dependencies:
   ```bash
   uv add fastai jupyter ipywidgets
   ```

### Running Jupyter Notebooks

To start a Jupyter Notebook server:

```bash
uv run jupyter notebook
```

## Project Structure

- `chapter1_intro.ipynb` - Code and exercises from Chapter 1 (image classification)
- Additional notebooks for subsequent chapters will be added later

## Learning Resources

- [fast.ai Course](https://course.fast.ai/)
- [fast.ai Documentation](https://docs.fast.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Practical Deep Learning for Coders](https://github.com/fastai/fastbook)