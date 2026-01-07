# project_mlops

Project repo for DTU course MLOPS
This repository contains a structured machine learning project following an
MLOps-oriented layout inspired by the Cookiecutter Data Science template.
The project trains, evaluates, and visualizes a CNN model on the **corrupted MNIST**
dataset using PyTorch.

The goal is to enforce **clean code organization**, **reproducibility**, and
**clear execution patterns** via CLI commands. Thanks MLOPS



## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).


# Running the code 

## Terminal Guide

This project follows a **src-based Python project structure** and uses **uv** for environment and dependency management.  
All commands below should be run **from the project root directory** (the folder containing `pyproject.toml`).

---

### Running Scripts Directly

All scripts use Typer, meaning they act like command-line programs.

### Preprocess the data

```bash
uv run <rawdata dir> <processed data dir>
```bash


Run to processes raw data. Normalizes image between 0 and 1, and saves in <processed data dir> augment directory

### Train model

uv run <src/project_mlops/train.py> --<learing rate> --<Batch size> --<epochs>
Run this to train model, with augments learning rate, batch size and epochs. Trained models are saved in models/ and plot and statistics are saved in reports/figures/

To inspect the model architechture
uv run src/project_mlops/model.py

### Model Evaluation

Evaluate a trained model checkpoint on the test set.

uv run src/project_mlops/evaluate.py models/model.pth

The output in terminal should be 

**Test accuracy: 0.87**

### Model visualization 

Visualize learned representations using t-SNE.

uv run src/project_mlops/visualize.py models/model.pth

This will:

Load the trained model

Extract features before the final layer

Reduce dimensionality with PCA + t-SNE

and save a plot to:

reports/figures/embeddings.png

### Tasks
We have also added tasks, included in tasks.py, which defines reusable CLI tasks using Invoke.

Install invoke
uvx invoke

List available tasks
uvx invoke --list

Example tasks
uvx invoke preprocess-data
uvx invoke train


tasks.py acts like a Makefile for Python and is useful for standardizing
repetitive commands.


### Adding dependencies

Add a dependency
uv add scikit-learn

Export requirements.txt (optional)
uv pip freeze > requirements.txt


Note: requirements.txt is optional when using uv, since pyproject.toml

uv.lock are the source of truth.
