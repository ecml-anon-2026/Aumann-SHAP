# \# Aumann-SHAP Experiments

# 

# This repository contains the code required to reproduce the experiments presented in the \*\*Aumann-SHAP\*\* study.

# 

# It includes evaluation pipelines for:

# \- \*\*German Credit\*\* (tabular counterfactual attribution + interaction analysis)

# \- \*\*MNIST (1 → 7)\*\* (pixel attributions, heatmaps, patch test, global explanations)

# 

# ---

# 

# \## Installation

# 

# 1\) Clone the repository:

# https://github.com/ecml-anon-2026/Aumann-SHAP

# 

# 2\) Install dependencies (from the repository root folder):

# pip install -r requirements.txt

# 

# ---

# 

# \## Quickstart (Primary runs)

# 

# German Credit (primary):

# python experiments/run\_german\_credit.py --task local

# 

# MNIST (primary):

# python experiments/run\_mnist.py --task patchtest

# 

# ---

# 

# \## Available tasks

# 

# German Credit:

# \- local

# \- within\_pot

# \- global

# \- msweep

# \- convergence

# 

# MNIST:

# \- train

# \- equal\_split

# \- micro\_game

# \- heatmaps

# \- patchtest

# \- global

# \- globalheat

# 

# ---

# 

# \## Reproducibility / Execution order (IMPORTANT)

# 

# Some scripts depend on files created by earlier scripts (caches/artifacts/checkpoints).

# Use the documented order here:

# 

# \- docs/REPRODUCIBILITY.md

# \- experiments/german\_credit/README.md

# \- experiments/mnist/README.md

# 

# ---

# 

# \## Notes

# 

# \- No datasets are committed to this repository.

# \- German Credit uses a pretrained cached model included at:

#   experiments/german\_credit/cache/models\_split\_rs1.joblib

# \- MNIST training can be skipped if the checkpoint already exists.

# 

# ---

# 





Package (optional):

pip install -e .

Python usage:

from aumann\_shap import explain\_tabular\_gridstate





# \## Project structure

# 

# docs/                 reproducibility + anonymity notes

# experiments/          entrypoints + dataset-specific scripts

# requirements.txt      dependencies

# LICENSE               MIT license

