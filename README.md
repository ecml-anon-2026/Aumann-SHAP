# Aumann-SHAP Experiments

This repository contains the code required to reproduce the experiments presented in the **Aumann-SHAP** study.

It includes evaluation pipelines for:
- **German Credit** (tabular counterfactual attribution + interaction analysis)
- **MNIST (1 → 7)** (pixel attributions, heatmaps, patch test, global explanations)
<img width="1393" height="331" alt="image" src="https://github.com/user-attachments/assets/e8e5f9d5-4b84-447d-a8e3-9af4aa3fa148" />

---

## Installation

1) Clone the repository:
https://github.com/ecml-anon-2026/Aumann-SHAP

2) Install dependencies (from the repository root folder):
pip install -r requirements.txt

---

## Quickstart (Primary runs)

German Credit (primary):
python experiments/run_german_credit.py --task local

MNIST (primary):
python experiments/run_mnist.py --task patchtest

---

## Available tasks

German Credit:
- local
- within_pot
- global
- msweep
- convergence

MNIST:
- train
- equal_split
- micro_game
- heatmaps
- patchtest
- global
- globalheat

---

## Reproducibility / Execution order (IMPORTANT)

Some scripts depend on files created by earlier scripts (caches/artifacts/checkpoints).  
Use the documented order here:
- docs/REPRODUCIBILITY.md
- experiments/german_credit/README.md
- experiments/mnist/README.md

---

## Notes

- No datasets are committed to this repository.
- German Credit uses a pretrained cached model included at:
  experiments/german_credit/cache/models_split_rs1.joblib
- MNIST training can be skipped if the checkpoint already exists.

---

## Package (optional)

You can install this repository as a local package:

pip install -e .

Minimal usage:

- Exact grid-state backend (tabular / small k): returns (Totals, Within-pot)
  from aumann_shap import explain
  totals, within_pot = explain(model, x0, x1, backend="grid_state", m=5)

- Monte Carlo backend (large k / pixels): returns (Totals, None)
  totals, within_pot = explain(model, x0, x1, backend="mc", m=10, n_perms=200, seed=0)

(Within-pot is only returned for the exact grid-state backend.)
## Usage for ML (Tabular data)

### 1) Minimal example (exact grid-state micro-game Shapley)

```python
import numpy as np
import pandas as pd
from aumann_shap import explain

# Any callable f(pd.Series) -> float works
def g(x: pd.Series) -> float:
    return float(np.tanh(0.8*x["x1"] + 0.4*x["x2"] + 0.6*x["x1"]*x["x2"]))

x0 = pd.Series({"x1": 0.0, "x2": 0.0})
x1 = pd.Series({"x1": 1.0, "x2": 1.0})

totals, within_pot = explain(g, x0, x1, backend="grid_state", m=5)

print("Totals (sum to Δ):")
print(totals)
print("\nWithin-pot:")
print(within_pot)
```
---

### 2) Monte Carlo micro-game Shapley (Totals only)

```python
import numpy as np
import pandas as pd
from aumann_shap import explain

def g(x: pd.Series) -> float:
    return float(np.tanh(0.8*x["x1"] + 0.4*x["x2"] + 0.6*x["x1"]*x["x2"]))

x0 = pd.Series({"x1": 0.0, "x2": 0.0})
x1 = pd.Series({"x1": 1.0, "x2": 1.0})

totals, within_pot = explain(
    g, x0, x1,
    backend="mc",
    m=10,
    n_perms=300,
    seed=0
)

print("Totals (MC estimate):")
print(totals)
print("within_pot:", within_pot)  # None for MC
```
If your model supports batch prediction (sklearn/xgboost/torch), explain uses it automatically; otherwise it falls back to per-row evaluation (may be slower).
---
##Usage for Vision (MNIST)
requires 
```python



```
## Project structure

docs/                 reproducibility + anonymity notes  
experiments/          entrypoints + dataset-specific scripts  
src/aumann_shap/      package source code  
requirements.txt      dependencies  
LICENSE               MIT license
