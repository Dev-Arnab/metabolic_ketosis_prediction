# Metabolic Ketosis Prediction

**DRIVE Lab** — predicting postpartum hyperketonemia (ketosis) in dairy cows from prepartum information.

## What this project is about

Hyperketonemia early in lactation is common, costly, and often subclinical. Cows are usually identified only after calving, when the metabolic disturbance is already underway.

This project asks:

> Can we identify cows at higher risk of postpartum ketosis using only data available **before** calving?

We compare prepartum metabolic measurements and management/parity records across machine-learning models, with careful handling of small samples and class imbalance.

## What's in this repository

| Path | Description |
|------|-------------|
| `final_analysis/` | Final consolidated analysis notebook and key figures |
| `evolutionary_feature_selection/` | Genetic-algorithm / evolutionary feature-selection code (NSGA-II and related GAs), notebook, and figures |

Manuscript drafts and intermediate exploration notebooks are **not** included.

## Approach (overview)

- **Outcome:** cow-level postpartum hyperketonemia (HYK+ / HYK−)
- **Predictors:** prepartum-only features (no postpartum leakage)
- **Feature domains:** metabolite / blood-gas panel vs management & parity vs combined
- **Models:** logistic regression, random forest, XGBoost, SVM
- **Imbalance handling:** baseline, cost-sensitive learning, SMOTE
- **Feature selection:** greedy forward selection, plus evolutionary / genetic search (multi-objective NSGA-II and related variants)
- **Evaluation:** leave-one-out cross-validation; primary metric **PR-AUC**

## How to run

1. Place the study dataset (`Dataset_metabolic_ac.xlsx`) in the **repository root** (not committed here).
2. Create a Python environment with the usual scientific stack (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `shap`, `matplotlib`, etc.).
3. Run:

```text
final_analysis/Final_Paper_Analysis.ipynb
evolutionary_feature_selection/evolutionary_feature_selection.ipynb
```

The evolutionary notebook can also be driven through `evolutionary_feature_selection/analysis.py`.

## Data

Raw datasets are private and excluded via `.gitignore`. Collaborators with approved access should obtain the files separately and place them in the repo root before running the notebooks.

## Contact

Work in progress — DRIVE Lab. Please contact the project team before reusing related materials.
