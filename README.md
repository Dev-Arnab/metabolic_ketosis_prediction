# Metabolic Ketosis Prediction

**DRIVE Lab** — predicting postpartum hyperketonemia (ketosis) in dairy cows from prepartum information.

## What this project is about

Hyperketonemia early in lactation is common, costly, and often subclinical. Cows are usually identified only after calving, when the metabolic disturbance is already underway.

This project asks a practical question:

> Can we identify cows at higher risk of postpartum ketosis using only data available **before** calving?

We use prepartum metabolic measurements and management/parity records to build and compare machine-learning screening models under small-sample, class-imbalanced conditions.

## Approach (overview)

- **Outcome:** cow-level postpartum hyperketonemia (HYK+ / HYK−)
- **Predictors:** prepartum-only features (no postpartum leakage)
- **Feature domains compared:** blood metabolite / blood-gas panel vs management & parity variables vs combined sets
- **Models compared:** logistic regression, random forest, XGBoost, SVM
- **Imbalance handling:** baseline, cost-sensitive learning, and resampling (e.g. SMOTE)
- **Evaluation focus:** leave-one-out cross-validation and precision–recall performance (PR-AUC), appropriate for low prevalence

## Repository status

This public repository is a **compact project overview** for sharing the research direction.

- Raw datasets are **not** included
- Detailed notebooks, manuscripts, and draft materials remain private

## Data availability

Study data are not publicly released in this repository. Collaborators with approved access can obtain the analysis files separately.

## Citation / contact

Work in progress — DRIVE Lab. Please contact the project team before reusing or redistributing any related materials.
