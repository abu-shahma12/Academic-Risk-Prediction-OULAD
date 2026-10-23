# Early Academic Risk Prediction Using Machine Learning

A research-oriented machine learning project for identifying
students at risk of failure or withdrawal using the Open
University Learning Analytics Dataset (OULAD).

## Machine Learning Architecture

- XGBoost Diagnostic Agent
- LightGBM Progress Agent
- Calibrated Random Forest Contextual Agent
- XGBoost Stacking Coordinator
- Histogram Gradient Boosting Baseline
- Fixed 50:50 HGB–Coordinator Hybrid

## Experiments

- Student-grouped five-fold cross-validation
- Early prediction at 15%, 30%, and 45% course completion
- Seven specialist-subset ablations
- Paired student-clustered bootstrap evaluation
- Nested fairness evaluation for gender and disability

## Key Results

At the 45% prediction horizon:

| Model | F1-score | ROC-AUC |
|---|---:|---:|
| Stacking Coordinator | 0.7394 | 0.8819 |
| HGB Baseline | 0.7451 | 0.8858 |
| HGB–Coordinator Hybrid | 0.7473 | 0.8869 |

The original coordinator and HGB values are five-fold means.
The hybrid values are pooled out-of-fold estimates.

The hybrid produced a small ROC-AUC improvement over HGB.
The paired bootstrap analysis did not establish a clear F1 gain.

## Dataset

Open University Learning Analytics Dataset (OULAD).

Dataset reference:
https://doi.org/10.1038/sdata.2017.171

The original dataset is not included in this repository.

## Project Structure

- `main.py` — Main experiment runner
- `data_pipeline.py` — Dataset preparation
- `feature_engineering.py` — Feature generation
- `specialist_agents.py` — Specialist models
- `coordinator_fusion.py` — Stacking and baseline models
- `ablation_study.py` — Specialist ablation experiments
- `hybrid_ensemble_experiment.py` — Hybrid evaluation
- `hybrid_bootstrap_test.py` — Bootstrap confidence intervals
- `nested_fairness.py` — Nested fairness evaluation
- `statistical_analysis.py` — Statistical analysis

## Limitations

- Evaluation uses one institutional dataset.
- Assessment-grade availability is not fully verifiable.
- External institutional validation was not conducted.
- The system has not been tested through real-world
  educational interventions.