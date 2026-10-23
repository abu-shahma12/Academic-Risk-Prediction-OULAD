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


  ## 👥 Project Contributors

This research project was developed collaboratively by a team from the **Apex Institute of Technology, Chandigarh University, Mohali, India**, under academic supervision.

| Name                 | Role                         | Contributions                                                                                          |
| -------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Somdatta Patra**   | Research Supervisor          | Academic supervision, research guidance, and project oversight.                                        |
| **Bedant Singhania** | Research and Documentation   | Research findings, literature review, preparation of supporting documents, and research paper writing. |
| **Misha Kumari**     | Research and Documentation   | Research findings, preparation of supporting documents, and research paper writing.                    |
| **Abu Shahma Khan**  | Machine Learning Development | Python programming, model implementation, training, testing, and experimental evaluation.              |
| **Lakshit Soni**     | Research and Documentation   | Research findings, preparation of supporting documents, and research paper writing.                    |

### 🛠️ Technical Implementation

The machine learning implementation was carried out by **Abu Shahma Khan**, including:

* Development and implementation of the machine learning pipeline.
* Implementation of XGBoost, LightGBM, and Random Forest specialist models.
* Development and testing of the stacking coordinator and hybrid ensemble.
* Execution of model training and evaluation experiments.
* Implementation of specialist-subset ablation experiments.
* Statistical evaluation and bootstrap analysis.
* Implementation and evaluation of fairness-aware threshold adjustments.

### 📚 Research and Documentation

**Bedant Singhania, Misha Kumari, and Lakshit Soni** contributed to the research and documentation activities, including:

* Research findings and literature review.
* Preparation of supporting research documents.
* Academic manuscript preparation and writing.

### 🎓 Academic Supervision

**Somdatta Patra** provided academic supervision and guidance throughout the research project.
