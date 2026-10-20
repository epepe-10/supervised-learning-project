# Supervised Learning & Model Engineering: Project

End-to-end supervised learning proof of concept for tabular binary classification, going from raw data to a tuned, explained and calibrated final model evaluated on an untouched test set.

- **Course:** MSA-DATB09-04 (Albert School), Ilana Sebag
- **Dataset:** Adult / Census Income (OpenML, `name="adult", version=2`). The task is to predict whether a person's annual income is above 50K$.
- **Team:** Eduardo Pepe, Louis Pimont-Nishimura, Jules Faucher

## Project structure

```
src/
  config.py            constants: seed, split sizes, column lists, Optuna budget
  data.py              load Adult, clean (duplicates, 0/1 target), stratified 80/20 dev/test split
  eda.py               exploratory analysis -> terminal findings + outputs/figures/
  features.py          3 engineered features (square term, interaction, indicator)
  pipeline.py          full sklearn Pipeline: features -> ColumnTransformer -> model
  evaluate.py          metrics, out-of-fold cross-validation, threshold choice, plots
  baseline.py          untuned experiments: feature engineering, models, imbalance remedies
  tune.py              Optuna + stratified 5-fold CV for each model
  train.py             OOF threshold, refit on the full dev set, save the models
  tracking.py          MLflow logging of every tuning trial and final training
  explain.py           SHAP global importance + local explanations of the final model
  calibration.py       reliability diagram, Brier score and ECE of the final model
  final_evaluation.py  single evaluation of the 3 models on the untouched test set
configs/               (reserved for YAML configuration)
notebooks/             supplementary notebooks (visualizations only)
report/                LaTeX report
data/                  OpenML download cache (not versioned)
outputs/               figures, tables, tuning results, saved models (not versioned)
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

Run from the project root, in this order:

```bash
python -m src.eda               # 1. EDA                               (~10 s)
python -m src.baseline          # 2. untuned CV experiments            (~5 min)
python -m src.tune              # 3. Optuna tuning, all 3 models       (~1 h)
python -m src.tune --model lightgbm --trials 10   # (tune one model / fewer trials)
python -m src.train             # 4. thresholds + refit + save models  (~3 min)
python -m src.explain           # 5. SHAP explanations, final model    (~10 s)
python -m src.calibration       # 6. calibration diagnostics           (~1 min)
python -m src.final_evaluation  # 7. one evaluation on the test set    (~10 s)
```

Tuning and training runs are logged to MLflow (`mlflow.db` + `mlruns/`, not versioned). To browse them:

```bash
mlflow ui                       # then open http://127.0.0.1:5000
```

## Methodology at a glance

| Step | Choice | Course session |
|---|---|---|
| Split | Stratified 80% development / 20% test. The test set is used only in `final_evaluation.py` | S1 |
| Validation | Stratified 5-fold CV on the development set, Pipeline refitted in each fold | S1, S5 |
| Cleaning | Drop 52 duplicate rows; drop `education` (same as `education-num`) and `fnlwgt` (survey weight) | S1, S2 |
| Missing values | Median / most-frequent imputation inside the Pipeline | S5 |
| Encoding | `StandardScaler` for numerical columns, `OneHotEncoder(handle_unknown="ignore")` for categorical ones | S2, S5 |
| New features | `age_squared`, `education_x_hours`, `has_capital` | S3 |
| Models | Logistic regression (Ridge / Lasso / Elastic Net) → Random Forest → LightGBM | S1–S5 |
| Imbalance | Class weights tuned as a hyperparameter + decision threshold chosen on out-of-fold scores (max F1); no SMOTE (categorical data) | S6 |
| Tuning | Optuna (TPE), mean CV Average Precision as the objective | S5, S6 |
| Tracking | MLflow: one run per Optuna trial (params + AP of each fold), grouped under one run per model (best params + tuning files); one run per final training (OOF metrics, threshold, model) | S5 |
| Explainability | SHAP `TreeExplainer` on the final model; one-hot SHAP values summed per original feature; global mean \|SHAP\| + beeswarm; waterfall plots for 3 people | S8 |
| Calibration | Final model, out-of-fold probabilities: reliability diagram (10 equal-width bins) + histogram, Brier score, ECE | S7 |

## Roadmap

- [x] EDA & problem framing (train/val/test split)
- [x] Feature engineering
- [x] Class imbalance handling
- [x] Modelling progression (3 models)
- [x] Single sklearn `Pipeline`
- [x] Hyperparameter tuning with Optuna + CV
- [x] Experiment tracking
- [x] Explainability (SHAP)
- [x] Calibration diagnostics
- [x] Final evaluation on the test set
