# Supervised Learning & Model Engineering: Project

End-to-end supervised learning proof of concept for tabular binary classification, going from raw data to a served prediction endpoint.

**Course:** MSA-DATB09-04 (Albert School)
**Dataset:** _TBD (Adult / Bank Marketing / Diabetes 130 Hospitals)_
**Team:** _TBD_

## Project structure

```
src/         Python source code (main implementation)
configs/     YAML configuration files
notebooks/   Supplementary notebooks (EDA, visualizations only)
report/      LaTeX report
data/        Local data cache (not versioned)
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Roadmap

- [ ] EDA & problem framing (train/val/test split)
- [ ] Feature engineering
- [ ] Class imbalance handling
- [ ] Modelling progression (3+ models)
- [ ] Single sklearn `Pipeline`
- [ ] Hyperparameter tuning with Optuna + CV
- [ ] Experiment tracking (MLflow)
- [ ] Explainability (SHAP)
- [ ] Calibration diagnostics
- [ ] Fairness & bias audit
- [ ] Prediction endpoint (FastAPI)
