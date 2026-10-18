"""Experiment tracking with MLflow (project requirement 7).

Requirement: "Log every run, parameters, metrics, and artifacts, for all
three models across your tuning trials."

How the runs are organised in MLflow (one experiment for the whole project):

    lightgbm - tuning             parent run: best CV score, best params, tuning files
        lightgbm - trial 0        child run: params + CV scores of one Optuna trial
        lightgbm - trial 1
        ...
    lightgbm - final training     train.py: OOF metrics, threshold, saved model

Runs are stored locally in mlflow.db and the files in mlruns/ (both
git-ignored). To browse them, run from the project root:
    mlflow ui        then open http://127.0.0.1:5000
"""

import mlflow
import numpy as np

from src import config

# Where MLflow writes the runs: a local SQLite file at the project root.
mlflow.set_tracking_uri(f"sqlite:///{(config.ROOT_DIR / 'mlflow.db').as_posix()}")
mlflow.set_experiment("adult-income")


def start_tuning_run(model_name):
    """Open the parent run that groups all the Optuna trials of one model."""
    mlflow.start_run(run_name=f"{model_name} - tuning")
    mlflow.set_tags({"model": model_name, "stage": "tuning"})


def log_trial(model_name, trial_number, params, fold_scores):
    """Log one Optuna trial as a child run of the tuning run.

    (S5) "One trial = one configuration + one CV score": we keep the
    configuration (params) and the score of each of the 5 folds, so we can
    also see how much the score depends on the split (std).
    """
    with mlflow.start_run(run_name=f"{model_name} - trial {trial_number}", nested=True):
        mlflow.set_tags({"model": model_name, "stage": "trial"})
        mlflow.log_params(params)
        mlflow.log_metric("cv_ap_mean", np.mean(fold_scores))
        mlflow.log_metric("cv_ap_std", np.std(fold_scores))
        for fold, score in enumerate(fold_scores, start=1):
            mlflow.log_metric(f"cv_ap_fold_{fold}", score)


def end_tuning_run(model_name, best_value, best_params):
    """Log the result of the study and the files saved by tune.py, then close the run."""
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_ap", best_value)
    # Artifacts = files attached to the run: the same ones tune.py writes.
    for suffix in ["_best_params.json", "_trials.csv", "_history.png"]:
        mlflow.log_artifact(str(config.TUNING_DIR / f"{model_name}{suffix}"))
    mlflow.end_run()


def log_final_training(model_name, params, metrics, model_path):
    """Log the final training of one model (train.py): tuned params,
    out-of-fold metrics, chosen threshold and the saved model file."""
    with mlflow.start_run(run_name=f"{model_name} - final training"):
        mlflow.set_tags({"model": model_name, "stage": "final_training"})
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)   # OOF metrics from compute_metrics(), incl. the threshold
        mlflow.log_artifact(str(model_path))
