"""Hyperparameter tuning with Optuna and 5-fold cross-validation.

Run with:   python -m src.tune                      (all three models)
            python -m src.tune --model lightgbm     (one model only)

(S5) "One trial = one configuration + one CV score." For every trial:
    1. Optuna proposes a set of hyperparameters,
    2. we build the COMPLETE Pipeline (features + preprocessing + model),
    3. we evaluate it with 5-fold stratified CV on the development set,
    4. we return the mean Average Precision to Optuna.
Optuna's TPE sampler uses the previous trials to choose the next ones, but
"does not guarantee that the next trial will perform better" (S5).

Why Average Precision? (S6) With a minority positive class, AP summarizes
how well the model retrieves the >50K people. It does not depend on a
threshold, which is chosen afterwards in train.py.

Important (S5): the best CV score is NOT our final performance: "The same
validation folds helped us choose the winner." The final, honest score
comes from the untouched test set (final_evaluation.py).

Outputs, in outputs/tuning/:
    <model>_best_params.json   best hyperparameters, read by train.py
    <model>_trials.csv         every trial: parameters + CV score
    <model>_history.png        CV score of each trial
Every trial is also logged to MLflow (see tracking.py).
"""

import argparse
import json

import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import cross_val_score

from src import config, tracking
from src.data import get_dev_test_data
from src.evaluate import get_cv_folds
from src.pipeline import MODEL_NAMES, build_pipeline


def suggest_params(trial, model_name, class_ratio):
    """Search space of each model: which hyperparameters, and in which range.

    class_ratio = N_negative / N_positive on the development set (~3.2).
    (S6) "Class weights are hyperparameters to validate": the weighting is
    part of the search, from no weighting at all up to the class ratio.
    """
    if model_name == "logistic_regression":
        return {
            # (S2) C = 1 / lambda: small C = strong regularization.
            # log=True: values are tried on a log scale (0.001, 0.01, 0.1, ...).
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            # (S2) 0 = Ridge (L2), 1 = Lasso (L1), in between = Elastic Net.
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    if model_name == "random_forest":
        return {
            # (S4) More trees = more stable average (not more overfitting).
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            # (S3/S4) Complexity of each tree: depth and minimum leaf size.
            "max_depth": trial.suggest_int("max_depth", 4, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            # (S4) Share of features tried at each split: smaller = more diverse trees.
            "max_features": trial.suggest_float("max_features", 0.1, 0.6),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    if model_name == "lightgbm":
        return {
            # (S4) "learning_rate and n_estimators must be considered together."
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            # (S4) Complexity of each tree (LightGBM grows trees leaf-wise).
            "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
            # (S4) Extra regularization: row / column sampling and L2 penalty.
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,   # needed for subsample to be applied
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            # (S6) Class weighting inside LightGBM, from 1 (none) to the class ratio.
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, class_ratio),
        }

    raise ValueError(f"Unknown model: {model_name}")


def tune_model(model_name, X_dev, y_dev, n_trials):
    """Run one Optuna study for one model and save its results."""
    class_ratio = (y_dev == 0).sum() / (y_dev == 1).sum()

    def objective(trial):
        params = suggest_params(trial, model_name, class_ratio)
        pipeline = build_pipeline(model_name, params)
        # cross_val_score fits a fresh copy of the Pipeline on each training
        # fold, so preprocessing never sees the validation fold (S5).
        # n_jobs=-1: the 5 folds are trained at the same time on different CPU cores.
        scores = cross_val_score(pipeline, X_dev, y_dev, cv=get_cv_folds(),
                                 scoring="average_precision", n_jobs=-1)
        print(f"  trial {trial.number}: mean CV AP = {scores.mean():.4f}")
        tracking.log_trial(model_name, trial.number, params, scores)   # requirement 7
        return scores.mean()

    # direction="maximize": a higher Average Precision is better.
    # seed: the same trials are proposed every time we re-run the script.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
        study_name=model_name,
    )
    tracking.start_tuning_run(model_name)   # requirement 7: one MLflow run groups all trials
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Save the best hyperparameters (read by train.py) and every trial.
    config.TUNING_DIR.mkdir(parents=True, exist_ok=True)
    best_params = suggest_params(optuna.trial.FixedTrial(study.best_params), model_name, class_ratio)
    with open(config.TUNING_DIR / f"{model_name}_best_params.json", "w") as file:
        json.dump(best_params, file, indent=2)
    study.trials_dataframe().to_csv(config.TUNING_DIR / f"{model_name}_trials.csv", index=False)

    # Plot: CV score of every trial, and the best score found so far.
    scores = [t.value for t in study.trials]
    best_so_far = [max(scores[: i + 1]) for i in range(len(scores))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(range(len(scores)), scores, s=15, label="trial CV score")
    ax.plot(best_so_far, color="red", label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel("mean CV Average Precision")
    ax.set_title(f"Optuna history - {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.TUNING_DIR / f"{model_name}_history.png", dpi=150)
    plt.close(fig)
    tracking.end_tuning_run(model_name, study.best_value, best_params)   # requirement 7

    print(f"{model_name}: best mean CV AP = {study.best_value:.4f} after {n_trials} trials")
    print(f"  best params: {best_params}")
    return study


def main():
    parser = argparse.ArgumentParser(description="Tune the models with Optuna.")
    parser.add_argument("--model", choices=MODEL_NAMES, help="tune only this model")
    parser.add_argument("--trials", type=int, help="override the number of trials")
    args = parser.parse_args()

    # Only print one line per trial instead of Optuna's long default logs.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_dev, X_test, y_dev, y_test = get_dev_test_data()   # X_test / y_test NOT used here
    model_names = [args.model] if args.model else MODEL_NAMES
    for model_name in model_names:
        n_trials = args.trials or config.N_TRIALS[model_name]
        print(f"\nTuning {model_name} ({n_trials} trials x {config.N_FOLDS} folds)...")
        tune_model(model_name, X_dev, y_dev, n_trials)


if __name__ == "__main__":
    main()
