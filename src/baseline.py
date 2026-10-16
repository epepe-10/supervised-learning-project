"""Untuned experiments on the development set: what does each step buy us?

Run with:  python -m src.baseline

Everything here uses 5-fold cross-validation on the DEVELOPMENT set only.
The test set is not touched (S1).

    Experiment 1 - Feature engineering: logistic regression with vs without
                   the new features (the "with" version is the
                   "logistic_regression | no remedy" row of experiment 2).
    Experiment 2 - Model progression: the 3 models with default settings.
    Experiment 3 - Class imbalance: no remedy vs class weights vs threshold
                   tuning, for each model (S6).

Results are printed, saved to outputs/baseline_results.csv, and the figures
go to outputs/figures/.
"""

import numpy as np
import pandas as pd

from src import config
from src.data import get_dev_test_data
from src.evaluate import (
    compute_metrics,
    find_best_threshold,
    get_oof_probabilities,
    plot_confusion_matrices,
    plot_roc_and_pr_curves,
)
from src.pipeline import MODEL_NAMES, build_pipeline


def run_experiment(name, pipeline, X_dev, y_dev, threshold=0.5):
    """Cross-validate one pipeline and return (row of results, OOF probabilities)."""
    print(f"  running: {name} ...")
    oof_proba, fold_scores = get_oof_probabilities(pipeline, X_dev, y_dev)
    metrics = compute_metrics(y_dev, oof_proba, threshold)
    # (S5) The spread across folds shows how sensitive the score is to the split.
    metrics["ap_fold_std"] = np.std(fold_scores)
    return {"experiment": name, **metrics}, oof_proba


def main():
    X_dev, X_test, y_dev, y_test = get_dev_test_data()   # X_test / y_test NOT used here
    results = []

    # -----------------------------------------------------------------------
    # Experiment 1 - Feature engineering (on the linear model, which needs it)
    # -----------------------------------------------------------------------
    # Only the version WITHOUT the new features is run here. Every pipeline
    # below uses them, so the "logistic_regression | no remedy" row is the
    # version WITH them: no need to train the same model twice.
    print("\nExperiment 1 - Feature engineering")
    pipeline = build_pipeline("logistic_regression", use_new_features=False)
    row, _ = run_experiment("logistic_regression | without new features", pipeline, X_dev, y_dev)
    results.append(row)

    # -----------------------------------------------------------------------
    # Experiments 2 and 3 - Model progression and class imbalance
    # -----------------------------------------------------------------------
    # Three ways of handling the 24% / 76% imbalance (S6, least invasive first):
    #   a) no remedy:       default model, threshold 0.5
    #   b) class weights:   class_weight="balanced" gives each class the same
    #                       TOTAL weight (weight of a positive = N_neg / N_pos,
    #                       the same idea as LightGBM's scale_pos_weight). The
    #                       weights are computed inside each fit, so on the
    #                       training folds only.
    #   c) threshold tuning: model a), but the 0.5 cutoff is replaced by the
    #                       threshold that maximizes F1 on the OOF probabilities.
    #                       "Threshold tuning changes decisions without
    #                       retraining the model" (S6). Its F1 is slightly
    #                       optimistic: the threshold is chosen AND scored on
    #                       the same OOF probabilities. The honest score comes
    #                       from the test set (final_evaluation.py).
    # SMOTE is NOT used: our data has 7 categorical features and SMOTE
    # interpolates between rows; "we cannot interpolate between categories
    # such as 'France' and 'Spain'" (S6).
    print("\nExperiments 2 & 3 - Models and class imbalance")
    oof_by_model = {}
    preds_default, preds_tuned = {}, {}

    for model_name in MODEL_NAMES:
        # a) no remedy
        row, oof_proba = run_experiment(f"{model_name} | no remedy",
                                        build_pipeline(model_name), X_dev, y_dev)
        results.append(row)
        oof_by_model[model_name] = oof_proba

        # b) class weights (retrains the model -> new probabilities)
        weighted = build_pipeline(model_name, params={"class_weight": "balanced"})
        row, _ = run_experiment(f"{model_name} | class weights", weighted, X_dev, y_dev)
        results.append(row)

        # c) threshold tuning (same probabilities as a), only the cutoff changes)
        best_threshold = find_best_threshold(y_dev, oof_proba)
        metrics = compute_metrics(y_dev, oof_proba, best_threshold)
        # (no fold spread here: we re-use the probabilities of a), not new folds)
        results.append({"experiment": f"{model_name} | tuned threshold",
                        **metrics, "ap_fold_std": np.nan})

        preds_default[model_name] = (oof_proba >= 0.5).astype(int)
        preds_tuned[model_name] = (oof_proba >= best_threshold).astype(int)

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    table = pd.DataFrame(results).set_index("experiment").round(3)
    print("\nOut-of-fold results on the development set (5-fold CV):")
    print(table.to_string())

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.OUTPUT_DIR / "baseline_results.csv")

    plot_roc_and_pr_curves(y_dev, oof_by_model, "Untuned models - out-of-fold (development set)",
                           config.FIGURES_DIR / "baseline_roc_pr.png")
    plot_confusion_matrices(y_dev, preds_default, "Threshold 0.5 - out-of-fold",
                            config.FIGURES_DIR / "baseline_confusion_default.png")
    plot_confusion_matrices(y_dev, preds_tuned, "Tuned threshold (max F1) - out-of-fold",
                            config.FIGURES_DIR / "baseline_confusion_tuned.png")
    print(f"\nTable saved to {config.OUTPUT_DIR / 'baseline_results.csv'}")


if __name__ == "__main__":
    main()
