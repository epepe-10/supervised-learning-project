"""Train the final version of each tuned model on the development set.

Run with:  python -m src.train      (after python -m src.tune)

For each of the three models:
    1. read the best hyperparameters found by Optuna (tune.py),
    2. compute out-of-fold (OOF) probabilities with 5-fold CV on the dev set,
    3. choose the decision threshold on these OOF probabilities (S6),
    4. refit the Pipeline on the WHOLE development set (S1, S5),
    5. save the fitted Pipeline and its threshold to outputs/models/.

The final model is chosen here, on cross-validation results, NOT on the
test set: "Never use the final test set to make additional modelling
decisions" (S5).
"""

import json

import joblib
import pandas as pd

from src import config, tracking
from src.data import get_dev_test_data
from src.evaluate import compute_metrics, find_best_threshold, get_oof_probabilities
from src.pipeline import MODEL_NAMES, build_pipeline


def load_best_params(model_name):
    """Read the best hyperparameters saved by tune.py."""
    path = config.TUNING_DIR / f"{model_name}_best_params.json"
    with open(path) as file:
        return json.load(file)


def main():
    X_dev, X_test, y_dev, y_test = get_dev_test_data()   # X_test / y_test NOT used here
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for model_name in MODEL_NAMES:
        print(f"\n{model_name}")
        params = load_best_params(model_name)
        pipeline = build_pipeline(model_name, params)

        # Step 2 - honest probabilities: each row is predicted by a model that
        # did not train on it.
        oof_proba, fold_scores = get_oof_probabilities(pipeline, X_dev, y_dev)

        # Step 3 - threshold chosen on OOF probabilities, then frozen (S6):
        # "Freeze the threshold; evaluate once on the untouched test set."
        threshold = find_best_threshold(y_dev, oof_proba)
        metrics = compute_metrics(y_dev, oof_proba, threshold)
        results.append({"model": model_name, **metrics})
        print(f"  OOF Average Precision = {metrics['average_precision']:.4f}, "
              f"threshold = {threshold:.2f}")

        # Step 4 - "Once the model is chosen, fit on the complete development set" (S1).
        pipeline.fit(X_dev, y_dev)

        # Step 5 - save the fitted Pipeline AND its threshold: both are needed
        # to turn a raw census row into a decision.
        model_path = config.MODELS_DIR / f"{model_name}.joblib"
        joblib.dump({"pipeline": pipeline, "threshold": threshold}, model_path)

        # Requirement 7: log params, OOF metrics, threshold and model file to MLflow.
        tracking.log_final_training(model_name, params, metrics, model_path)

    # Compare the three tuned models on cross-validation (same metrics for all).
    table = pd.DataFrame(results).set_index("model").round(4)
    print("\nTuned models - out-of-fold results on the development set:")
    print(table.to_string())
    table.to_csv(config.OUTPUT_DIR / "tuned_cv_results.csv")

    # Final model = best OOF Average Precision (our selection metric, S6).
    final_model = table["average_precision"].idxmax()
    (config.MODELS_DIR / "final_model.txt").write_text(final_model)
    print(f"\nFinal model (best CV Average Precision): {final_model}")


if __name__ == "__main__":
    main()
