"""Evaluate the three trained models ONCE on the untouched test set.

Run with:  python -m src.final_evaluation      (after python -m src.train)

(S1) "The test set is not another validation set." This script is the only
place where X_test / y_test are used. Nothing is tuned here: the models,
their hyperparameters and their thresholds were all frozen in train.py.

We report the SAME metrics for all three models (project requirement 4),
so the report can discuss what each step of the progression bought us.
"""

import joblib
import pandas as pd

from src import config
from src.data import get_dev_test_data
from src.evaluate import compute_metrics, plot_confusion_matrices, plot_roc_and_pr_curves
from src.pipeline import MODEL_NAMES


def main():
    X_dev, X_test, y_dev, y_test = get_dev_test_data()
    final_model = (config.MODELS_DIR / "final_model.txt").read_text()

    results = []
    probas, predictions = {}, {}
    for model_name in MODEL_NAMES:
        saved = joblib.load(config.MODELS_DIR / f"{model_name}.joblib")
        pipeline, threshold = saved["pipeline"], saved["threshold"]

        # predict_proba returns [P(<=50K), P(>50K)]: we keep the second column.
        proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, proba, threshold)
        results.append({"model": model_name, **metrics})

        probas[model_name] = proba
        predictions[model_name] = (proba >= threshold).astype(int)

    # Reference row (S6): a model that always predicts "<=50K".
    prevalence = y_test.mean()
    print(f"Test set: {len(y_test)} rows, prevalence of >50K = {100 * prevalence:.1f}%")
    print(f"Accuracy of 'always predict <=50K' = {100 * (1 - prevalence):.1f}%\n")

    table = pd.DataFrame(results).set_index("model").round(4)
    print("Test-set results (each model evaluated once, with its frozen threshold):")
    print(table.to_string())
    print(f"\nFinal model, chosen on cross-validation in train.py: {final_model}")
    table.to_csv(config.OUTPUT_DIR / "test_results.csv")

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_roc_and_pr_curves(y_test, probas, "Tuned models - test set",
                           config.FIGURES_DIR / "test_roc_pr.png")
    plot_confusion_matrices(y_test, predictions, "Tuned models - test set (frozen thresholds)",
                            config.FIGURES_DIR / "test_confusion.png")


if __name__ == "__main__":
    main()
