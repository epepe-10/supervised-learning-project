"""What regularization does to the logistic regression (S2).

Run with:  python -m src.regularization      (after python -m src.train)

(S2) The penalized loss is  log-loss + lambda * penalty(coefficients),
with C = 1 / lambda in scikit-learn (small C = strong penalty). The L1 part
(Lasso) can set coefficients exactly to zero; the L2 part (Ridge) only
shrinks them.

Two analyses, on the development set only (the test set stays untouched, S1):
    1. The tuned model (C = 1.89, l1_ratio = 0.77, mostly Lasso): which
       coefficients did the penalty set to zero?
    2. Regularization path: train and validation AP (5-fold CV) for C from
       1e-4 to 100, l1_ratio kept at its tuned value. Small C = underfitting,
       large C = the penalty disappears.

Outputs:
    outputs/regularization_coefficients.csv   coefficient of every model column
    outputs/regularization_path.csv           train / validation AP per C
    outputs/figures/regularization_path.png   the path, plotted
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from src import config
from src.data import get_dev_test_data
from src.evaluate import get_cv_folds
from src.pipeline import build_pipeline

C_VALUES = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]


def main():
    X_dev, X_test, y_dev, y_test = get_dev_test_data()   # X_test / y_test NOT used here
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Coefficients of the tuned logistic regression (fitted on the dev set by train.py).
    pipeline = joblib.load(config.MODELS_DIR / "logistic_regression.joblib")["pipeline"]
    # Column names without the ColumnTransformer prefix ("num__age" -> "age").
    preprocessor = pipeline.named_steps["preprocessor"]
    columns = [name.split("__", 1)[1] for name in preprocessor.get_feature_names_out()]
    coefficients = pd.Series(pipeline.named_steps["model"].coef_[0], index=columns)
    coefficients = coefficients.reindex(coefficients.abs().sort_values(ascending=False).index)
    coefficients.rename("coefficient").to_csv(config.OUTPUT_DIR / "regularization_coefficients.csv")

    zero = coefficients[coefficients == 0]
    print(f"Tuned logistic regression: {len(zero)} of {len(coefficients)} "
          f"coefficients set to exactly 0 by the L1 part of the penalty:")
    for column in zero.index:
        print(f"  {column}")
    print("\nLargest coefficients (scaled / one-hot columns, log-odds):")
    print(coefficients.head(10).round(3).to_string())

    # 2. Regularization path: same Pipeline, same folds, only C changes.
    with open(config.TUNING_DIR / "logistic_regression_best_params.json") as file:
        best_params = json.load(file)

    rows = []
    for C in C_VALUES:
        params = {**best_params, "C": C}
        scores = cross_validate(build_pipeline("logistic_regression", params), X_dev, y_dev,
                                cv=get_cv_folds(), scoring="average_precision",
                                return_train_score=True, return_estimator=True, n_jobs=-1)
        # Number of non-zero coefficients, averaged over the 5 fitted folds.
        n_nonzero = np.mean([(e.named_steps["model"].coef_ != 0).sum()
                             for e in scores["estimator"]])
        rows.append({"C": C,
                     "train_ap": scores["train_score"].mean(),
                     "valid_ap": scores["test_score"].mean(),
                     "valid_ap_std": scores["test_score"].std(),
                     "n_nonzero": n_nonzero})
        print(f"C = {C:g}: train AP = {rows[-1]['train_ap']:.4f}, "
              f"valid AP = {rows[-1]['valid_ap']:.4f}, non-zero coefficients = {n_nonzero:.0f}")

    path = pd.DataFrame(rows)
    path.round(4).to_csv(config.OUTPUT_DIR / "regularization_path.csv", index=False)

    # Left: train vs validation AP. Right: how many coefficients survive.
    fig, (ax_ap, ax_nz) = plt.subplots(1, 2, figsize=(11, 4))
    ax_ap.plot(path["C"], path["train_ap"], marker="o", label="train (5-fold mean)")
    ax_ap.plot(path["C"], path["valid_ap"], marker="o", label="validation (5-fold mean)")
    ax_nz.plot(path["C"], path["n_nonzero"], marker="o", color="tab:gray")
    for ax in (ax_ap, ax_nz):
        ax.set_xscale("log")
        ax.set_xlabel("C = 1 / lambda (log scale, right = weaker penalty)")
        ax.axvline(best_params["C"], color="black", linestyle="--", linewidth=1,
                   label=f"tuned C = {best_params['C']:.2f}")
    ax_ap.set_ylabel("Average Precision")
    ax_ap.set_title("Train and validation AP")
    ax_ap.legend()
    ax_nz.set_ylabel(f"non-zero coefficients (of {len(coefficients)})")
    ax_nz.set_title(f"Coefficients kept (l1_ratio = {best_params['l1_ratio']:.2f})")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "regularization_path.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
