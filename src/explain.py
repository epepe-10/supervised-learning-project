"""Explain the final model with SHAP (project requirement 8).

Run with:  python -m src.explain      (after python -m src.train)

Requirement: "Produce global feature importance [...] with SHAP"

(S8) SHAP splits ONE prediction into a reference value plus one contribution
per feature:
    model output(x) = base value + SHAP_1 + SHAP_2 + ... + SHAP_p
For LightGBM, TreeExplainer explains the raw score, i.e. the log-odds of
>50K (positive = pushes towards >50K, negative = towards <=50K).
(S8) SHAP explains what the MODEL does, it does not prove causality.

We explain rows of the development set: the test set stays untouched until
final_evaluation.py (S1).

Outputs, in outputs/figures/:
    shap_global_importance.png   mean |SHAP| per original feature
    shap_beeswarm.png            SHAP value of every row, per model column
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src import config
from src.data import get_dev_test_data

N_ROWS = 1000   # rows explained: enough for a stable global picture, still fast


def original_feature(column):
    """Name of the original feature behind a model column.

    One-hot encoding turned e.g. "relationship" into several columns
    ("relationship_Husband", "relationship_Wife", ...). Numerical columns
    keep their name.
    """
    for feature in config.CATEGORICAL_FEATURES:
        if column.startswith(feature + "_"):
            return feature
    return column


def group_by_original_feature(shap_values, columns):
    """Sum the SHAP values of the one-hot columns of each categorical feature.

    (S8) SHAP values are additive, so the contribution of "relationship" for
    one person is the sum of the contributions of its one-hot columns.
    Returns a table: one row per person, one column per original feature.
    """
    table = pd.DataFrame(shap_values, columns=[original_feature(c) for c in columns])
    return table.T.groupby(level=0, sort=False).sum().T


def save_current_figure(name):
    """Save the figure SHAP just drew in outputs/figures/, then close it."""
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    X_dev, X_test, y_dev, y_test = get_dev_test_data()   # X_test / y_test NOT used here
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # The final model chosen in train.py (on CV Average Precision).
    final_model = (config.MODELS_DIR / "final_model.txt").read_text()
    saved = joblib.load(config.MODELS_DIR / f"{final_model}.joblib")
    pipeline, threshold = saved["pipeline"], saved["threshold"]
    print(f"Final model: {final_model} (threshold = {threshold:.2f})")

    # 1. The matrix the model really sees: features + preprocessing applied.
    #    pipeline[:-1] = every step except the model. One-hot encoding gives a
    #    sparse matrix, turned into a normal (dense) one for SHAP.
    X_sample = X_dev.sample(N_ROWS, random_state=config.RANDOM_STATE)
    X_model = pipeline[:-1].transform(X_sample).toarray()
    # Column names without the ColumnTransformer prefix ("num__age" -> "age").
    columns = [name.split("__", 1)[1] for name in pipeline[:-1].get_feature_names_out()]

    # 2. SHAP values. TreeExplainer computes them exactly for tree models (S8).
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = explainer(pd.DataFrame(X_model, columns=columns))

    # 3. Global importance (S8): mean |SHAP| over many people = how much a
    #    feature moves the predictions on average, whatever the direction.
    per_feature = group_by_original_feature(shap_values.values, columns)
    importance = per_feature.abs().mean().sort_values(ascending=False)
    print("\nGlobal importance - mean |SHAP| (log-odds), per original feature:")
    print(importance.round(3).to_string())

    fig, ax = plt.subplots(figsize=(7, 5))
    importance.head(12).iloc[::-1].plot.barh(ax=ax)   # most important at the top
    ax.set_xlabel("mean |SHAP value| (log-odds)")
    ax.set_title(f"Global feature importance - {final_model}")
    save_current_figure("shap_global_importance.png")

    # Beeswarm (S8): one dot = one person. Position = SHAP value, colour =
    # value of the column (red = high). Shows the DIRECTION of each effect.
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    save_current_figure("shap_beeswarm.png")


if __name__ == "__main__":
    main()
