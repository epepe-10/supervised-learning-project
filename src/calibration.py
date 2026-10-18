"""Calibration diagnostics of the final model (project requirement 9).

Run with:  python -m src.calibration      (after python -m src.train)

Requirement: "For your final model, plot a reliability curve and report the
Brier score (and/or ECE). State whether the model is over- or under-confident,
and whether you'd trust its probabilities for a downstream decision as-is."

(S7) Ranking and calibration are two different questions:
    ranking      -> do >50K people get higher scores?      (ROC-AUC, AP)
    calibration  -> when the model says 0.7, is it right ~70% of the time?
Our class weighting (scale_pos_weight, tuned by Optuna) gives more weight to
the >50K people during training, which can push the probabilities up.

(S7) "Calibrate on unseen predictions": we judge calibration on the
out-of-fold probabilities of the development set (each row predicted by a
model that did not train on it). The test set stays untouched (S1).
"""

from src import config
from src.data import get_dev_test_data
from src.evaluate import get_oof_probabilities
from src.pipeline import build_pipeline
from src.train import load_best_params


def main():
    X_dev, X_test, y_dev, y_test = get_dev_test_data()   # X_test / y_test NOT used here

    # 1. Out-of-fold probabilities of the final model (chosen in train.py),
    #    rebuilt with its tuned hyperparameters.
    final_model = (config.MODELS_DIR / "final_model.txt").read_text()
    pipeline = build_pipeline(final_model, load_best_params(final_model))
    oof_proba, _ = get_oof_probabilities(pipeline, X_dev, y_dev)
    print(f"Final model: {final_model} - out-of-fold probabilities on {len(y_dev)} rows")

    # 2. Direction of the miscalibration: compare what the model predicts on
    #    average with what is observed.
    prevalence = y_dev.mean()
    direction = "over" if oof_proba.mean() > prevalence else "under"
    print(f"\nMean predicted P(>50K) = {oof_proba.mean():.3f} vs observed share = "
          f"{prevalence:.3f}: the model {direction}-estimates the chance of >50K.")


if __name__ == "__main__":
    main()
