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

Outputs:
    outputs/calibration_bins.csv                 one row per probability bin
    outputs/figures/calibration_reliability.png  reliability diagram + histogram
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src import config
from src.data import get_dev_test_data
from src.evaluate import get_oof_probabilities
from src.pipeline import build_pipeline
from src.train import load_best_params

N_BINS = 10   # (S7) 10 equal-width bins: [0, 0.1), [0.1, 0.2), ..., [0.9, 1]


def reliability_curve(y_true, y_proba, n_bins=N_BINS):
    """Points of the reliability diagram, one per non-empty bin (S7):

        conf(B_m) = mean predicted probability of the people in bin m
        acc(B_m)  = observed share of >50K among the people in bin m
        counts    = number of people in bin m
    """
    y_true = np.asarray(y_true)
    edges = np.linspace(0, 1, n_bins + 1)
    # Bin number of each probability (0 for [0, 0.1], 1 for ]0.1, 0.2], ...).
    bin_ids = np.searchsorted(edges[1:-1], y_proba)
    conf, acc, counts = [], [], []
    for m in range(n_bins):
        in_bin = bin_ids == m
        if in_bin.any():   # empty bins are skipped
            conf.append(y_proba[in_bin].mean())
            acc.append(y_true[in_bin].mean())
            counts.append(in_bin.sum())
    return np.array(conf), np.array(acc), np.array(counts)


def expected_calibration_error(y_true, y_proba, n_bins=N_BINS):
    """(S7) ECE = sum over bins of (|B_m| / n) * |acc(B_m) - conf(B_m)|:
    the calibration gap of each bin, weighted by its share of people."""
    conf, acc, counts = reliability_curve(y_true, y_proba, n_bins)
    return np.sum(counts / counts.sum() * np.abs(acc - conf))


def plot_reliability(conf, acc, counts, y_proba, title, path):
    """Reliability diagram (left) next to the histogram of predicted
    probabilities (right): (S7) "Pair it with a histogram", to see where the
    model actually puts its predictions."""
    fig, (ax_curve, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))

    ax_curve.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    # Marker size proportional to the number of people in the bin (S7):
    # a point built on few people is weaker evidence.
    ax_curve.plot(conf, acc, color="tab:blue", label="final model")
    ax_curve.scatter(conf, acc, s=300 * counts / counts.max(), color="tab:blue")
    ax_curve.set_xlabel("mean predicted probability")
    ax_curve.set_ylabel("observed share of >50K")
    ax_curve.set_title("Reliability diagram")
    ax_curve.legend()

    ax_hist.hist(y_proba, bins=N_BINS, range=(0, 1), edgecolor="white")
    ax_hist.set_xlabel("predicted probability of >50K")
    ax_hist.set_ylabel("number of people")
    ax_hist.set_title("Where the predictions are")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    X_dev, X_test, y_dev, y_test = get_dev_test_data()   # X_test / y_test NOT used here

    # 1. Out-of-fold probabilities of the final model (chosen in train.py),
    #    rebuilt with its tuned hyperparameters.
    final_model = (config.MODELS_DIR / "final_model.txt").read_text()
    pipeline = build_pipeline(final_model, load_best_params(final_model))
    oof_proba, _ = get_oof_probabilities(pipeline, X_dev, y_dev)
    print(f"Final model: {final_model} - out-of-fold probabilities on {len(y_dev)} rows")

    # 2. Reliability curve, written by hand and checked against scikit-learn.
    conf, acc, counts = reliability_curve(y_dev, oof_proba)
    sk_acc, sk_conf = calibration_curve(y_dev, oof_proba, n_bins=N_BINS, strategy="uniform")
    assert np.allclose(conf, sk_conf) and np.allclose(acc, sk_acc), "reliability curve mismatch"

    # gap > 0: the model predicts MORE >50K than observed in this bin.
    table = pd.DataFrame({"mean_predicted": conf, "observed_rate": acc,
                          "gap": conf - acc, "n_people": counts})
    print("\nReliability table (10 equal-width bins):")
    print(table.round(3).to_string(index=False))
    table.to_csv(config.OUTPUT_DIR / "calibration_bins.csv", index=False)

    # 3. Scores of probability quality (S7), lower is better for both.
    #    Reference for the Brier score: always predicting the prevalence is
    #    perfectly calibrated on average but cannot rank anybody.
    prevalence = y_dev.mean()
    brier = brier_score_loss(y_dev, oof_proba)
    ece = expected_calibration_error(y_dev, oof_proba)
    print(f"\nBrier score = {brier:.4f}  (always predicting {prevalence:.3f}: "
          f"{prevalence * (1 - prevalence):.4f})")
    print(f"ECE         = {ece:.4f}")

    # 4. Direction of the miscalibration: compare what the model predicts on
    #    average with what is observed.
    direction = "over" if oof_proba.mean() > prevalence else "under"
    print(f"\nMean predicted P(>50K) = {oof_proba.mean():.3f} vs observed share = "
          f"{prevalence:.3f}: the model {direction}-estimates the chance of >50K.")

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_reliability(conf, acc, counts, oof_proba,
                     f"{final_model} - out-of-fold calibration "
                     f"(Brier = {brier:.3f}, ECE = {ece:.3f})",
                     config.FIGURES_DIR / "calibration_reliability.png")


if __name__ == "__main__":
    main()
