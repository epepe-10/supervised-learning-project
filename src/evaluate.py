"""Metrics and plots, identical for every model so they can be compared.

(S6) "Optimize the metric that reflects the deployment objective; report
complementary metrics too."

    Question                                  Metric
    How many >50K people do we find?          Recall
    How trustworthy is a ">50K" prediction?   Precision
    One balance of precision and recall?      F1
    How well are people ranked?               ROC-AUC
    How good is minority-class retrieval?     Average Precision (PR-AUC)
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config


def get_cv_folds():
    """The 5 stratified folds used everywhere, always the same thanks to the seed (S1)."""
    return StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def get_oof_probabilities(pipeline, X, y):
    """5-fold cross-validation that returns the out-of-fold (OOF) probabilities.

    (S5) "Each run: train on four parts, validate on the fifth. Every
    observation is used for validation exactly once."
    (S5) "Define the Pipeline before cross-validation, but fit a fresh copy
    of it inside each CV fold": clone() makes that fresh, unfitted copy, so
    the imputer, scaler and encoder never see the validation fold.

    Returns:
        oof_proba:  P(>50K) for every row, predicted by the model that did NOT
                    train on that row -> honest scores, usable to pick a threshold.
        fold_scores: Average Precision of each fold, to see how much the score
                    depends on the split.
    """
    oof_proba = np.zeros(len(y))
    fold_scores = []
    for train_index, valid_index in get_cv_folds().split(X, y):
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X.iloc[train_index], y.iloc[train_index])
        proba = fold_pipeline.predict_proba(X.iloc[valid_index])[:, 1]
        oof_proba[valid_index] = proba
        fold_scores.append(average_precision_score(y.iloc[valid_index], proba))
    return oof_proba, fold_scores


def compute_metrics(y_true, y_proba, threshold=0.5):
    """All metrics for one model, from its predicted probabilities.

    y_proba is P(income > 50K), i.e. predict_proba(X)[:, 1].
    (S1) "A probability is not yet a class": the threshold turns it into one.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        # Threshold-free metrics: they judge the probabilities / the ranking.
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
        # Threshold-dependent metrics: they judge the final decisions.
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "threshold": threshold,
    }


def find_best_threshold(y_true, y_proba):
    """Threshold that maximizes F1, searched on a grid from 0.05 to 0.95.

    (S6) Must be called on OUT-OF-FOLD validation scores, never on the test
    set: "That would leak test information into model selection."
    We use F1 because we have no real business costs for errors on this
    dataset; F1 assumes false positives and false negatives cost about the same.
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    f1_scores = [f1_score(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    best_index = int(np.argmax(f1_scores))
    return float(thresholds[best_index])


def plot_roc_and_pr_curves(y_true, probas_by_model, title, path):
    """ROC curve and Precision-Recall curve of several models, side by side.

    probas_by_model is a dictionary {model name: predicted probabilities}.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    for name, y_proba in probas_by_model.items():
        RocCurveDisplay.from_predictions(y_true, y_proba, name=name, ax=ax_roc)
        PrecisionRecallDisplay.from_predictions(y_true, y_proba, name=name, ax=ax_pr)

    # Reference lines: a random model gives the diagonal on the ROC curve (AUC = 0.5)
    # and a flat line at the prevalence on the PR curve (S6).
    ax_roc.plot([0, 1], [0, 1], "k--", label="random")
    ax_pr.axhline(np.mean(y_true), color="k", linestyle="--", label="random (prevalence)")
    ax_roc.set_title("ROC curve")
    ax_pr.set_title("Precision-Recall curve")
    ax_roc.legend()
    ax_pr.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(y_true, preds_by_model, title, path):
    """One confusion matrix per model. Rows = actual class, columns = predicted."""
    fig, axes = plt.subplots(1, len(preds_by_model), figsize=(5 * len(preds_by_model), 4))
    axes = np.atleast_1d(axes)
    for ax, (name, y_pred) in zip(axes, preds_by_model.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=["<=50K", ">50K"], ax=ax, colorbar=False
        )
        ax.set_title(name)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
