import matplotlib
matplotlib.use("Agg")

import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

from config import REPORTS_DIR


def get_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        return scores
    return None


def compute_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }

    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def save_metrics(metrics, filename):
    path = REPORTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def plot_confusion_matrix(cm, title, filename):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.tight_layout()
    fig.savefig(REPORTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_prob, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(y_true, y_prob, title, filename):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision)
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(model, X, y, split_name):
    y_pred = model.predict(X)
    y_prob = get_probabilities(model, X)

    metrics = compute_metrics(y, y_pred, y_prob)

    save_metrics(metrics, f"{split_name}_metrics.json")
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        f"{split_name.capitalize()} Confusion Matrix",
        f"{split_name}_confusion_matrix.png"
    )

    if y_prob is not None:
        plot_roc_curve(y, y_prob, f"{split_name.capitalize()} ROC Curve", f"{split_name}_roc_curve.png")
        plot_pr_curve(y, y_prob, f"{split_name.capitalize()} PR Curve", f"{split_name}_pr_curve.png")

    return metrics