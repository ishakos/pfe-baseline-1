import matplotlib
matplotlib.use("Agg")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve
from sklearn.calibration import calibration_curve
from sklearn.metrics import f1_score

from config import REPORTS_DIR, RANDOM_STATE


def save_json(data, filename):
    with open(REPORTS_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def plot_learning_curve(model, X, y, filename="learning_curve.png"):
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=1,   # safer on Windows with plotting workflows
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_sizes, train_mean, marker="o", label="Train F1")
    ax.plot(train_sizes, val_mean, marker="o", label="Validation F1")
    ax.set_title("Learning Curve")
    ax.set_xlabel("Training Size")
    ax.set_ylabel("F1 Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def permutation_importance_report(model, X_test, y_test, top_n=15):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,   # safer
    )

    importances = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)
    top_features = importances.head(top_n)

    top_features.to_csv(REPORTS_DIR / "permutation_importance.csv", header=["importance"])

    fig, ax = plt.subplots(figsize=(8, 5))
    top_features.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top Permutation Importances")
    ax.set_xlabel("Mean Importance")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "permutation_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return top_features.to_dict()


def feature_ablation_test(model, X_test, y_test, top_features):
    baseline_f1 = f1_score(y_test, model.predict(X_test))
    results = {"baseline_f1": float(baseline_f1), "ablation_f1": {}}

    for feature in top_features:
        if feature not in X_test.columns:
            continue

        temp = X_test.copy()

        if pd.api.types.is_numeric_dtype(temp[feature]):
            temp[feature] = temp[feature].median()
        else:
            mode_value = temp[feature].mode(dropna=True)
            temp[feature] = mode_value.iloc[0] if not mode_value.empty else "missing"

        f1 = f1_score(y_test, model.predict(temp))
        results["ablation_f1"][feature] = float(f1)

    save_json(results, "feature_ablation.json")
    return results


def robustness_test(model, X_test, y_test):
    X_noise = X_test.copy()
    numeric_cols = X_noise.select_dtypes(include=[np.number]).columns

    rng = np.random.default_rng(RANDOM_STATE)

    for col in numeric_cols:
        noise = rng.normal(0, 0.10, size=len(X_noise))
        X_noise[col] = X_noise[col] * (1 + noise)

    noisy_f1 = f1_score(y_test, model.predict(X_noise))

    X_missing = X_test.copy()
    for col in numeric_cols:
        idx = rng.choice(len(X_missing), size=max(1, int(0.05 * len(X_missing))), replace=False)
        X_missing.loc[X_missing.index[idx], col] = np.nan

    try:
        missing_f1 = f1_score(y_test, model.predict(X_missing))
    except Exception:
        missing_f1 = None

    results = {
        "noise_f1": float(noisy_f1),
        "missing_f1": None if missing_f1 is None else float(missing_f1),
    }

    save_json(results, "robustness_report.json")
    return results


def calibration_report(model, X_test, y_test):
    if not hasattr(model, "predict_proba"):
        return None

    y_prob = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "calibration_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    results = {
        "mean_predicted_value": mean_pred.tolist(),
        "fraction_of_positives": frac_pos.tolist(),
    }

    save_json(results, "calibration_report.json")
    return results


def run_diagnostics(model, X_train_full, y_train_full, X_test, y_test):
    plot_learning_curve(model, X_train_full, y_train_full)
    top_features = permutation_importance_report(model, X_test, y_test, top_n=10)
    feature_ablation_test(model, X_test, y_test, list(top_features.keys())[:5])
    robustness_test(model, X_test, y_test)
    calibration_report(model, X_test, y_test)