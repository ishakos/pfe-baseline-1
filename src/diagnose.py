# python diagnose.py --model model.pkl --data test_data.csv

# python diagnose.py --model ../model/random_forest.pkl --data ../../Data/iot_dataset_clean.csv

import argparse
import numpy as np
import pandas as pd
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json
import warnings
warnings.filterwarnings("ignore")
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss, brier_score_loss,
    precision_score, recall_score, confusion_matrix
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from scipy.stats import ks_2samp
from train import split_features_target
from train import get_train_test_data
from collections import Counter

# Support XGBoost / LightGBM / Keras
try:
    import xgboost as xgb
except:
    pass

try:
    import lightgbm as lgb
except:
    pass

# ===============================
# HELPER FUNCTIONS
# ===============================

def load_model(model_path):
    """Load model from pickle or joblib"""
    ext = os.path.splitext(model_path)[1]
    if ext in [".pkl", ".joblib"]:
        return joblib.load(model_path)
    else:
        raise ValueError("Unsupported model format. Use .pkl, .joblib, or .h5")

def is_tree_model(model):
    """Check if model has feature_importances_"""
    return hasattr(model, "feature_importances_") or \
           ("booster" in dir(model) and hasattr(model.booster(), "feature_importances_"))

def predict_proba_safe(model, X):
    """Return probabilities for binary classification"""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:,1]
    elif hasattr(model, "predict"):
        preds = model.predict(X)
        if preds.ndim > 1 and preds.shape[1] == 2:
            return preds[:,1]
        else:
            return preds
    else:
        raise ValueError("Model has no predict or predict_proba method")

# ===============================
# TESTS
# ===============================

def baseline_sanity(model, X_test, y_test, X_train=None, y_train=None):
    """Dummy classifier & label shuffle test"""
    report = {}
    # Dummy
    dummy = DummyClassifier(strategy="most_frequent")
    if X_train is not None and y_train is not None:
        dummy.fit(X_train, y_train)
    else:
        dummy.fit(X_test, y_test)
    dummy_pred = dummy.predict(X_test)
    dummy_f1 = f1_score(y_test, dummy_pred)
    report["dummy_f1"] = dummy_f1

    # Model on test
    y_pred = model.predict(X_test)
    model_f1 = f1_score(y_test, y_pred)
    report["model_f1"] = model_f1

    # Train/Test gap
    gap = None
    if X_train is not None and y_train is not None:
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, y_pred)
        gap = train_acc - test_acc
        report["train_acc"] = train_acc
        report["test_acc"] = test_acc
        report["train_test_gap"] = gap

    return report

def cross_validation_test(model, X, y, cv=5):
    """CV F1 / Accuracy / AUC"""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    f1s = cross_val_score(model, X, y, cv=skf, scoring="f1")
    accs = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    aucs = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
    return {
        "f1_mean": f1s.mean(),
        "f1_std": f1s.std(),
        "acc_mean": accs.mean(),
        "acc_std": accs.std(),
        "auc_mean": aucs.mean(),
        "auc_std": aucs.std()
    }

def feature_importance_tests(model, X_test, y_test, top_n=5):
    """
    Top feature ablation + permutation importance (safe version)

    - Ablation: replaces top features with 0 (or mean)
    - Randomization: permutes top features
    """
    results = {}

    # --------------------------
    # Permutation / Built-in Importance
    # --------------------------
    if is_tree_model(model):
        importances = pd.Series(model.feature_importances_, index=X_test.columns)
    else:
        perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
        importances = pd.Series(perm.importances_mean, index=X_test.columns)
    
    importances = importances.sort_values(ascending=False)
    results["top_features"] = importances.head(top_n).to_dict()

    # --------------------------
    # Ablation Test (zero-out instead of drop)
    # --------------------------
    ablation = {}
    for f in importances.head(top_n).index:
        X_mod = X_test.copy()
        # Zero-out or replace with mean
        X_mod[f] = 0  # you can also do X_mod[f] = X_mod[f].mean()
        y_pred = model.predict(X_mod)
        ablation[f] = f1_score(y_test, y_pred)
    results["ablation_f1"] = ablation

    # --------------------------
    # Randomization Test
    # --------------------------
    X_rand = X_test.copy()
    for f in importances.head(top_n).index:
        X_rand[f] = np.random.permutation(X_rand[f].values)
    y_pred_rand = model.predict(X_rand)
    results["randomized_f1"] = f1_score(y_test, y_pred_rand)

    return results

def robustness_tests(model, X_test, y_test):
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    """Noise, outlier, missing simulation"""
    results = {}
    numeric = X_test.select_dtypes(include=np.number).columns
    # Gaussian noise
    X_noise = X_test.copy()
    for col in numeric:
        X_noise[col] = X_noise[col] * (1 + np.random.normal(0, 0.1, len(X_noise)))
    results["noise_f1"] = f1_score(y_test, model.predict(X_noise))

    # Outliers
    X_out = X_test.copy()
    for col in numeric:
        idx = np.random.choice(len(X_out), int(0.05*len(X_out)), replace=False)
        X_out.loc[idx, col] = X_out[col].max()*10
    results["outlier_f1"] = f1_score(y_test, model.predict(X_out))

    # Missing values
    X_nan = X_test.copy()
    for col in numeric:
        idx = np.random.choice(len(X_nan), int(0.05*len(X_nan)), replace=False)
        X_nan.loc[idx, col] = np.nan
    try:
        results["nan_f1"] = f1_score(y_test, model.predict(X_nan.fillna(X_test.mean())))
    except:
        results["nan_f1"] = None
    return results

def distribution_shift_tests(X_train, X_test):
    """KS test per feature, mean/std drift"""
    drift = {}
    for col in X_train.columns:
        if np.issubdtype(X_train[col].dtype, np.number):
            ks_stat, ks_p = ks_2samp(X_train[col], X_test[col])
            drift[col] = ks_stat
    return drift

def data_quality_checks(X_train, X_test, y_train, y_test):
    """Missing, duplicates, imbalance, correlation"""
    results = {}

    y_train = np.bincount(y_train)
    y_test = np.bincount(y_test)

    # Class imbalance
    train_dist = Counter(y_train)
    test_dist = Counter(y_test)

    print("\nClass distribution:")

    print("Train:", train_dist)
    print("Test:", test_dist)

    # Missing / duplicate
    results["missing_values"] = X_test.isnull().sum().sum()
    results["duplicate_rows"] = X_test.duplicated().sum()

    # Correlation
    corr = X_train.corr()
    high_corr = np.sum(np.abs(corr.values[np.triu_indices_from(corr.values,1)])>0.95)
    results["high_correlation_features"] = high_corr
    return results

def model_calibration(model, X_test, y_test, bins=10):
    """Calibration curve, Brier score"""
    prob = predict_proba_safe(model, X_test)
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob, n_bins=bins)
    brier = brier_score_loss(y_test, prob)
    return {
        "brier_score": brier,
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted_value": mean_predicted_value.tolist()
    }

def make_json_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    return obj

def detect_high_correlation(X, threshold=0.9):

    corr_matrix = X.corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    high_corr = [
        column for column in upper.columns
        if any(upper[column] > threshold)
    ]

    print("\nHighly correlated features:", high_corr)

    return high_corr

def correlation_heatmap(X):

    plt.figure(figsize=(12,8))

    sns.heatmap(
        X.corr(),
        cmap="coolwarm",
        center=0
    )

    plt.title("Feature Correlation Heatmap")

    plt.show()

def plot_learning_curve(model, X, y):

    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        train_sizes=np.linspace(0.1,1.0,5)
    )

    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    plt.plot(train_sizes, train_mean, label="Training Score")
    plt.plot(train_sizes, test_mean, label="Validation Score")

    plt.xlabel("Training Size")
    plt.ylabel("F1 Score")
    plt.title("Learning Curve")

    plt.legend()
    plt.show()



def shap_analysis(model, X_sample):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample)
# ===============================
# MAIN FUNCTION
# ===============================

def diagnose_model(model_path, X_test, y_test, X_train=None, y_train=None, X=None, y=None):
    model = load_model(model_path)
    report = {}

    # Baseline
    report["baseline"] = baseline_sanity(model, X_test, y_test, X_train, y_train)
    # CV
    if X_train is not None and y_train is not None:
        report["cross_validation"] = cross_validation_test(model, X_train, y_train)
    # Feature importance
    report["feature_importance"] = feature_importance_tests(model, X_test, y_test)
    # Robustness
    report["robustness"] = robustness_tests(model, X_test, y_test)
    # Distribution drift
    if X_train is not None:
        report["distribution_drift"] = distribution_shift_tests(X_train, X_test)
    # Data quality
    if X_train is not None:
        report["data_quality"] = data_quality_checks(X_train, X_test, y_train, y_test)
    # Calibration
    report["calibration"] = model_calibration(model, X_test, y_test)
    # Correlation
    report["correlation"] = detect_high_correlation(X)
    # Heatmap
    report["correlation_heatmap"] = correlation_heatmap(X_train)
    # Learning curve
    report["learning_curve"] = plot_learning_curve(model, X, y)
    # SHAP analysis
    X_sample = X_train.sample(1000)
    report["shap_analysis"] = shap_analysis(model, X_sample)

    # Print summary
    print("\nðŸš¨ MODEL HEALTH REPORT ðŸš¨")
    print("="*50)
    print(json.dumps(make_json_serializable(report), indent=4))

    return report

# ===============================
# COMMAND-LINE INTERFACE
# ===============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to saved model (.pkl/.joblib/.h5)")
    parser.add_argument("--data", type=str, required=True, help="Path to test CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if "label" not in df.columns:
        raise ValueError("Test CSV must contain 'label' column")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = get_train_test_data(X, y)
    diagnose_model(args.model, X_test, y_test, X_train, y_train, X, y)