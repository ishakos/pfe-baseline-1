import json
import joblib
import pandas as pd
from itertools import product

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV

from config import (
    DATA_PATH,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
    CV_FOLDS,
    N_ITER_SEARCH,
    PRIMARY_SCORING,
    MODEL_DIR,
    REPORTS_DIR,
    DROP_COLUMNS,
    OPTIONAL_DROP_COLUMNS,
)
from pipeline import build_model_configs
from evaluate import evaluate_model


CSV_DTYPE_MAP = {
    "src_ip": "string",
    "proto": "string",
    "ssl_version": "string",
    "ssl_cipher": "string",
    "ssl_resumed": "string",
    "http_method": "string",
    "http_version": "string",
    "type": "string",
}


def load_data(path=DATA_PATH):
    df = pd.read_csv(
        path,
        low_memory=False,
        dtype=CSV_DTYPE_MAP,
    )
    return df


def prepare_features(df):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    columns_to_drop = [c for c in DROP_COLUMNS + OPTIONAL_DROP_COLUMNS if c in df.columns]
    X = df.drop(columns=columns_to_drop, errors="ignore")
    y = df[TARGET_COL].copy()

    if TARGET_COL in X.columns:
        X = X.drop(columns=[TARGET_COL], errors="ignore")

    return X, y


def split_data(X, y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    relative_val_size = VAL_SIZE / (1.0 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    split_info = {
        "train_shape": X_train.shape,
        "val_shape": X_val.shape,
        "test_shape": X_test.shape,
        "train_positive_ratio": float(y_train.mean()),
        "val_positive_ratio": float(y_val.mean()),
        "test_positive_ratio": float(y_test.mean()),
    }

    with open(REPORTS_DIR / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=4)

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_total_param_combinations(param_grid):
    values = list(param_grid.values())
    total = 1
    for v in values:
        total *= len(v)
    return total


def fit_single_model(name, config, X_train, y_train):
    pipeline = config["pipeline"]
    params = config["params"]

    if params is None:
        pipeline.fit(X_train, y_train)
        return pipeline, None

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    total_combinations = get_total_param_combinations(params)
    effective_n_iter = min(N_ITER_SEARCH, total_combinations)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=params,
        n_iter=effective_n_iter,
        scoring=PRIMARY_SCORING,
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=1,
    )

    search.fit(X_train, y_train)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "n_iter_used": effective_n_iter,
        "total_param_combinations": total_combinations,
    }


def train_and_select_best_model(X_train, y_train, X_val, y_val):
    model_configs, numeric_cols, categorical_cols = build_model_configs(X_train)

    summary = {}
    fitted_models = {}

    for name, config in model_configs.items():
        print(f"\n========== Training: {name} ==========")
        estimator, search_info = fit_single_model(name, config, X_train, y_train)
        val_metrics = evaluate_model(estimator, X_val, y_val, f"val_{name}")

        summary[name] = {
            "search_info": search_info,
            "val_metrics": val_metrics,
        }
        fitted_models[name] = estimator

    with open(REPORTS_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    best_name = max(summary, key=lambda n: summary[n]["val_metrics"]["f1"])
    best_model = fitted_models[best_name]

    print(f"\nBest model selected on validation F1: {best_name}")

    with open(REPORTS_DIR / "best_model_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_model_name": best_name,
            "selection_metric": "validation_f1",
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
        }, f, indent=4)

    return best_name, best_model, summary


def retrain_best_model(best_name, X_train, y_train, X_val, y_val):
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    model_configs, _, _ = build_model_configs(X_train_full)
    best_config = model_configs[best_name]

    final_model, final_search_info = fit_single_model(best_name, best_config, X_train_full, y_train_full)

    joblib.dump(final_model, MODEL_DIR / f"{best_name}_final.joblib")

    with open(REPORTS_DIR / "final_model_training_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_model_name": best_name,
            "final_search_info": final_search_info,
            "train_full_shape": X_train_full.shape,
        }, f, indent=4)

    return final_model