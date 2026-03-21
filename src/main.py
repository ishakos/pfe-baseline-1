import json
import pandas as pd

from train import (
    load_data,
    prepare_features,
    split_data,
    train_and_select_best_model,
    retrain_best_model,
)
from evaluate import evaluate_model
from diagnose import run_diagnostics
from config import REPORTS_DIR


def main():
    print("Loading data...")
    df = load_data()

    print("Preparing features...")
    X, y = prepare_features(df)

    print("Splitting data into train / validation / test...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("Training and selecting best baseline model...")
    best_name, best_model, comparison_summary = train_and_select_best_model(X_train, y_train, X_val, y_val)

    print("Retraining best model on train + validation...")
    final_model = retrain_best_model(best_name, X_train, y_train, X_val, y_val)

    print("Evaluating final model on untouched test set...")
    test_metrics = evaluate_model(final_model, X_test, y_test, "final_test")

    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    print("Running diagnostics...")
    diagnostics_status = "success"
    diagnostics_error = None

    try:
        run_diagnostics(final_model, X_train_full, y_train_full, X_test, y_test)
    except Exception as e:
        diagnostics_status = "failed"
        diagnostics_error = str(e)
        print(f"Diagnostics failed: {e}")

    with open(REPORTS_DIR / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_model_name": best_name,
            "final_test_metrics": test_metrics,
            "diagnostics_status": diagnostics_status,
            "diagnostics_error": diagnostics_error,
        }, f, indent=4)

    print("\nDone.")
    print(f"Best model: {best_name}")
    print("Final test metrics:")
    for k, v in test_metrics.items():
        if isinstance(v, (int, float)) or v is None:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()