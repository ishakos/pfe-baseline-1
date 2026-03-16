import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib

# ----------------------------
# Load your dataset
# ----------------------------
df = pd.read_csv("../../Data/iot_dataset_clean.csv")

# Drop unnecessary columns (your previous preprocessing)
cols_to_drop = ["dst_ip", "src_port", "dst_port", "conn_state", "service",
                "dns_query", "dns_AA", "dns_RD", "dns_RA", "dns_rcode",
                "ssl_subject", "ssl_issuer", "ssl_established",
                "http_uri", "http_user_agent", "http_orig_mime_types",
                "http_resp_mime_types", "http_status_code", "weird_addl",
                "weird_name", "weird_notice"]

df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

# Features and target
target_col = "label"
X = df.drop(columns=[target_col])
y = df[target_col]

# Create groups to prevent leakage (identical feature rows)
groups = pd.util.hash_pandas_object(X, index=False)

# ----------------------------
# Group-aware train-test split
# ----------------------------
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

# ----------------------------
# Train baseline model
# ----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

baseline_f1 = f1_score(y_test, model.predict(X_test))
print(f"Baseline F1 (all features): {baseline_f1:.4f}")

# ----------------------------
# Feature ablation
# ----------------------------
feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
top_features = feature_importances.sort_values(ascending=False).index.tolist()

ablation_results = []

for feature in top_features:
    print(f"\n--- Ablating feature: {feature} ---")
    
    X_train_ablate = X_train.drop(columns=[feature])
    X_test_ablate = X_test.drop(columns=[feature])
    
    model_ablate = RandomForestClassifier(n_estimators=100, random_state=42)
    model_ablate.fit(X_train_ablate, y_train)
    
    f1 = f1_score(y_test, model_ablate.predict(X_test_ablate))
    ablation_results.append((feature, f1))
    
    print(f"F1 without {feature}: {f1:.4f}")

# ----------------------------
# Results table
# ----------------------------
ablation_df = pd.DataFrame(ablation_results, columns=["Feature", "F1_without"])
ablation_df["F1_drop"] = baseline_f1 - ablation_df["F1_without"]
ablation_df = ablation_df.sort_values("F1_drop", ascending=False)
print("\n--- Feature Ablation Summary ---")
print(ablation_df)

# Save results for thesis
ablation_df.to_csv("../results/feature_ablation_results.csv", index=False)

# Save final trained model
joblib.dump(model, "../model/random_forest2.pkl")