import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score


def load_clean_data(path):
    df = pd.read_csv(path)
    return df


def split_features_target(df):

    y = df["label"]

    X = df.drop(columns=[
    "src_ip",
    'src_bytes', 
    'dst_bytes',
    'src_ip_bytes', 
    'dst_ip_bytes', 
    'http_version',
    'http_method',
    'ssl_resumed',
    "label",
    "type",
    ], errors="ignore")

    return X, y


def train_model(model, X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Cross validation (training set only)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    print("Cross Validation F1 Scores:", cv_scores)
    print("Mean CV F1:", cv_scores.mean())

    # ---- FINAL TRAINING ----
    model.fit(X_train, y_train)

    feat_importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    print(feat_importance)

    joblib.dump(model, "../model/random_forest.pkl")

    return model, X_test, y_test

def get_train_test_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test