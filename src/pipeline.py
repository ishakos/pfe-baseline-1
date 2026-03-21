import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_STATE


def build_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", build_one_hot_encoder()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_cols, categorical_cols


def build_model_configs(X):
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)

    models = {
        "dummy": {
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]),
            "params": None,
        },
        "logistic_regression": {
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )),
            ]),
            "params": {
                "model__C": np.logspace(-2, 2, 10),
                "model__solver": ["lbfgs"],
            },
        },
        "decision_tree": {
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("model", DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )),
            ]),
            "params": {
                "model__max_depth": [5, 10, 15, 20, 30, None],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__criterion": ["gini", "entropy"],
            },
        },
        "random_forest": {
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced"
                )),
            ]),
            "params": {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [10, 20, 30, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
        },
    }

    return models, numeric_cols, categorical_cols