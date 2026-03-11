import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


def load_clean_data(path):
    df = pd.read_csv(path)

    """
    print(df["label"].value_counts())
    print(df["label"].unique())
    print(df.shape)
    """

    return df


def split_features_target(df):

    y = df["label"]

    X = df.drop(columns=[
    "label",
    "type",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    'src_ip_bytes', 
    'dst_ip_bytes', 
    'src_bytes', 
    'dst_bytes',
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

    """
    print("Train distribution:", np.bincount(y_train))
    print("Test distribution:", np.bincount(y_test))

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    selector = VarianceThreshold(threshold=0.01)

    X = selector.fit_transform(X)

    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    """

    model.fit(X_train, y_train)

    """
    feat_importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns if hasattr(X_train, 'columns') else range(len(model.feature_importances_))
    ).sort_values(ascending=False)

    print(feat_importance.head(15))
    """

    joblib.dump(model, "../model/random_forest.pkl")

    return model, X_test, y_test

    

