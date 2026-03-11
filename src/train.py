import pandas as pd
import numpy as np
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


def load_clean_data(path):
    df = pd.read_csv(path)
    return df


def split_features_target(df):

    y = df["label"]

    X = df.drop(columns=[
    'src_bytes', 
    'dst_bytes',
    'src_ip_bytes', 
    'dst_ip_bytes', 
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
    y_train = np.random.permutation(y_train)

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

    

