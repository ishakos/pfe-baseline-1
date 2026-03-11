import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("../data/iot_dataset_raw.csv")

print("Columns:", df.columns.tolist())  

"""
print(df.groupby("label")["duration"].describe())
print(df.groupby("label")["src_bytes"].describe())


total = len(df)
unique = len(df.drop_duplicates())
print("total rows:", total)
print("unique rows:", unique)
print("unique / total:", unique / total)

""
print(df.shape)
print(df["label"].value_counts())
print(df.duplicated().sum())

X = df.drop(columns=["label"])
y = df["label"]

# encode non-numeric columns before fitting
X = pd.get_dummies(X, drop_first=True)

model = RandomForestClassifier(random_state=0, n_estimators=50)
model.fit(X, y)

feat_imp = pd.Series(model.feature_importances_, index=X.columns)
print(feat_imp.sort_values(ascending=False).head(10))
"""