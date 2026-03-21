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
print(list(df.columns))

