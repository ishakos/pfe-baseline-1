from train import load_clean_data
from train import split_features_target
from train import train_model
from evaluate import evaluate_model
from pipeline import build_model
from diagnose import diagnose_model



DATA_PATH = "../../Data/iot_dataset_clean.csv"


def main():

    df = load_clean_data(DATA_PATH)

    X, y = split_features_target(df)

    model = build_model()

    model, X_test, y_test = train_model(model, X, y)

    evaluate_model(model, X_test, y_test)

    #diagnose_model("../model/random_forest.pkl", X_test, y_test, X_train=X_train, y_train=y_train")

if __name__ == "__main__":
    main()