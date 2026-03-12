from train import load_clean_data
from train import split_features_target
from train import train_model
from pipeline import build_model
from evaluate import evaluate_model


DATA_PATH = "../../Data/iot_dataset_clean.csv"


def main():

    df = load_clean_data(DATA_PATH)

    X, y = split_features_target(df)

    model = build_model()

    model, X_test, y_test = train_model(model, X, y)

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()