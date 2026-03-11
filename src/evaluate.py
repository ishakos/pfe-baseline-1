from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    try:
        probs = model.predict_proba(X_test)[:,1]
        print("\nROC AUC:", roc_auc_score(y_test, probs))
    except:
        pass