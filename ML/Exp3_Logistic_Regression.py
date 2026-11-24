"""Classify breast cancer tumors with logistic regression."""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000, random_state=42)
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Logistic Regression: Breast Cancer Wisconsin dataset")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions, target_names=dataset.target_names))


if __name__ == "__main__":
    main()
