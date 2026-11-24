"""Classify breast cancer tumors with a scaled support vector machine."""

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Support Vector Machine: Breast Cancer Wisconsin dataset")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, target_names=dataset.target_names))


if __name__ == "__main__":
    main()
