"""Classify wine cultivars with Gaussian Naive Bayes."""

from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def main():
    dataset = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )
    model = GaussianNB().fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Gaussian Naive Bayes: Wine dataset")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions, target_names=dataset.target_names))


if __name__ == "__main__":
    main()
