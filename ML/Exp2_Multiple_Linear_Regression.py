"""Fit multiple linear regression to all diabetes dataset features."""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    dataset = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Multiple Linear Regression: Diabetes dataset")
    for name, coefficient in zip(dataset.feature_names, model.coef_):
        print(f"{name:>4}: {coefficient:8.2f}")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.2f}")
    print(f"R²: {r2_score(y_test, predictions):.3f}")


if __name__ == "__main__":
    main()
