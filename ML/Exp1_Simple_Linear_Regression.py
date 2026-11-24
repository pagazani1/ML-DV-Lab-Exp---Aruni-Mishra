"""Fit simple linear regression to the diabetes dataset's BMI feature."""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    dataset = load_diabetes()
    bmi_index = dataset.feature_names.index("bmi")
    X = dataset.data[:, [bmi_index]]
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Simple Linear Regression: Diabetes dataset (BMI)")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"BMI coefficient: {model.coef_[0]:.2f}")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.2f}")
    print(f"R²: {r2_score(y_test, predictions):.3f}")


if __name__ == "__main__":
    main()
