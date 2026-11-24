"""Compare ordinary, Ridge, and Lasso regression on diabetes data."""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    dataset = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=42
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge (alpha=1.0)": Ridge(alpha=1.0),
        "Lasso (alpha=0.1)": Lasso(alpha=0.1, max_iter=10000),
    }

    print("Regression with Regularization: Diabetes dataset")
    for name, estimator in models.items():
        model = make_pipeline(StandardScaler(), estimator)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"{name:22} RMSE: {rmse:6.2f}  R²: {r2_score(y_test, predictions):.3f}")


if __name__ == "__main__":
    main()
