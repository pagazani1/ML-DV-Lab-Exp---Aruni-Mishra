"""Forecast six months of sample sales using a seasonal linear model."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def make_features(periods):
    date_index = pd.DatetimeIndex(periods)
    month_number = (date_index.year - 2023) * 12 + date_index.month - 1
    return pd.DataFrame(
        {
            "trend": month_number,
            "sin_year": np.sin(2 * np.pi * date_index.month / 12),
            "cos_year": np.cos(2 * np.pi * date_index.month / 12),
        }
    )


def main():
    dates = pd.date_range("2023-01-01", periods=36, freq="MS")
    month_number = np.arange(len(dates))
    sales = 120 + 2.4 * month_number + 18 * np.sin(2 * np.pi * dates.month / 12)
    history = pd.DataFrame({"date": dates, "sales": sales.round(1)})

    split = len(history) - 6
    train_dates = history["date"].iloc[:split]
    test_dates = history["date"].iloc[split:]
    validation_model = LinearRegression().fit(
        make_features(train_dates), history["sales"].iloc[:split]
    )
    validation_predictions = validation_model.predict(make_features(test_dates))
    print(f"Last-six-month validation MAE: {mean_absolute_error(history['sales'].iloc[split:], validation_predictions):.2f}")

    model = LinearRegression().fit(make_features(history["date"]), history["sales"])
    future_dates = pd.date_range(dates[-1] + pd.offsets.MonthBegin(1), periods=6, freq="MS")
    forecast = pd.DataFrame(
        {"date": future_dates, "sales": model.predict(make_features(future_dates)).round(1)}
    )
    print("\nSix-month forecast:")
    print(forecast.to_string(index=False))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=history, x="date", y="sales", label="Observed", ax=ax)
    sns.lineplot(data=forecast, x="date", y="sales", label="Forecast", marker="o", ax=ax)
    ax.axvline(dates[-1], color="gray", linestyle="--", linewidth=1)
    ax.set(title="Monthly sales and six-month forecast", xlabel="Month", ylabel="Sales")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
