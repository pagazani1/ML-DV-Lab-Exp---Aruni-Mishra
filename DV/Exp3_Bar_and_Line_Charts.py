"""Visualize monthly sales by product with bar and line charts."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    data = pd.DataFrame(
        {
            "month": pd.to_datetime(
                ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-05-01"]
                * 2
            ),
            "product": ["Orchid"] * 5 + ["Cedar"] * 5,
            "sales": [28, 34, 31, 42, 48, 22, 27, 35, 33, 40],
        }
    )
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    sns.barplot(data=data, x="month", y="sales", hue="product", ax=axes[0])
    axes[0].set(title="Monthly sales by product", xlabel="Month", ylabel="Sales")
    axes[0].tick_params(axis="x", rotation=30)
    sns.lineplot(data=data, x="month", y="sales", hue="product", marker="o", ax=axes[1])
    axes[1].set(title="Sales trend", xlabel="Month", ylabel="Sales")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
