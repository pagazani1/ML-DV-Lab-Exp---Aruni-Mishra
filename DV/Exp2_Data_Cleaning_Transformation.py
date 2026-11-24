"""Clean a small sales table and demonstrate derived columns and summaries."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    data = pd.DataFrame(
        {
            "date": ["2025-01-03", "2025-01-10", "2025-01-10", "2025-02-02", "2025-02-14"],
            "category": ["Books", "Games", "Games", "Books", "Games"],
            "units": [2, 1, 1, None, 3],
            "unit_price": [15.0, 40.0, 40.0, 18.0, None],
        }
    )
    print("Raw data:")
    print(data.to_string(index=False))

    clean = data.drop_duplicates().copy()
    clean["date"] = pd.to_datetime(clean["date"])
    clean["units"] = clean["units"].fillna(clean["units"].median()).astype(int)
    clean["unit_price"] = clean["unit_price"].fillna(clean["unit_price"].median())
    clean["revenue"] = clean["units"] * clean["unit_price"]
    clean["month"] = clean["date"].dt.to_period("M").astype(str)

    summary = clean.groupby("category", as_index=False)["revenue"].sum()
    print("\nCleaned data:")
    print(clean.to_string(index=False))
    print("\nRevenue by category:")
    print(summary.to_string(index=False))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=summary, x="category", y="revenue", color="steelblue", ax=ax)
    ax.set(title="Revenue after cleaning", xlabel="Category", ylabel="Revenue")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
