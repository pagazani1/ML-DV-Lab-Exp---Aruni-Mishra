"""Compare category shares with pie and donut charts."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    expenses = pd.DataFrame(
        {"category": ["Rent", "Food", "Transport", "Utilities", "Savings"], "amount": [900, 420, 180, 160, 340]}
    )
    colors = sns.color_palette("Set2", n_colors=len(expenses))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].pie(
        expenses["amount"],
        labels=expenses["category"],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
    )
    axes[0].set_title("Monthly budget: pie chart")
    axes[1].pie(
        expenses["amount"],
        labels=expenses["category"],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
    )
    axes[1].text(0, 0, f"Total\n${expenses['amount'].sum():,}", ha="center", va="center")
    axes[1].set_title("Monthly budget: donut chart")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
