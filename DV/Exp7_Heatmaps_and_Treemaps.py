"""Build a correlation heatmap and a proportional-rectangle treemap."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


def main():
    sales = pd.DataFrame(
        {
            "category": ["Books", "Games", "Music", "Office", "Outdoor", "Toys"],
            "revenue": [240, 180, 120, 90, 210, 160],
            "units": [80, 45, 60, 30, 42, 64],
            "returns": [4, 8, 3, 2, 5, 6],
        }
    )
    sns.set_theme(style="white")
    fig, (heat_ax, tree_ax) = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(
        sales[["revenue", "units", "returns"]].corr(),
        annot=True,
        cmap="vlag",
        center=0,
        square=True,
        ax=heat_ax,
    )
    heat_ax.set_title("Sales metric correlations")

    palette = sns.color_palette("Set3", n_colors=len(sales))
    total = sales["revenue"].sum()
    left = 0.0
    for row, color in zip(sales.itertuples(), palette):
        width = row.revenue / total
        tree_ax.add_patch(
            Rectangle((left, 0), width, 1, facecolor=color, edgecolor="white", linewidth=2)
        )
        tree_ax.text(
            left + width / 2,
            0.5,
            f"{row.category}\n{row.revenue / total:.0%}",
            ha="center",
            va="center",
            fontsize=9,
        )
        left += width
    tree_ax.set(xlim=(0, 1), ylim=(0, 1), title="Revenue share treemap", xticks=[], yticks=[])
    for spine in tree_ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
