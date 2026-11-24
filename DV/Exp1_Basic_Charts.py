"""Create a scatter plot, histogram, and box plot from sample measurements."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    data = pd.DataFrame(
        {
            "study_hours": [1, 2, 2.5, 3, 4, 4.5, 5, 6, 6.5, 7, 8, 9],
            "score": [52, 58, 61, 63, 69, 72, 74, 79, 82, 85, 90, 94],
            "group": ["A"] * 6 + ["B"] * 6,
        }
    )
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sns.scatterplot(data=data, x="study_hours", y="score", hue="group", ax=axes[0])
    axes[0].set_title("Study time and score")
    sns.histplot(data=data, x="score", bins=6, kde=True, ax=axes[1])
    axes[1].set_title("Score distribution")
    sns.boxplot(data=data, x="group", y="score", ax=axes[2])
    axes[2].set_title("Scores by group")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
