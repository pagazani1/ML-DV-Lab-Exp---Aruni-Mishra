"""Filter a sample sales dashboard from the command line.

Examples:
    python Exp4_Interactive_Dashboard_Filters.py --region North
    python Exp4_Interactive_Dashboard_Filters.py --product Tea --region South
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=["all", "North", "South", "West"], default="all")
    parser.add_argument("--product", choices=["all", "Tea", "Coffee", "Juice"], default="all")
    return parser


def main():
    data = pd.DataFrame(
        {
            "region": ["North", "North", "South", "South", "West", "West"] * 2,
            "product": ["Tea", "Coffee", "Tea", "Juice", "Coffee", "Juice"] * 2,
            "month": ["Jan", "Jan", "Jan", "Jan", "Jan", "Jan", "Feb", "Feb", "Feb", "Feb", "Feb", "Feb"],
            "sales": [32, 45, 28, 36, 41, 30, 38, 51, 35, 40, 47, 34],
        }
    )
    args = build_parser().parse_args()
    filtered = data.copy()
    if args.region != "all":
        filtered = filtered[filtered["region"] == args.region]
    if args.product != "all":
        filtered = filtered[filtered["product"] == args.product]
    if filtered.empty:
        raise ValueError("The selected filters returned no records.")

    print(f"Filters: region={args.region}, product={args.product}")
    print(f"Total sales: {filtered['sales'].sum()}")
    print(filtered.to_string(index=False))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=filtered, x="month", y="sales", hue="product", ax=ax)
    ax.set(title="Filtered sales dashboard", xlabel="Month", ylabel="Sales")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
