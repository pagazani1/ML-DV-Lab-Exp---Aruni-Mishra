"""Plot a built-in sample of city locations without external map services."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    cities = pd.DataFrame(
        {
            "city": ["Seattle", "San Francisco", "Denver", "Chicago", "Atlanta", "New York"],
            "longitude": [-122.33, -122.42, -104.99, -87.63, -84.39, -74.01],
            "latitude": [47.61, 37.77, 39.74, 41.88, 33.75, 40.71],
            "population_millions": [0.75, 0.81, 0.72, 2.67, 0.50, 8.34],
            "region": ["West", "West", "Central", "Central", "East", "East"],
        }
    )
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=cities,
        x="longitude",
        y="latitude",
        size="population_millions",
        hue="region",
        sizes=(80, 500),
        alpha=0.8,
        ax=ax,
    )
    for row in cities.itertuples():
        ax.annotate(row.city, (row.longitude, row.latitude), xytext=(5, 5), textcoords="offset points")
    ax.set(
        title="Sample US city locations (bubble size = population)",
        xlabel="Longitude",
        ylabel="Latitude",
        xlim=(-127, -67),
        ylim=(25, 50),
    )
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
