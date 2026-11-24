"""Cluster the Iris dataset with K-means and report internal metrics."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def main():
    dataset = load_iris()
    X = StandardScaler().fit_transform(dataset.data)
    model = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = model.fit_predict(X)

    print("K-means Clustering: Iris dataset")
    print(f"Cluster sizes: {np.bincount(labels)}")
    print(f"Silhouette score: {silhouette_score(X, labels):.3f}")
    print(
        "Adjusted Rand index (reference labels only): "
        f"{adjusted_rand_score(dataset.target, labels):.3f}"
    )


if __name__ == "__main__":
    main()
