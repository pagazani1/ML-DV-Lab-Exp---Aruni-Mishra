"""Reduce the Iris dataset to two principal components."""

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    dataset = load_iris()
    scaled = StandardScaler().fit_transform(dataset.data)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled)

    print("PCA: Iris dataset")
    print(f"Original shape: {dataset.data.shape}")
    print(f"Reduced shape: {reduced.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.round(3)}")
    print(f"Total variance retained: {pca.explained_variance_ratio_.sum():.3f}")


if __name__ == "__main__":
    main()
