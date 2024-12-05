import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from typing import Tuple


class KMeansClustering:
    """
    A custom implementation of the K-Means Clustering algorithm.
    """

    def __init__(self, n_clusters: int, max_iter: int = 300, tol: float = 1e-4):
        """
        Initializes the K-Means clustering model with user-defined parameters.

        Args:
            n_clusters (int): The number of clusters to form.
            max_iter (int): The maximum number of iterations for convergence.
            tol (float): The tolerance level to check for convergence based on centroid movement.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids: np.ndarray = None
        self.labels: np.ndarray = None

    def fit(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the K-Means clustering algorithm on the given data.

        Args:
            data (np.ndarray): The input data matrix where rows represent samples.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The cluster labels and the final centroids.
        """
        np.random.seed(42)
        initial_indices = np.random.choice(data.shape[0], self.n_clusters, replace=False)
        self.centroids = data[initial_indices]

        for _ in range(self.max_iter):
            distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            updated_centroids = np.empty_like(self.centroids)
            for cluster_idx in range(self.n_clusters):
                cluster_points = data[self.labels == cluster_idx]
                if cluster_points.size == 0:
                    updated_centroids[cluster_idx] = data[np.random.choice(data.shape[0])]
                else:
                    updated_centroids[cluster_idx] = cluster_points.mean(axis=0)

            if np.linalg.norm(updated_centroids - self.centroids) < self.tol:
                break

            self.centroids = updated_centroids

        return self.labels, self.centroids


if __name__ == "__main__":
    # Load preprocessed data
    input_csv = "ClusterAlgorithmComparison/backend/sp500_preprocessed_data.csv"
    print(f"Loading data from {input_csv}...")
    data = pd.read_csv(input_csv, index_col=0)

    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data.values)

    # Apply PCA
    print("Performing PCA for dimensionality reduction...")
    pca = PCA(n_components=5)
    pca_data = pca.fit_transform(standardized_data)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    # Apply K-Means clustering
    n_clusters = 2
    print(f"Fitting K-Means with {n_clusters} clusters...")
    kmeans = KMeansClustering(n_clusters=n_clusters)
    labels, centroids = kmeans.fit(pca_data)

    # Save results
    output_csv = "ClusterAlgorithmComparison/backend/sp500_kmeans_clusters.csv"
    data["KMeans_Cluster"] = labels
    print(f"Saving cluster results to {output_csv}...")
    data.to_csv(output_csv, index=True)

    # Visualize using t-SNE
    print("Applying t-SNE for visualization...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=42)
    tsne_data = tsne.fit_transform(pca_data)

    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels, cmap="viridis", s=10)
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="x", s=100, label="Centroids")
    plt.title("K-Means Clustering Results")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.show()
