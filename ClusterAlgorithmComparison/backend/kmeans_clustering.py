import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.preprocessing import StandardScaler  # For feature scaling


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
    
    def calculate_within_cluster_variances(self, data: np.ndarray) -> pd.DataFrame:
        """
        Calculate within-cluster variances for each cluster.

        Args:
            data (np.ndarray): Standardized data to evaluate.

        Returns:
            pd.DataFrame: Variance metrics for each cluster.
        """
        variances = {
            f"Cluster_{i}": data[self.labels == i].var(axis=0).mean()
            for i in range(self.n_clusters)
        }
        return pd.DataFrame.from_dict(variances, orient="index", columns=["Within-Cluster Variance"])


def find_optimal_clusters(data: np.ndarray, max_clusters: int = 10) -> None:
    """
    Uses the elbow method to identify the optimal number of clusters.

    Args:
        data (np.ndarray): The input data for clustering.
        max_clusters (int): The maximum number of clusters to test.
    """
    distortions = []
    for k in range(1, max_clusters + 1):
        model = KMeansClustering(n_clusters=k)
        labels, centroids = model.fit(data)
        distortion = sum(
            np.min(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
        )
        distortions.append(distortion)

    plt.figure()
    plt.plot(range(1, max_clusters + 1), distortions, marker="o")
    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Distortion")
    plt.show()

if __name__ == "__main__":
    # Load preprocessed data
    input_csv = "ClusterAlgorithmComparison/backend/sp500_preprocessed_data.csv"
    print(f"Loading data from {input_csv}...")
    data = pd.read_csv(input_csv, index_col=0)

    # Ensure features are numeric and standardized
    print("Standardizing data...")
    scaler = StandardScaler()
    feature_data = scaler.fit_transform(data)

    # Elbow method to determine optimal clusters
    print("Running elbow method to find optimal number of clusters...")
    find_optimal_clusters(feature_data)

    # Perform clustering with a selected number of clusters
    n_clusters = 4  # Adjust based on the elbow method results
    print(f"Applying K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeansClustering(n_clusters=n_clusters)
    labels, centroids = kmeans.fit(feature_data)

    # Save clustering results
    data["KMeans_Cluster"] = labels
    output_csv = "ClusterAlgorithmComparison/backend/sp500_kmeans_clusters.csv"
    print(f"Saving cluster results to {output_csv}...")
    data.to_csv(output_csv, index=True)

    # Calculate within-cluster variances and save
    cluster_variances = kmeans.calculate_within_cluster_variances(feature_data)
    variances_csv = "ClusterAlgorithmComparison/backend/sp500_kmeans_variances.csv"
    print(f"Saving cluster variances to {variances_csv}...")
    cluster_variances.to_csv(variances_csv, index=True)

    # Visualize clustering
    plt.scatter(feature_data[:, 0], feature_data[:, 1], c=labels, cmap="viridis", s=10)
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="x", s=100, label="Centroids")
    plt.title("K-Means Clustering Results")
    plt.xlabel("Feature 1 (Standardized)")
    plt.ylabel("Feature 2 (Standardized)")
    plt.legend()
    plt.show()
