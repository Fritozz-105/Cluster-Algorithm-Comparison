import os
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple
from kmeans_clustering import KMeansClustering


class GMM:
    """
    Gaussian Mixture Model (GMM) clustering implementation.

    Attributes:
        n_clusters (int): Number of clusters (Gaussian components).
        max_iter (int): Maximum number of iterations for the EM algorithm.
        tol (float): Convergence threshold for log-likelihood improvement.
        reg_covar (float): Regularization term for covariance matrices.
        weights (np.ndarray): Mixing weights of the Gaussian components.
        means (np.ndarray): Means of the Gaussian components.
        covariances (np.ndarray): Covariance matrices of the Gaussian components.
        responsibilities (np.ndarray): Responsibility matrix (soft assignments).
    """

    def __init__(self, n_clusters: int, max_iter: int = 100, tol: float = 1e-4, reg_covar: float = 1e-2) -> None:
        self.n_clusters: int = n_clusters
        self.max_iter: int = max_iter
        self.tol: float = tol
        self.reg_covar: float = reg_covar
        self.weights: np.ndarray = None
        self.means: np.ndarray = None
        self.covariances: np.ndarray = None
        self.responsibilities: np.ndarray = None

    def _initialize_parameters(self, X: np.ndarray) -> None:
        """
        Initialize the parameters (weights, means, covariances) using KMeans.

        Args:
            X (np.ndarray): Input feature data.
        """
        n_samples, n_features = X.shape
        kmeans = KMeansClustering(n_clusters=self.n_clusters, max_iter=300, tol=1e-4)
        labels, centroids = kmeans.fit(X)
        self.means = centroids
        self.weights = np.array([np.mean(labels == k) for k in range(self.n_clusters)])
        self.covariances = np.array(
            [np.cov(X[labels == k].T) + self.reg_covar * np.eye(n_features) for k in range(self.n_clusters)]
        )

        # Handle potential empty clusters during initialization
        for k in range(self.n_clusters):
            if np.sum(labels == k) == 0:
                print(f"Cluster {k} is empty during initialization. Reinitializing...")
                self.means[k] = X[np.random.choice(n_samples)]
                self.covariances[k] = np.cov(X.T) + self.reg_covar * np.eye(n_features)

        print("Initialization complete.")

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        Expectation step: Compute responsibilities.

        Args:
            X (np.ndarray): Input feature data.

        Returns:
            np.ndarray: Updated responsibility matrix.
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_clusters))

        for k in range(self.n_clusters):
            try:
                responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(
                    X, mean=self.means[k], cov=self.covariances[k], allow_singular=True
                )
            except Exception as e:
                print(f"Responsibility computation error for cluster {k}: {e}")
                responsibilities[:, k] = 1e-10  # Assign small value to prevent zero division

        total_responsibility = responsibilities.sum(axis=1, keepdims=True)
        total_responsibility[total_responsibility == 0] = 1e-10
        responsibilities /= total_responsibility

        print(f"Responsibilities shape: {responsibilities.shape}")
        return responsibilities

    def _m_step(self, X: np.ndarray) -> None:
        """
        Maximization step: Update weights, means, and covariances.

        Args:
            X (np.ndarray): Input feature data.
        """
        n_samples = X.shape[0]
        for k in range(self.n_clusters):
            responsibility = self.responsibilities[:, k]
            total_responsibility = responsibility.sum()

            if total_responsibility == 0:
                print(f"Cluster {k} is empty. Reinitializing...")
                self.means[k] = X[np.random.choice(n_samples)]
                self.covariances[k] = np.cov(X.T) + self.reg_covar * np.eye(X.shape[1])
                self.weights[k] = 1.0 / self.n_clusters
            else:
                self.weights[k] = total_responsibility / n_samples
                self.means[k] = np.sum(responsibility[:, np.newaxis] * X, axis=0) / total_responsibility
                diff = X - self.means[k]
                self.covariances[k] = (
                    np.dot(responsibility * diff.T, diff) / total_responsibility
                    + self.reg_covar * np.eye(X.shape[1])
                )

    def _compute_log_likelihood(self, X: np.ndarray) -> float:
        """
        Compute the log-likelihood of the data given the current parameters.

        Args:
            X (np.ndarray): Input feature data.

        Returns:
            float: Log-likelihood value.
        """
        log_likelihood = 0.0
        for k in range(self.n_clusters):
            try:
                cluster_pdf = self.weights[k] * multivariate_normal.pdf(
                    X, mean=self.means[k], cov=self.covariances[k], allow_singular=True
                )
                log_likelihood += np.log(np.clip(cluster_pdf, 1e-10, None)).sum()
            except Exception as e:
                print(f"Log-likelihood computation error for cluster {k}: {e}")
        return log_likelihood

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the Gaussian Mixture Model to the data.

        Args:
            X (np.ndarray): Input feature data.
        """
        self._initialize_parameters(X)
        log_likelihood_old = -np.inf

        for iteration in range(self.max_iter):
            self.responsibilities = self._e_step(X)
            self._m_step(X)
            log_likelihood_new = self._compute_log_likelihood(X)

            print(f"Iteration {iteration + 1}, Log-Likelihood: {log_likelihood_new:.4f}")
            if np.abs(log_likelihood_new - log_likelihood_old) < self.tol:
                print("Convergence achieved.")
                break

            log_likelihood_old = log_likelihood_new

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for the input data.

        Args:
            X (np.ndarray): Input feature data.

        Returns:
            np.ndarray: Cluster labels for each data point.
        """
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def plot_clusters(self, X: np.ndarray, labels: np.ndarray) -> None:
        """
        Visualize clustering results in a 2D plot.

        Args:
            X (np.ndarray): Input feature data.
            labels (np.ndarray): Cluster labels for each data point.
        """
        plt.figure(figsize=(10, 6))
        for cluster in range(self.n_clusters):
            cluster_points = X[labels == cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")
        plt.scatter(self.means[:, 0], self.means[:, 1], c="red", marker="x", s=100, label="Centroids")
        plt.title("GMM Clustering Results")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    input_csv = "ClusterAlgorithmComparison/backend/sp500_preprocessed_data.csv"
    output_csv = "ClusterAlgorithmComparison/backend/sp500_gmm_clusters.csv"

    print(f"Loading data from {input_csv}...")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"The file {input_csv} does not exist. Ensure preprocessing is complete.")

    data = pd.read_csv(input_csv, index_col=0)

    print("Standardizing features...")
    scaler = StandardScaler()
    feature_data = scaler.fit_transform(data.values)

    print("Performing PCA for dimensionality reduction...")
    pca = PCA(n_components=2)
    feature_data_pca = pca.fit_transform(feature_data)

    print(f"Fitting GMM with {2} clusters...")
    gmm = GMM(n_clusters=2, max_iter=100, tol=1e-4, reg_covar=1e-2)
    gmm.fit(feature_data_pca)

    print("Predicting cluster assignments...")
    labels = gmm.predict(feature_data_pca)

    print("Visualizing clusters...")
    gmm.plot_clusters(feature_data_pca, labels)
