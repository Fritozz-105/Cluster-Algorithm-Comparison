import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
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

    def __init__(self, n_clusters: int, max_iter: int = 100, tol: float = 1e-4, reg_covar: float = 1e-6) -> None:
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
        Initialize the parameters (weights, means, covariances) randomly.

        Args:
            X (np.ndarray): Input feature data.
        """
        n_samples, n_features = X.shape
        self.weights = np.full(self.n_clusters, 1 / self.n_clusters)
        self.means = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        self.covariances = np.array([np.cov(X, rowvar=False) + self.reg_covar * np.eye(n_features)
                                     for _ in range(self.n_clusters)])

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
                # Compute the PDF for the k-th cluster
                responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(
                    X, mean=self.means[k], cov=self.covariances[k], allow_singular=True
                )
            except Exception as e:
                print(f"Error in responsibility computation for cluster {k}: {e}")
                responsibilities[:, k] = 1e-10  # Assign a small value to avoid zeros

        total_responsibility = responsibilities.sum(axis=1, keepdims=True)
        # Prevent division by zero in normalization
        total_responsibility[total_responsibility == 0] = 1e-10
        responsibilities /= total_responsibility
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
                # Ensure covariance matrix is SPD
                self.covariances[k] = np.clip(self.covariances[k], a_min=self.reg_covar, a_max=None)



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
                # Compute the weighted PDF for the k-th cluster
                cluster_pdf = self.weights[k] * multivariate_normal.pdf(
                    X, mean=self.means[k], cov=self.covariances[k], allow_singular=True
                )
                # Clip values to avoid log(0) or inf
                cluster_pdf = np.clip(cluster_pdf, a_min=1e-10, a_max=None)
                log_likelihood += np.log(cluster_pdf).sum()
            except Exception as e:
                print(f"Error in log-likelihood computation for cluster {k}: {e}")
                continue
        return log_likelihood


    def fit(self, X: np.ndarray) -> None:
        """
        Fit the Gaussian Mixture Model to the data.

        Args:
            X (np.ndarray): Input feature data.
        """
        n_samples, n_features = X.shape
        self.responsibilities = np.zeros((n_samples, self.n_clusters))

        # Initialize parameters
        print("Initializing parameters...")
        kmeans = KMeansClustering(n_clusters=self.n_clusters, max_iter=300, tol=1e-4)
        labels, centroids = kmeans.fit(X)
        self.means = kmeans.centroids
        self.covariances = np.array(
            [np.cov(X[labels == k].T) + self.reg_covar * np.eye(n_features) for k in range(self.n_clusters)]
        )
        self.weights = np.array([np.mean(labels == k) for k in range(self.n_clusters)])
        
        # Handle potential empty clusters during initialization
        for k in range(self.n_clusters):
            if np.sum(labels == k) == 0:
                print(f"Cluster {k} is empty during initialization. Reinitializing...")
                self.means[k] = X[np.random.choice(n_samples)]
                self.covariances[k] = np.cov(X.T) + self.reg_covar * np.eye(n_features)

        # EM algorithm
        log_likelihood_old = None
        for iteration in range(self.max_iter):
            self.responsibilities = self._e_step(X)
            self._m_step(X)
            log_likelihood_new = self._compute_log_likelihood(X)

            if log_likelihood_old is not None:
                if abs(log_likelihood_new - log_likelihood_old) < self.tol:
                    print("Convergence achieved.")
                    break

            log_likelihood_old = log_likelihood_new
            print(f"Iteration {iteration + 1}, Log-Likelihood: {log_likelihood_new:.4f}")


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
        plt.xlabel("Feature 1 (Standardized)")
        plt.ylabel("Feature 2 (Standardized)")
        plt.legend()
        plt.show()


def validate_data(data: np.ndarray) -> np.ndarray:
    """
    Validate and clean input data by removing rows containing NaN or inf values.

    Args:
        data (np.ndarray): Input feature data.

    Returns:
        np.ndarray: Cleaned data without NaN or inf values.
    """
    if not np.isfinite(data).all():
        print("Warning: Data contains NaN or inf values. Cleaning data...")
        data = data[np.isfinite(data).all(axis=1)]
    if data.size == 0:
        raise ValueError("All rows were removed during cleaning. Check your dataset.")
    return data



if __name__ == "__main__":
    import os

    # File paths
    input_csv: str = "ClusterAlgorithmComparison/backend/sp500_preprocessed_data.csv"
    output_csv: str = "ClusterAlgorithmComparison/backend/sp500_gmm_clusters.csv"

    # Load preprocessed data
    print(f"Loading data from {input_csv}...")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"The file {input_csv} does not exist. Ensure preprocessing is complete.")
    data: pd.DataFrame = pd.read_csv(input_csv, index_col=0)

    # Standardize data
    print("Standardizing features...")
    feature_data: np.ndarray = data.values

    # Validate data
    feature_data = validate_data(feature_data)
    if feature_data.size == 0:
        raise ValueError("All rows were removed during cleaning. Check your dataset.")

    # Initialize and fit the GMM model
    n_clusters: int = 3
    print(f"Fitting Gaussian Mixture Model with {n_clusters} clusters...")
    gmm = GMM(n_clusters=n_clusters, max_iter=100, tol=1e-4, reg_covar=1e-6)
    gmm.fit(feature_data)

    # Predict cluster assignments
    print("Predicting cluster assignments...")
    labels: np.ndarray = gmm.predict(feature_data)

    # Save cluster assignments
    cluster_results: pd.DataFrame = data.copy()
    cluster_results["Cluster"] = labels
    print(f"Saving cluster assignments to {output_csv}...")
    cluster_results.to_csv(output_csv)

    # Visualize clustering results
    if feature_data.shape[1] >= 2:
        print("Visualizing clusters...")
        gmm.plot_clusters(feature_data, labels)
    else:
        print("Visualization skipped (data has fewer than 2 dimensions).")

    print("GMM clustering process complete.")
