import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from typing import Tuple, Optional


class GMM:
    """
    Gaussian Mixture Model (GMM) clustering with enhancements for visualization and evaluation.

    Attributes:
        n_clusters (int): Number of clusters.
        max_iter (int): Maximum iterations for the EM algorithm.
        tol (float): Convergence threshold for log-likelihood improvement.
        reg_covar (float): Regularization term for covariance matrices.
        weights (Optional[np.ndarray]): Mixing weights of the Gaussian components.
        means (Optional[np.ndarray]): Means of the Gaussian components.
        covariances (Optional[np.ndarray]): Covariance matrices of the Gaussian components.
        responsibilities (Optional[np.ndarray]): Responsibility matrix (soft assignments).
    """

    def __init__(self, n_clusters: int, max_iter: int = 100, tol: float = 1e-4, reg_covar: float = 1e-6) -> None:
        """
        Initialize the GMM model.

        Args:
            n_clusters (int): Number of clusters.
            max_iter (int): Maximum iterations for the EM algorithm.
            tol (float): Convergence threshold for log-likelihood improvement.
            reg_covar (float): Regularization term for covariance matrices.
        """
        self.n_clusters: int = n_clusters
        self.max_iter: int = max_iter
        self.tol: float = tol
        self.reg_covar: float = reg_covar
        self.weights: Optional[np.ndarray] = None
        self.means: Optional[np.ndarray] = None
        self.covariances: Optional[np.ndarray] = None
        self.responsibilities: Optional[np.ndarray] = None

    def _initialize_parameters(self, X: np.ndarray) -> None:
        """
        Initialize the parameters (weights, means, covariances).

        Args:
            X (np.ndarray): Input feature data of shape (n_samples, n_features).
        """
        n_samples, n_features = X.shape
        self.weights = np.full(self.n_clusters, 1 / self.n_clusters)
        self.means = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        self.covariances = np.array([np.cov(X.T) + self.reg_covar * np.eye(n_features) for _ in range(self.n_clusters)])

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        Expectation step: Compute responsibilities.

        Args:
            X (np.ndarray): Input feature data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Updated responsibility matrix of shape (n_samples, n_clusters).
        """
        n_samples: int = X.shape[0]
        responsibilities: np.ndarray = np.zeros((n_samples, self.n_clusters))

        for k in range(self.n_clusters):
            try:
                responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(
                    X, mean=self.means[k], cov=self.covariances[k], allow_singular=True
                )
            except Exception as e:
                print(f"Error in responsibility computation for cluster {k}: {e}")
                responsibilities[:, k] = 1e-10  # Assign a small value to avoid zeros

        total_responsibility: np.ndarray = responsibilities.sum(axis=1, keepdims=True)
        total_responsibility[total_responsibility == 0] = 1e-10  # Prevent division by zero
        responsibilities /= total_responsibility
        return responsibilities

    def _m_step(self, X: np.ndarray) -> None:
        """
        Maximization step: Update weights, means, and covariances.

        Args:
            X (np.ndarray): Input feature data of shape (n_samples, n_features).
        """
        n_samples: int = X.shape[0]

        for k in range(self.n_clusters):
            responsibility: np.ndarray = self.responsibilities[:, k]
            total_responsibility: float = responsibility.sum()

            if total_responsibility == 0:
                print(f"Cluster {k} is empty. Reinitializing...")
                self.means[k] = X[np.random.choice(n_samples)]
                self.covariances[k] = np.cov(X.T) + self.reg_covar * np.eye(X.shape[1])
                self.weights[k] = 1.0 / self.n_clusters
            else:
                self.weights[k] = total_responsibility / n_samples
                self.means[k] = np.sum(responsibility[:, np.newaxis] * X, axis=0) / total_responsibility
                diff: np.ndarray = X - self.means[k]
                self.covariances[k] = (
                    np.dot(responsibility * diff.T, diff) / total_responsibility
                    + self.reg_covar * np.eye(X.shape[1])
                )

    def _compute_log_likelihood(self, X: np.ndarray) -> float:
        """
        Compute the log-likelihood of the data given the current parameters.

        Args:
            X (np.ndarray): Input feature data of shape (n_samples, n_features).

        Returns:
            float: Log-likelihood value.
        """
        log_likelihood: float = 0.0
        for k in range(self.n_clusters):
            try:
                cluster_pdf: np.ndarray = self.weights[k] * multivariate_normal.pdf(
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
            X (np.ndarray): Input feature data of shape (n_samples, n_features).
        """
        n_samples, n_features = X.shape
        self._initialize_parameters(X)
        self.responsibilities = np.zeros((n_samples, self.n_clusters))
        log_likelihood_old: Optional[float] = None

        for iteration in range(self.max_iter):
            self.responsibilities = self._e_step(X)
            self._m_step(X)
            log_likelihood_new: float = self._compute_log_likelihood(X)

            print(f"Iteration {iteration + 1}, Log-Likelihood: {log_likelihood_new:.4f}")
            if log_likelihood_old is not None and abs(log_likelihood_new - log_likelihood_old) < self.tol:
                print("Convergence achieved.")
                break
            log_likelihood_old = log_likelihood_new

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for the input data.

        Args:
            X (np.ndarray): Input feature data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Cluster labels for each data point.
        """
        responsibilities: np.ndarray = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def plot_clusters(self, X: np.ndarray, labels: np.ndarray, title: str = "GMM Clustering Results") -> None:
        """
        Visualize clustering results in a 2D plot using t-SNE.

        Args:
            X (np.ndarray): Input feature data of shape (n_samples, n_features).
            labels (np.ndarray): Cluster labels for each data point.
            title (str): Title of the plot.
        """
        tsne = TSNE(n_components=2, perplexity=40, random_state=42)
        reduced_data = tsne.fit_transform(X)

        plt.figure(figsize=(10, 6))
        for cluster in range(self.n_clusters):
            cluster_points = reduced_data[labels == cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")

        plt.scatter(
            self.means[:, 0], self.means[:, 1], c="red", marker="x", s=100, label="Centroids"
        )
        plt.title(title)
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend()
        plt.show()


if __name__ == "__main__":
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
    scaler: StandardScaler = StandardScaler()
    feature_data: np.ndarray = scaler.fit_transform(data.values)

    # Validate data
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

    feature_data = validate_data(feature_data)

    # Dimensionality reduction using PCA
    print("Performing PCA for dimensionality reduction...")
    n_components: int = 5  # Reduce to 2 dimensions for simplicity and visualization
    pca: PCA = PCA(n_components=n_components)
    reduced_data: np.ndarray = pca.fit_transform(feature_data)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    # Fit GMM model
    n_clusters: int = 2
    print(f"Fitting GMM with {n_clusters} clusters...")
    gmm: GMM = GMM(n_clusters=n_clusters, max_iter=100, tol=1e-4, reg_covar=1e-9)
    gmm.fit(reduced_data)

    # Predict cluster assignments
    print("Predicting cluster assignments...")
    labels: np.ndarray = gmm.predict(reduced_data)

    # Save cluster assignments and log-likelihood
    def save_cluster_results(file_path: str, labels: np.ndarray, log_likelihood: float):
        """
        Save GMM cluster assignments and log-likelihood to a CSV file.

        Args:
            file_path (str): Path to the output CSV file.
            labels (np.ndarray): Cluster assignments.
            log_likelihood (float): Final log-likelihood of the model.
        """
        print("Saving cluster assignments and log-likelihood...")
        results = pd.DataFrame({
            "Cluster": labels,
            "Log_Likelihood": [log_likelihood] + [None] * (len(labels) - 1)
        })
        results.to_csv(file_path, index=False)
        print(f"Cluster assignments and log-likelihood saved to {file_path}.")

    log_likelihood: float = gmm._compute_log_likelihood(reduced_data)
    save_cluster_results(output_csv, labels, log_likelihood)

    # Visualize clusters
    print("Visualizing clusters...")
    gmm.plot_clusters(reduced_data, labels)

    print("GMM clustering process complete.")
