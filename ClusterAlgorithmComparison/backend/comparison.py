import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from typing import Tuple

def load_cluster_results(kmeans_csv: str, gmm_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load cluster assignments from KMeans and GMM results.

    Args:
        kmeans_csv (str): Path to the KMeans cluster results CSV file.
        gmm_csv (str): Path to the GMM cluster results CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing KMeans and GMM cluster assignments.
    """
    kmeans_results = pd.read_csv(kmeans_csv)
    gmm_results = pd.read_csv(gmm_csv)
    return kmeans_results, gmm_results

def compute_silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Silhouette Score for clustering.

    Args:
        data (np.ndarray): The feature data used for clustering.
        labels (np.ndarray): Cluster labels.

    Returns:
        float: The Silhouette Score.
    """
    return silhouette_score(data, labels)

def plot_comparison(kmeans_data: pd.DataFrame, gmm_data: pd.DataFrame, feature_data: np.ndarray):
    """
    Plot side-by-side comparisons of KMeans and GMM clusters.

    Args:
        kmeans_data (pd.DataFrame): KMeans cluster assignments and centroids.
        gmm_data (pd.DataFrame): GMM cluster assignments and centroids.
        feature_data (np.ndarray): The original feature data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot KMeans Clusters
    axes[0].scatter(feature_data[:, 0], feature_data[:, 1], c=kmeans_data['KMeans_Cluster'], cmap='viridis', alpha=0.6)
    axes[0].scatter(kmeans_data['Centroid_0'], kmeans_data['Centroid_1'], c='red', marker='x', label='Centroids')
    axes[0].set_title("KMeans Clustering")
    axes[0].legend()

    # Plot GMM Clusters
    axes[1].scatter(feature_data[:, 0], feature_data[:, 1], c=gmm_data['Cluster'], cmap='viridis', alpha=0.6)
    axes[1].scatter(gmm_data['Centroid_0'], gmm_data['Centroid_1'], c='red', marker='x', label='Centroids')
    axes[1].set_title("GMM Clustering")
    axes[1].legend()

    plt.show()

def generate_summary(kmeans_silhouette: float, gmm_log_likelihood: float):
    """
    Generate a comparison summary of KMeans and GMM results.

    Args:
        kmeans_silhouette (float): Silhouette Score for KMeans.
        gmm_log_likelihood (float): Log-Likelihood value for GMM.

    Prints:
        A textual summary of the clustering comparison.
    """
    print("### Comparison Summary ###")
    print(f"KMeans Silhouette Score: {kmeans_silhouette:.4f}")
    print(f"GMM Log-Likelihood: {gmm_log_likelihood:.4f}")
    print("\nObservations:")
    print("- KMeans clusters evaluated with Silhouette Score; higher is better.")
    print("- GMM clusters evaluated with Log-Likelihood; higher indicates better fit.")
    print("- Examine cluster visuals for separation and density.")

if __name__ == "__main__":
    # File paths
    kmeans_csv = "ClusterAlgorithmComparison/backend/sp500_kmeans_clusters.csv"
    gmm_csv = "ClusterAlgorithmComparison/backend/sp500_gmm_clusters.csv"
    feature_data_csv = "ClusterAlgorithmComparison/backend/sp500_preprocessed_data.csv"

    # Load feature data and cluster results
    print("Loading feature data and cluster results...")
    feature_data = pd.read_csv(feature_data_csv, index_col=0).values
    kmeans_data, gmm_data = load_cluster_results(kmeans_csv, gmm_csv)

    # Compute metrics
    print("Computing Silhouette Score for KMeans...")
    kmeans_silhouette = compute_silhouette_score(feature_data, kmeans_data['KMeans_Cluster'].values)

    print("Retrieving Log-Likelihood for GMM...")
    gmm_log_likelihood = gmm_data['Log_Likelihood'].iloc[0]

    # Plot comparison
    print("Plotting cluster comparison...")
    plot_comparison(kmeans_data, gmm_data, feature_data)

    # Generate and print summary
    print("Generating comparison summary...")
    generate_summary(kmeans_silhouette, gmm_log_likelihood)
