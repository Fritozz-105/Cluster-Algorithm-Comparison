import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from gmm_clustering import GMM
from kmeans_clustering import KMeansClustering

# Load data
input_csv = "ClusterAlgorithmComparison/backend/sp500_preprocessed_data.csv"
print(f"Loading data from {input_csv}...")
data = pd.read_csv(input_csv, index_col=0)

# Standardize features
print("Standardizing features...")
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data.values)

# Apply PCA
print("Performing PCA for dimensionality reduction...")
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca_data = pca.fit_transform(standardized_data)
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

# Fit GMM
print("Fitting GMM...")
gmm = GMM(n_clusters=2)
gmm.fit(pca_data)
gmm_labels = gmm.predict(pca_data)

# Fit K-Means
print("Fitting K-Means...")
kmeans = KMeansClustering(n_clusters=2)
kmeans_labels, _ = kmeans.fit(pca_data)

# Compute metrics
print("Calculating clustering metrics...")
silhouette_gmm = silhouette_score(pca_data, gmm_labels)
silhouette_kmeans = silhouette_score(pca_data, kmeans_labels)
calinski_gmm = calinski_harabasz_score(pca_data, gmm_labels)
calinski_kmeans = calinski_harabasz_score(pca_data, kmeans_labels)
davies_gmm = davies_bouldin_score(pca_data, gmm_labels)
davies_kmeans = davies_bouldin_score(pca_data, kmeans_labels)

print("\nClustering Performance Metrics:")
print(f"Silhouette Score (GMM): {silhouette_gmm:.4f}, (K-Means): {silhouette_kmeans:.4f}")
print(f"Calinski-Harabasz Index (GMM): {calinski_gmm:.4f}, (K-Means): {calinski_kmeans:.4f}")
print(f"Davies-Bouldin Index (GMM): {davies_gmm:.4f}, (K-Means): {davies_kmeans:.4f}")

# Visualize comparison with t-SNE
print("Generating t-SNE visualization for both models...")
tsne = TSNE(n_components=2, perplexity=40, random_state=42)
tsne_data = tsne.fit_transform(pca_data)

plt.figure(figsize=(12, 8))
# GMM clusters
plt.scatter(tsne_data[gmm_labels == 0, 0], tsne_data[gmm_labels == 0, 1], label="GMM Cluster 0", alpha=0.6)
plt.scatter(tsne_data[gmm_labels == 1, 0], tsne_data[gmm_labels == 1, 1], label="GMM Cluster 1", alpha=0.6)

# K-Means clusters
plt.scatter(tsne_data[kmeans_labels == 0, 0], tsne_data[kmeans_labels == 0, 1], marker="x", label="K-Means Cluster 0", alpha=0.6)
plt.scatter(tsne_data[kmeans_labels == 1, 0], tsne_data[kmeans_labels == 1, 1], marker="x", label="K-Means Cluster 1", alpha=0.6)

plt.title("GMM vs. K-Means Clustering Results (t-SNE)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.show()
