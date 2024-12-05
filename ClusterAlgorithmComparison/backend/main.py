from flask import Flask, jsonify, make_response
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import io

from kmeans_clustering import KMeansClustering
from gmm_clustering import GMM

app = Flask(__name__)
cors = CORS(app, origins="*")

def perform_clustering():
    # Load data
    input_csv = "ClusterAlgorithmComparison/backend/sp500_preprocessed_data.csv"
    data = pd.read_csv(input_csv, index_col=0)

    # Standardize features
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data.values)

    # Apply PCA
    pca = PCA(n_components=5)
    pca_data = pca.fit_transform(standardized_data)

    # Fit K-Means
    kmeans = KMeansClustering(n_clusters=2)
    kmeans_labels, _ = kmeans.fit(pca_data)

    # Apply GMM
    gmm = GMM(n_clusters=2)
    gmm.fit(pca_data)
    gmm_labels = gmm.predict(pca_data)

    # Compute metrics
    metrics = {
        'Silhouette_Score_KMeans': silhouette_score(pca_data, kmeans_labels),
        'Silhouette_Score_GMM': silhouette_score(pca_data, gmm_labels),
        'Calinski_Harabasz_KMeans': calinski_harabasz_score(pca_data, kmeans_labels),
        'Calinski_Harabasz_GMM': calinski_harabasz_score(pca_data, gmm_labels),
        'Davies_Bouldin_KMeans': davies_bouldin_score(pca_data, kmeans_labels),
        'Davies_Bouldin_GMM': davies_bouldin_score(pca_data, gmm_labels)
    }

    # Apply t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=40, random_state=42)
    tsne_data = tsne.fit_transform(pca_data)

    # Create DataFrame for t-SNE results
    tsne_reduced_data = pd.DataFrame(tsne_data, columns=["t-SNE Dimension 1", "t-SNE Dimension 2"])
    tsne_reduced_data["KMeans_Cluster"] = kmeans_labels
    tsne_reduced_data["GMM_Cluster"] = gmm_labels

    # Convert DataFrame to CSV string
    csv_string_kmeans = tsne_reduced_data.to_csv(index=False, columns=["t-SNE Dimension 1", "t-SNE Dimension 2", "KMeans_Cluster"])
    csv_string_gmm = tsne_reduced_data.to_csv(index=False, columns=["t-SNE Dimension 1", "t-SNE Dimension 2", "GMM_Cluster"])

    return metrics, csv_string_kmeans, csv_string_gmm

@app.route("/api/clustering-results", methods=["GET"])
def clustering_results():
    metrics, csv_string_kmeans, csv_string_gmm = perform_clustering()
    return jsonify({
        "metrics": metrics,
        "kmeans_csv": csv_string_kmeans,
        "gmm_csv": csv_string_gmm
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
