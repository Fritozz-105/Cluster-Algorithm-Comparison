from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

from kmeans_clustering import KMeansClustering
from gmm_clustering import GMM

app = Flask(__name__)
cors = CORS(app, origins="*")

@app.route("/api/books", methods=["GET"])
def greeting():
    return jsonify(
        {
            "books": [
                "To Kill a Mocking Bird",
                "Hamlet",
                "Brave New World",
                "1984",
                "The Great Gatsby"
            ]
        }
    )

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

    # Fit GMM
    gmm = GMM(n_clusters=2)
    gmm.fit(pca_data)
    gmm_labels = gmm.predict(pca_data)

    # Fit K-Means
    kmeans = KMeansClustering(n_clusters=2)
    kmeans_labels, _ = kmeans.fit(pca_data)

    # Compute metrics
    silhouette_gmm = silhouette_score(pca_data, gmm_labels)
    silhouette_kmeans = silhouette_score(pca_data, kmeans_labels)
    calinski_gmm = calinski_harabasz_score(pca_data, gmm_labels)
    calinski_kmeans = calinski_harabasz_score(pca_data, kmeans_labels)
    davies_gmm = davies_bouldin_score(pca_data, gmm_labels)
    davies_kmeans = davies_bouldin_score(pca_data, kmeans_labels)

    gmm_df = pd.DataFrame({'GMM_Cluster': gmm_labels})
    kmeans_df = pd.DataFrame({'KMeans_Cluster': kmeans_labels})

    metrics = {
        'Silhouette_Score_GMM': silhouette_gmm,
        'Silhouette_Score_KMeans': silhouette_kmeans,
        'Calinski_Harabasz_GMM': calinski_gmm,
        'Calinski_Harabasz_KMeans': calinski_kmeans,
        'Davies_Bouldin_GMM': davies_gmm,
        'Davies_Bouldin_KMeans': davies_kmeans
    }

    # Save CSVs
    os.makedirs('clustering_results', exist_ok=True)
    gmm_df.to_csv('clustering_results/gmm_clusters.csv', index=False)
    kmeans_df.to_csv('clustering_results/kmeans_clusters.csv', index=False)

    return metrics

@app.route("/api/kmeans", methods=["GET"])
def kmeans_route():
    metrics = perform_clustering()
    return jsonify({
        "clusters_output_path": "../../../backend/kmeans_clusters.csv",
        "metrics": metrics
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
