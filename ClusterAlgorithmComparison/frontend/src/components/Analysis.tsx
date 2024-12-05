import { useState, useEffect } from "react";
import axios from "axios";
import "./Analysis.css";
import Header from "./Header";
import Footer from "./Footer";
import Graph from "./Graph";

interface ClusteringMetrics {
    Silhouette_Score_GMM: number;
    Silhouette_Score_KMeans: number;
    Calinski_Harabasz_GMM: number;
    Calinski_Harabasz_KMeans: number;
    Davies_Bouldin_GMM: number;
    Davies_Bouldin_KMeans: number;
}

const Analysis = () => {
    const [kmeansResult, setKmeansResult] = useState("");
    const [gmmResult, setGmmResult] = useState("");
    const [metrics, setMetrics] = useState<ClusteringMetrics | null>(null);

    const fetchClusteringData = async () => {
        try {
            const results = await axios.get("http://localhost:5000/api/clustering-results");

            setKmeansResult(results.data.kmeans_csv);
            setGmmResult(results.data.gmm_csv);

            setMetrics(results.data.metrics);
        } catch (error) {
            console.error("Error fetching clustering data:", error);
        }
    };

    useEffect(() => {
        fetchClusteringData();
    }, []);

    return (
        <div className="analysis-content">
            <Header />

            <div className="analysis-title">
                <h1>Clustering Analysis</h1>
            </div>
            <div className="analysis-form">
                <p>
                GMM is a soft clustering method, which means it assigns probabilities for each point belonging to different clusters rather than forcing them into a single cluster. This makes it better suited for data with overlapping distributions. K-Means, on the other hand, is a hard clustering method that strictly assigns each data point to one cluster, making it faster but potentially less flexible when the data structure is more complex. To ensure our models handle the data well, we’ve been preprocessing it using standard scaling (to make sure features are comparable by removing mean and scaling to unit variance) and PCA (Principal Component Analysis) to reduce dimensionality. PCA helps retain most of the variance in fewer dimensions, which not only makes clustering more efficient but also improves visualization. For both models, we used t-SNE to reduce the data to two dimensions for visualization purposes while preserving the clustering relationships as best as possible. To evaluate performance, we calculated several metrics. The Silhouette Score measures how well-separated the clusters are, with higher scores indicating better-defined groups. The Calinski-Harabasz Index assesses the compactness of clusters relative to their separation, where higher values mean better-defined clusters. The Davies-Bouldin Index looks at the average similarity ratio between clusters, with lower values indicating better clustering. In our results, K-Means had a higher Silhouette Score and Calinski-Harabasz Index and a lower Davies-Bouldin Index compared to GMM, suggesting it performed better overall. However, GMM still provides valuable flexibility for datasets where clusters may overlap.
                </p>
            </div>

            <div className="graph-section">
                <Graph
                    csv={kmeansResult}
                    title="K-Means Clustering"
                    xAxisTitle="t-SNE Dimension 1"
                    yAxisTitle="t-SNE Dimension 2"
                />
                <Graph
                    csv={gmmResult}
                    title="Gaussian Mixture Model"
                    xAxisTitle="t-SNE Dimension 1"
                    yAxisTitle="t-SNE Dimension 2"
                />
            </div>

            {metrics && (
                <div className="metrics-section">
                    <h2>Clustering Performance Metrics</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>GMM</th>
                                <th>K-Means</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Silhouette Score</td>
                                <td>{metrics.Silhouette_Score_GMM.toFixed(4)}</td>
                                <td>{metrics.Silhouette_Score_KMeans.toFixed(4)}</td>
                            </tr>
                            <tr>
                                <td>Calinski-Harabasz Index</td>
                                <td>{metrics.Calinski_Harabasz_GMM.toFixed(4)}</td>
                                <td>{metrics.Calinski_Harabasz_KMeans.toFixed(4)}</td>
                            </tr>
                            <tr>
                                <td>Davies-Bouldin Index</td>
                                <td>{metrics.Davies_Bouldin_GMM.toFixed(4)}</td>
                                <td>{metrics.Davies_Bouldin_KMeans.toFixed(4)}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            )}

            <Footer />
        </div>
    );
};

export default Analysis;
