import { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import axios from "axios";
import "./Analysis.css";
import Header from "./Header";
import Footer from "./Footer";
import Graph from "./Graph";
import Data1 from "../../../backend/test.csv?url";
import Data2 from "../assets/sampledata2.csv?url";

interface ClusteringMetrics {
    Silhouette_Score_GMM: number;
    Silhouette_Score_KMeans: number;
    Calinski_Harabasz_GMM: number;
    Calinski_Harabasz_KMeans: number;
    Davies_Bouldin_GMM: number;
    Davies_Bouldin_KMeans: number;
}

const Analysis = () => {
    const location = useLocation();
    const { input1, input2 } = location.state || {};
    const [kmeansResult, setKmeansResult] = useState("");
    const [gmmResult, setGmmResult] = useState("");
    const [metrics, setMetrics] = useState<ClusteringMetrics | null>(null);

    const fetchClusteringData = async () => {
        try {
            const kmeans_response = await axios.get("http://localhost:5000/api/kmeans");
            const gmm_response = await axios.get("http://localhost:5000/api/gmm");

            // CSV file paths from backend
            setKmeansResult(kmeans_response.data.clusters_output_path);
            setGmmResult(gmm_response.data.clusters_output_path);

            // Set metrics
            setMetrics(kmeans_response.data.metrics);
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
                <h5>Input 1</h5>
                <p>{input1}</p>
                <h5>Input 2</h5>
                <p>{input2}</p>
            </div>

            <div className="graph-section">
                <Graph
                    csv={Data1}
                    title="K-Means Clustering"
                    xAxisTitle="t-SNE Dimension 1"
                    yAxisTitle="t-SNE Dimension 2"
                />
                <Graph
                    csv={Data1}
                    title="K-Means Clustering"
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
