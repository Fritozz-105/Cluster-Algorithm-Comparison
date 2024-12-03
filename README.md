# S&P 500 Clustering Project

## Overview
This project applies clustering techniques to analyze S&P 500 stock data based on their historical adjusted close prices. The goal is to group stocks into clusters that share similar characteristics, using **K-Means Clustering** as the primary algorithm. The project involves data preprocessing, cluster analysis, and validation, providing insights into market behavior and asset groupings.

The repository is structured to allow scalability for additional clustering methods and advanced analytics.

---

## Project Features

### Backend
1. **Data Collection and Cleaning**:
   - The list of S&P 500 tickers is fetched dynamically from Wikipedia.
   - Historical adjusted close prices for each ticker are fetched using the `yfinance` library.
   - Stocks with incomplete data are excluded to ensure consistency.

2. **Data Preprocessing**:
   - Daily **returns** are calculated for each stock to measure price changes.
   - Rolling **volatility** is calculated over a 20-day window to assess risk.
   - Both features are normalized using Z-scores to standardize the data.

3. **Clustering Algorithm**:
   - **K-Means Clustering** is implemented from scratch to identify clusters in the data.
   - The number of clusters is determined using the **Elbow Method**, which evaluates distortion as a function of cluster count.

4. **Cluster Validation**:
   - Within-cluster variances are calculated to measure cluster compactness and validate clustering performance.

5. **Outputs**:
   - Cluster assignments for each stock are saved to a CSV file.
   - Within-cluster variances are stored for deeper analysis.
   - Visualizations are generated for clustering results and the elbow method.

### Frontend
- A minimal frontend is scaffolded using **Vite** and **React** to provide visualization capabilities or display outputs if needed in the future. Currently, it includes placeholder files in the `src` directory.

---

## File Structure
Cluster-Algorithm-Comparison/ ├── backend/ │ ├── sp500_adj_close_data.csv # Raw adjusted close price data │ ├── sp500_preprocessed_data.csv # Preprocessed data (returns & volatility) │ ├── sp500_kmeans_clusters.csv # Clustering results with assigned labels │ ├── sp500_kmeans_variances.csv # Within-cluster variance analysis │ ├── data_preprocessing.py # Data collection and preprocessing script │ ├── kmeans_clustering.py # Main K-Means clustering implementation │ ├── main.py # Entry point for running backend scripts ├── frontend/ │ ├── src/ # React frontend files │ │ ├── App.tsx │ │ ├── App.css │ │ ├── main.tsx │ │ ├── index.css │ │ ├── vite-env.d.ts │ ├── public/ # Public assets ├── resources/ │ ├── MARKDOWN.md # Miscellaneous markdown notes ├── .gitignore # Git ignore file ├── README.md # Project documentation ├── pyproject.toml # Poetry configuration ├── poetry.lock # Poetry lock file