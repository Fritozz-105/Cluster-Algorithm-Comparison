# S&P 500 Clustering Project

## Overview
This project applies clustering techniques to analyze S&P 500 stock data based on their historical adjusted close prices. The goal is to group stocks into clusters that share similar characteristics, using **K-Means Clustering** and **Gaussian Mixture Model** as the primary algorithms. The project involves data preprocessing, cluster analysis, and validation, providing insights into market behavior and asset groupings.

---

## Prerequisites

### Install Node JS
Refer to https://nodejs.org/en/ to install Node.js

### Installation Poetry
This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging. Poetry allows for an easy and reliable way to manage project dependencies, ensuring a consistent

To get started, you need to have Poetry installed. You can install Poetry by following the instructions on the [official documentation](https://python-poetry.org/docs/#installation).

Alternatively, you can use this command to install it:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Installation
Clone the repo
   ```sh
   https://github.com/Fritozz-105/Cluster-Algorithm-Comparison.git
   ```
Install NPM packages
   ```sh
   npm install
   ```
To run the application locally use in the frontend directory:

```bash
npm run dev
```

And use this command in the backend directory:
```bash
python main.py
```

## Project Features

### Frontend
1. **Navigation**:
   - Navigate towards the cluster algorithm analysis page by pressing the next button.
   - Click the logo in the header to return to the start page.

2. **Data Display**:
   - Fetches the CSV data returned from the **K-Means Clustering Algorithm** and **Gausian Mixture Model Algorithm** and displays it in a scatterplot.
   - Fetches the Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index to compare the performance between the two algorithms and displays it as a table.

### Backend
1. **Data Collection and Cleaning**:
   - The list of S&P 500 tickers is fetched dynamically from Wikipedia.
   - Historical adjusted close prices for each ticker are fetched using the `yfinance` library.
   - Stocks with incomplete data are excluded to ensure consistency.

2. **Data Preprocessing**:
   - Daily **returns** are calculated for each stock to measure price changes.
   - Rolling **volatility** is calculated over a 20-day window to assess risk.

3. **Clustering Algorithm**:
   - **K-Means Clustering** and **Gaussian Mixture Model** are implemented from scratch to identify clusters in the data.
   - The number of clusters is determined using the **Elbow Method**, which evaluates distortion as a function of cluster count.

4. **Cluster Validation**:
   - Within-cluster variances are calculated to measure cluster compactness and validate clustering performance.
   - Other metrics are calculated for comparison, such as silhouette score, calinski-harabasz Index, and Davies-Bouldin Index.
