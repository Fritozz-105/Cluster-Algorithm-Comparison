# S&P 500 Clustering Project

## Overview
This project applies clustering techniques to analyze S&P 500 stock data based on their historical adjusted close prices. The goal is to group stocks into clusters that share similar characteristics, using **K-Means Clustering** and **Gaussian Mixture Model** as the primary algorithms. The project involves data preprocessing, cluster analysis, and validation, providing insights into market behavior and asset groupings.

---

## Prerequisites

### Install Node JS
Refer to https://nodejs.org/en/ to install Node.js

## Cloning and Running the Application in local

Clone the project into local

Install all the npm packages. Go into the project folder and type the following command to install all npm packages.

```bash
npm install
```

To run the application locally use:

```bash
npm run dev
```

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
