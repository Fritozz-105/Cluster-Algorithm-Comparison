import requests
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List
from datetime import datetime, timedelta
from io import StringIO


def fetch_sp500_tickers() -> List[str]:
    """
    Fetch the current list of S&P 500 tickers from Wikipedia.

    Returns:
        List[str]: A list of S&P 500 stock tickers.
    """
    url: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response: requests.Response = requests.get(url)
    tables: List[pd.DataFrame] = pd.read_html(StringIO(response.text))
    sp500_table: pd.DataFrame = tables[0]  # First table contains the tickers
    return sp500_table["Symbol"].tolist()


def fetch_adj_close_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily adjusted close prices for a list of tickers over a given date range,
    ensuring all tickers have the same data length.

    Args:
        tickers (List[str]): List of stock tickers to fetch data for.
        start_date (str): Start date in the format 'YYYY-MM-DD'.
        end_date (str): End date in the format 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: A DataFrame containing daily adjusted close prices for tickers with complete data.
    """
    all_data: dict[str, np.ndarray] = {}
    common_index: pd.Index = None

    for ticker in tickers:
        try:
            # Fetch adjusted close data
            stock_data: pd.DataFrame = yf.download(ticker, start=start_date, end=end_date)
            if not stock_data.empty and 'Adj Close' in stock_data:
                adj_close = stock_data['Adj Close'].dropna()

                # Determine the common index from the first valid ticker
                if common_index is None:
                    common_index = adj_close.index

                # Include ticker only if its length matches the expected length
                if len(adj_close) == len(common_index) and adj_close.index.equals(common_index):
                    all_data[ticker]= adj_close.to_numpy().flatten()
                else:
                    print(f"{ticker}: Length {len(adj_close)} does not match expected length ({len(common_index)}). Skipping.")
            else:
                print(f"No valid data for {ticker}. Skipping.")
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")

    if not all_data:
        raise ValueError("No valid data retrieved for the provided tickers with matching lengths.")

    print(f"Successfully fetched data for {len(all_data)} tickers with consistent data length.")

    df: pd.DataFrame = pd.DataFrame(all_data, index=common_index)
    return df

def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns for stock price data.

    Args:
        data (pd.DataFrame): DataFrame containing stock prices.

    Returns:
        pd.DataFrame: DataFrame containing daily returns.
    """
    returns: pd.DataFrame = data.pct_change().dropna()
    return returns


def calculate_volatility(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate rolling volatility for stock price data.

    Args:
        data (pd.DataFrame): DataFrame containing stock prices.
        window (int): Rolling window size for volatility calculation.

    Returns:
        pd.DataFrame: DataFrame containing rolling volatility.
    """
    volatility: pd.DataFrame = data.pct_change().rolling(window=window).std().dropna()
    return volatility


def normalize_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize features using z-scores.

    Args:
        data (pd.DataFrame): DataFrame containing features to normalize.

    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    normalized: pd.DataFrame = (data - data.mean()) / data.std()
    return normalized


if __name__ == "__main__":
    """
    # Fetch S&P 500 tickers
    print("Fetching S&P 500 tickers from Wikipedia...")
    sp500_tickers: List[str] = fetch_sp500_tickers()
    print(f"Found {len(sp500_tickers)} tickers.")

    # Define date range (last 5 years)
    end_date: str = datetime.today().strftime('%Y-%m-%d')
    start_date: str = (datetime.today() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

    # Fetch historical adjusted close prices
    print(f"Fetching daily adjusted close data for {len(sp500_tickers)} tickers...")
    stock_data: pd.DataFrame = fetch_adj_close_data(sp500_tickers, start_date, end_date)

    # Save data to CSV
    output_csv: str = "ClusterAlgorithmComparison/backend/sp500_adj_close_data.csv"
    print(f"Saving data to {output_csv}...")
    stock_data.to_csv(output_csv, index=True)
    """

    if __name__ == "__main__":
        # File paths
        input_csv: str = "ClusterAlgorithmComparison/backend/sp500_adj_close_data.csv"
        output_csv: str = "ClusterAlgorithmComparison/backend/sp500_preprocessed_data.csv"

        # Load cleaned stock price data
        print(f"Loading data from {input_csv}...")
        raw_data: pd.DataFrame = pd.read_csv(input_csv, index_col=0, parse_dates=True)

        # Calculate returns and volatility
        print("Calculating daily returns...")
        returns: pd.DataFrame = calculate_returns(raw_data)

        print("Calculating rolling volatility...")
        volatility: pd.DataFrame = calculate_volatility(raw_data)

        # Combine features
        print("Combining features (returns and volatility)...")
        combined_data: pd.DataFrame = pd.concat(
            [returns, volatility], axis=1, keys=["Returns", "Volatility"]
        ).dropna()

        # Flatten multi-level columns to make them unique
        combined_data.columns = [
            f"{col[0]}_{col[1]}" for col in combined_data.columns.to_flat_index()
        ]

        # Normalize features
        print("Normalizing features...")
        normalized_data: pd.DataFrame = normalize_features(combined_data)

        # Save preprocessed data
        print(f"Saving preprocessed data to {output_csv}...")
        normalized_data.to_csv(output_csv, index=True)
        print("Preprocessing complete.")

