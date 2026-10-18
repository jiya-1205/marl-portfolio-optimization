"""Data acquisition module for downloading and caching multi-asset OHLCV data."""

import os
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / "cache"
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
DEFAULT_BENCHMARK = "SPY"


def generate_synthetic_data(
    tickers: List[str],
    start_date: str = "2018-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generates synthetic correlated geometric Brownian motion data with realistic market regimes."""
    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")  # Business days
    n_days = len(date_range)
    all_tickers = list(dict.fromkeys(tickers + [DEFAULT_BENCHMARK]))
    n_assets = len(all_tickers)

    # Base realistic parameters
    initial_prices = {
        "AAPL": 45.0,
        "MSFT": 85.0,
        "GOOGL": 52.0,
        "AMZN": 59.0,
        "JPM": 105.0,
        "SPY": 270.0,
    }
    
    # Generate correlation matrix
    corr = np.full((n_assets, n_assets), 0.45)
    np.fill_diagonal(corr, 1.0)
    # Market benchmark (SPY) has higher correlation with all assets
    if DEFAULT_BENCHMARK in all_tickers:
        spy_idx = all_tickers.index(DEFAULT_BENCHMARK)
        corr[spy_idx, :] = 0.65
        corr[:, spy_idx] = 0.65
        corr[spy_idx, spy_idx] = 1.0
        
    chol = np.linalg.cholesky(corr)

    # Regime shifts: Bull (2018-2019), Pandemic Shock & Recovery (2020), Bull (2021), Bear (2022), Tech Rally (2023-2024)
    # Construct annualized drift and volatility series
    drifts = np.zeros((n_days, n_assets))
    vols = np.zeros((n_days, n_assets))

    for i, dt in enumerate(date_range):
        year = dt.year
        if year in [2018, 2019]:
            mu, sigma = 0.14, 0.18
        elif year == 2020:
            # High volatility year
            mu, sigma = 0.16, 0.32
        elif year == 2021:
            mu, sigma = 0.22, 0.16
        elif year == 2022:
            # Bear regime
            mu, sigma = -0.15, 0.28
        else: # 2023, 2024
            mu, sigma = 0.20, 0.19
            
        drifts[i, :] = mu / 252.0
        vols[i, :] = sigma / np.sqrt(252.0)

    # Standard normal innovations
    raw_normals = np.random.normal(0, 1, size=(n_days, n_assets))
    correlated_normals = raw_normals @ chol.T
    daily_returns = drifts + vols * correlated_normals

    datasets = {}
    for idx, ticker in enumerate(all_tickers):
        p0 = initial_prices.get(ticker, 100.0)
        price_series = p0 * np.cumprod(1 + daily_returns[:, idx])
        
        # High, Low, Open, Volume creation
        high_mult = 1.0 + np.abs(np.random.normal(0.006, 0.003, n_days))
        low_mult = 1.0 - np.abs(np.random.normal(0.006, 0.003, n_days))
        open_mult = 1.0 + np.random.normal(0.0, 0.003, n_days)
        
        close = price_series
        high = close * high_mult
        low = close * low_mult
        open_p = close * open_mult
        
        # Base daily volume
        base_vol = 2.5e7 if ticker in ["AAPL", "SPY"] else 1.2e7
        volume = np.random.lognormal(mean=np.log(base_vol), sigma=0.4, size=n_days)
        
        df = pd.DataFrame(
            {
                "Date": date_range,
                "Open": open_p,
                "High": high,
                "Low": low,
                "Close": close,
                "Adj Close": close,
                "Volume": volume,
            }
        ).set_index("Date")
        datasets[ticker] = df

    return datasets


def fetch_and_cache_data(
    tickers: Optional[List[str]] = None,
    benchmark: str = DEFAULT_BENCHMARK,
    start_date: str = "2018-01-01",
    end_date: str = "2024-12-31",
    force_download: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetches OHLCV data from yfinance with disk cache and robust fallback."""
    if tickers is None:
        tickers = DEFAULT_TICKERS

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_tickers = list(dict.fromkeys(tickers + [benchmark]))
    data_dict: dict[str, pd.DataFrame] = {}

    missing_tickers = []
    for ticker in all_tickers:
        cache_path = CACHE_DIR / f"{ticker}.csv"
        if cache_path.exists() and not force_download:
            try:
                df = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
                if len(df) > 50:
                    data_dict[ticker] = df
                    continue
            except Exception:
                pass
        missing_tickers.append(ticker)

    if missing_tickers:
        try:
            import yfinance as yf
            print(f"Downloading historical market data for: {missing_tickers}")
            downloaded = yf.download(
                missing_tickers,
                start=start_date,
                end=end_date,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
            )
            
            for ticker in missing_tickers:
                if len(missing_tickers) == 1:
                    df = downloaded.copy()
                else:
                    if ticker in downloaded.columns.levels[0]:
                        df = downloaded[ticker].dropna(how="all").copy()
                    else:
                        df = pd.DataFrame()
                        
                if not df.empty and len(df) > 50:
                    df.index.name = "Date"
                    data_dict[ticker] = df
                    df.to_csv(CACHE_DIR / f"{ticker}.csv")
                else:
                    print(f"Warning: No valid data for {ticker}, generating fallback.")
                    synth = generate_synthetic_data([ticker], start_date, end_date)
                    data_dict[ticker] = synth[ticker]
                    synth[ticker].to_csv(CACHE_DIR / f"{ticker}.csv")
        except Exception as e:
            print(f"yfinance download failed: {e}. Generating synthetic fallback dataset.")
            synth = generate_synthetic_data(all_tickers, start_date, end_date)
            for ticker in all_tickers:
                if ticker not in data_dict:
                    data_dict[ticker] = synth[ticker]
                    synth[ticker].to_csv(CACHE_DIR / f"{ticker}.csv")

    # Final verification & alignment
    common_index = None
    for ticker in all_tickers:
        df = data_dict[ticker]
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)

    aligned_dict = {}
    for ticker in all_tickers:
        aligned_dict[ticker] = data_dict[ticker].loc[common_index].sort_index()

    print(f"Data ready for {len(aligned_dict)} symbols with {len(common_index)} trading days.")
    return aligned_dict


if __name__ == "__main__":
    fetch_and_cache_data()
