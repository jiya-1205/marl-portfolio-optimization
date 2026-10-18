"""Feature engineering pipeline for financial time series."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes standard Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Computes MACD line, signal line, and histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Computes Bollinger Middle, Upper, Lower bands, and %B."""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    pct_b = (series - lower) / ((upper - lower) + 1e-9)
    bandwidth = (upper - lower) / (middle + 1e-9)
    return upper, lower, pct_b.fillna(0.5), bandwidth.fillna(0.0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Computes Average True Range (ATR)."""
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.bfill()


def extract_features_for_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts standardized technical and price features for a single ticker."""
    features = pd.DataFrame(index=df.index)
    close = df["Close"]
    volume = df["Volume"]

    # Returns
    features["return_1d"] = close.pct_change(1).fillna(0.0)
    features["return_5d"] = close.pct_change(5).fillna(0.0)
    features["return_21d"] = close.pct_change(21).fillna(0.0)

    # Trend / Moving averages normalized by close
    sma_10 = close.rolling(10).mean()
    sma_30 = close.rolling(30).mean()
    ema_20 = close.ewm(span=20, adjust=False).mean()
    features["sma_10_ratio"] = (close / (sma_10 + 1e-9)) - 1.0
    features["sma_30_ratio"] = (close / (sma_30 + 1e-9)) - 1.0
    features["ema_20_ratio"] = (close / (ema_20 + 1e-9)) - 1.0

    # RSI (scaled to [-1, 1])
    rsi = compute_rsi(close, 14)
    features["rsi_14"] = (rsi - 50.0) / 50.0

    # MACD normalized by close
    macd, signal, hist = compute_macd(close)
    features["macd_norm"] = macd / (close + 1e-9)
    features["macd_sig_norm"] = signal / (close + 1e-9)
    features["macd_hist_norm"] = hist / (close + 1e-9)

    # Bollinger Bands
    upper, lower, pct_b, bandwidth = compute_bollinger_bands(close, 20)
    features["bb_pct_b"] = pct_b.clip(-1.0, 2.0)
    features["bb_bandwidth"] = bandwidth.clip(0.0, 1.0)

    # Volatility & ATR normalized
    vol_20 = features["return_1d"].rolling(20).std() * np.sqrt(252)
    features["volatility_20d"] = vol_20.fillna(0.20)
    atr = compute_atr(df, 14)
    features["atr_ratio"] = (atr / (close + 1e-9)).fillna(0.01)

    # Volume ratio
    vol_ma_20 = volume.rolling(20).mean() + 1e-9
    features["volume_ratio"] = (volume / vol_ma_20).clip(0.1, 5.0) - 1.0

    # Forward-fill and clip outliers
    features = features.ffill().bfill().clip(-5.0, 5.0)
    return features


class FeaturePipeline:
    """End-to-end multi-asset feature pipeline."""

    def __init__(self, tickers: List[str], benchmark: str = "SPY"):
        self.tickers = tickers
        self.benchmark = benchmark
        self.feature_names: List[str] = []

    def fit_transform(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
        """Processes raw OHLCV data into aligned arrays for RL environments.

        Returns:
            dates: DatetimeIndex of trading days.
            prices: np.ndarray shape (T, N) of close prices.
            volumes: np.ndarray shape (T, N) of trading volume.
            feature_tensor: np.ndarray shape (T, N, F) of asset features.
        """
        # Ensure common index
        common_idx = data_dict[self.benchmark].index
        for t in self.tickers:
            common_idx = common_idx.intersection(data_dict[t].index)
        common_idx = common_idx.sort_values()

        # Extract benchmark macro features
        bench_df = data_dict[self.benchmark].loc[common_idx]
        bench_ret1 = bench_df["Close"].pct_change(1).fillna(0.0).values
        bench_vol20 = (bench_df["Close"].pct_change().rolling(20).std() * np.sqrt(252)).bfill().values

        feature_list = []
        prices = []
        volumes = []

        for ticker in self.tickers:
            df = data_dict[ticker].loc[common_idx]
            feats = extract_features_for_ticker(df)
            
            # Add macro features to each asset's feature vector
            feats["market_ret_1d"] = bench_ret1
            feats["market_vol_20d"] = bench_vol20
            
            self.feature_names = list(feats.columns)
            feature_list.append(feats.values)
            prices.append(df["Close"].values)
            volumes.append(df["Volume"].values)

        # prices: (T, N), volumes: (T, N)
        prices_arr = np.column_stack(prices)
        volumes_arr = np.column_stack(volumes)
        # feature_tensor: (T, N, F)
        feature_tensor = np.stack(feature_list, axis=1)

        return common_idx, prices_arr, volumes_arr, feature_tensor
