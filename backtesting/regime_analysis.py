"""Market regime segmentation and robustness analysis across market conditions."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from backtesting.metrics import calculate_portfolio_metrics


class MarketRegimeDetector:
    """Classifies market history into Bullish, Bearish, and High-Volatility regimes."""

    def __init__(
        self,
        benchmark_prices: np.ndarray,
        dates: pd.DatetimeIndex = None,
        vol_window: int = 20,
        vol_threshold_quantile: float = 0.75,
    ):
        self.prices = pd.Series(benchmark_prices, index=dates)
        self.vol_window = vol_window
        self.vol_threshold_quantile = vol_threshold_quantile

    def classify_regimes(self) -> pd.Series:
        """Assigns each time step to 'Bullish', 'Bearish', or 'High-Volatility'."""
        sma_50 = self.prices.rolling(50).mean()
        sma_200 = self.prices.rolling(200).mean().bfill()
        sma_50 = sma_50.bfill()

        # Rolling annualized volatility
        daily_ret = self.prices.pct_change().fillna(0.0)
        vol_20 = daily_ret.rolling(self.vol_window).std() * np.sqrt(252)
        vol_threshold = vol_20.quantile(self.vol_threshold_quantile)

        regimes = pd.Series(index=self.prices.index, dtype=str)

        for dt in self.prices.index:
            is_high_vol = vol_20.loc[dt] >= vol_threshold
            is_bull = sma_50.loc[dt] >= sma_200.loc[dt]

            if is_high_vol:
                regimes.loc[dt] = "High-Volatility"
            elif is_bull:
                regimes.loc[dt] = "Bullish"
            else:
                regimes.loc[dt] = "Bearish"

        return regimes


def evaluate_regimes(
    strategy_results: Dict[str, pd.DataFrame],
    regimes: pd.Series,
) -> pd.DataFrame:
    """Calculates performance breakdown for each strategy across market regimes."""
    records = []

    for strategy_name, df in strategy_results.items():
        ret_series = pd.Series(df["daily_return"].values, index=regimes.index[:len(df)])
        
        for regime in ["Bullish", "Bearish", "High-Volatility"]:
            mask = regimes[:len(df)] == regime
            regime_returns = ret_series[mask].values
            
            if len(regime_returns) > 5:
                # Reconstruct localized value trajectory
                val_traj = 100_000.0 * np.cumprod(1.0 + regime_returns)
                metrics = calculate_portfolio_metrics(val_traj, regime_returns)
                records.append({
                    "Strategy": strategy_name,
                    "Regime": regime,
                    "Days_In_Regime": int(np.sum(mask)),
                    "Cumulative_Return_%": metrics["cumulative_return_pct"],
                    "Annualized_Return_%": metrics["annualized_return_cagr_pct"],
                    "Volatility_%": metrics["annualized_volatility_pct"],
                    "Sharpe_Ratio": metrics["sharpe_ratio"],
                    "Max_Drawdown_%": metrics["max_drawdown_pct"],
                })

    return pd.DataFrame(records)
