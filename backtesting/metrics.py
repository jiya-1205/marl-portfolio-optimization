"""Quantitative financial performance metrics and portfolio analytics."""

from typing import Dict
import numpy as np
import pandas as pd


def calculate_portfolio_metrics(
    portfolio_values: np.ndarray,
    daily_returns: np.ndarray,
    costs: np.ndarray = None,
    risk_free_rate: float = 0.02,
    trading_days_per_year: int = 252,
) -> Dict[str, float]:
    """Computes comprehensive quantitative performance and risk metrics."""
    values = np.asarray(portfolio_values, dtype=np.float64)
    returns = np.asarray(daily_returns, dtype=np.float64)
    daily_rf = risk_free_rate / trading_days_per_year
    n_days = len(values)

    # 1. Cumulative & Annualized Return (CAGR)
    cum_return = (values[-1] - values[0]) / (values[0] + 1e-9)
    n_years = max(n_days / trading_days_per_year, 1.0 / trading_days_per_year)
    cagr = (values[-1] / values[0]) ** (1.0 / n_years) - 1.0

    # 2. Volatility (Annualized)
    volatility = np.std(returns) * np.sqrt(trading_days_per_year)

    # 3. Sharpe Ratio
    excess_returns = returns - daily_rf
    sharpe = (
        (np.mean(excess_returns) / (np.std(returns) + 1e-9))
        * np.sqrt(trading_days_per_year)
    )

    # 4. Downside Deviation & Sortino Ratio
    downside_returns = returns[returns < daily_rf] - daily_rf
    if len(downside_returns) > 0:
        downside_std = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(trading_days_per_year)
        sortino = (np.mean(excess_returns) * trading_days_per_year) / (downside_std + 1e-9)
    else:
        sortino = 0.0

    # 5. Maximum Drawdown (MDD)
    peaks = np.maximum.accumulate(values)
    drawdowns = (peaks - values) / (peaks + 1e-9)
    max_drawdown = float(np.max(drawdowns))

    # 6. Calmar Ratio
    calmar = cagr / (max_drawdown + 1e-9) if max_drawdown > 0 else 0.0

    # 7. Total Transaction & Slippage Costs
    total_costs = float(np.sum(costs)) if costs is not None else 0.0
    cost_drag_pct = (total_costs / values[0]) * 100.0

    return {
        "cumulative_return_pct": float(cum_return * 100.0),
        "annualized_return_cagr_pct": float(cagr * 100.0),
        "annualized_volatility_pct": float(volatility * 100.0),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown_pct": float(max_drawdown * 100.0),
        "calmar_ratio": float(calmar),
        "total_friction_costs": total_costs,
        "cost_drag_pct": float(cost_drag_pct),
    }


def compute_drawdown_series(portfolio_values: np.ndarray) -> np.ndarray:
    """Computes time series of percentage drawdowns from the running peak."""
    peaks = np.maximum.accumulate(portfolio_values)
    return (portfolio_values - peaks) / (peaks + 1e-9)
