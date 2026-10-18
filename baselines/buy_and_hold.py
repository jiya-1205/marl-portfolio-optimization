"""Buy and Hold baseline strategy."""

from typing import List, Tuple
import numpy as np
import pandas as pd

from environments.market_frictions import MarketFrictionConfig, MarketFrictionModel


class BuyAndHoldStrategy:
    """Allocates equally into all assets at step 0 and holds without rebalancing."""

    def __init__(self, n_assets: int, friction_config: MarketFrictionConfig = None):
        self.n_assets = n_assets
        self.friction_model = MarketFrictionModel(friction_config)

    def run(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        initial_capital: float = 100_000.0,
    ) -> pd.DataFrame:
        n_timesteps = len(prices)
        portfolio_values = np.zeros(n_timesteps)
        cash = initial_capital
        holdings = np.zeros(self.n_assets)
        costs_incurred = np.zeros(n_timesteps)

        # Day 0: allocate equal dollar amount into all assets
        p0 = prices[0]
        v0 = volumes[0]
        target_per_asset = initial_capital / self.n_assets

        for i in range(self.n_assets):
            shares = target_per_asset / p0[i]
            res = self.friction_model.calculate_execution(p0[i], shares, v0[i])
            sh = res["executed_shares"]
            eff_p = res["effective_price"]
            cash -= (sh * eff_p) + res["fee_cost"]
            holdings[i] = sh
            costs_incurred[0] += res["total_cost"]

        portfolio_values[0] = cash + np.sum(holdings * p0)

        # Subsequent days: no rebalancing, mark to market
        for t in range(1, n_timesteps):
            portfolio_values[t] = cash + np.sum(holdings * prices[t])

        daily_returns = np.zeros(n_timesteps)
        daily_returns[1:] = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]

        return pd.DataFrame({
            "portfolio_value": portfolio_values,
            "daily_return": daily_returns,
            "costs": costs_incurred,
        })
