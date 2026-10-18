"""Markowitz Mean-Variance Portfolio Optimization benchmark."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from environments.market_frictions import MarketFrictionConfig, MarketFrictionModel


class MarkowitzOptimizationStrategy:
    """Mean-Variance Modern Portfolio Theory (MPT) strategy.
    
    Optimizes portfolio weights to maximize rolling Sharpe ratio subject to no-short-sale constraints.
    """

    def __init__(
        self,
        n_assets: int,
        lookback_window: int = 60,
        rebalance_interval: int = 5,
        risk_free_rate: float = 0.02,
        friction_config: MarketFrictionConfig = None,
    ):
        self.n_assets = n_assets
        self.lookback_window = lookback_window
        self.rebalance_interval = rebalance_interval
        self.daily_rf = risk_free_rate / 252.0
        self.friction_model = MarketFrictionModel(friction_config)

    def optimize_weights(self, returns_window: np.ndarray) -> np.ndarray:
        """Solves quadratic programming problem to find tangency portfolio (max Sharpe)."""
        # Ledoit-Wolf shrinkage / ridge regularization on covariance
        cov_matrix = np.cov(returns_window, rowvar=False)
        cov_matrix += 1e-4 * np.eye(self.n_assets)
        mean_returns = np.mean(returns_window, axis=0)

        # Objective: Negative Sharpe ratio
        def neg_sharpe(weights):
            p_return = np.sum(mean_returns * weights)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) + 1e-8
            return -(p_return - self.daily_rf) / p_vol

        # Constraints & Bounds: sum(w) = 1.0, 0 <= w_i <= 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = tuple((0.0, 1.0) for _ in range(self.n_assets))
        init_guess = np.ones(self.n_assets) / self.n_assets

        try:
            res = minimize(
                neg_sharpe,
                init_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 100, "ftol": 1e-7},
            )
            if res.success and not np.isnan(res.x).any():
                weights = np.clip(res.x, 0.0, 1.0)
                return weights / np.sum(weights)
        except Exception:
            pass

        return init_guess

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
        weights_history = np.zeros((n_timesteps, self.n_assets))

        # Calculate daily asset returns
        asset_returns = np.zeros_like(prices)
        asset_returns[1:] = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-9)

        # Initial equal weight allocation during warmup
        warmup = min(self.lookback_window, n_timesteps - 1)
        init_weights = np.ones(self.n_assets) / self.n_assets

        # Day 0 initialization
        p0 = prices[0]
        v0 = volumes[0]
        target_dollars = (initial_capital * 0.99) * init_weights
        for i in range(self.n_assets):
            sh = target_dollars[i] / p0[i]
            res = self.friction_model.calculate_execution(p0[i], sh, v0[i])
            cash -= (res["executed_shares"] * res["effective_price"]) + res["fee_cost"]
            holdings[i] = res["executed_shares"]
            costs_incurred[0] += res["total_cost"]

        portfolio_values[0] = cash + np.sum(holdings * p0)
        weights_history[0] = init_weights

        target_weights = init_weights.copy()

        for t in range(1, n_timesteps):
            curr_prices = prices[t]
            curr_vols = volumes[t]
            port_val = cash + np.sum(holdings * curr_prices)

            # Rebalance check
            if t >= warmup and (t % self.rebalance_interval == 0):
                window = asset_returns[t - self.lookback_window : t]
                target_weights = self.optimize_weights(window)

                target_dollars = (port_val * 0.99) * target_weights
                target_shares = target_dollars / (curr_prices + 1e-9)
                delta_shares = target_shares - holdings

                sell_idx = np.where(delta_shares < 0)[0]
                buy_idx = np.where(delta_shares > 0)[0]

                for i in np.concatenate([sell_idx, buy_idx]):
                    sh_trade = delta_shares[i]
                    if abs(sh_trade) < 1e-4:
                        continue
                    res = self.friction_model.calculate_execution(
                        curr_prices[i], sh_trade, curr_vols[i]
                    )
                    sh = res["executed_shares"]
                    eff_p = res["effective_price"]
                    cash += -(sh * eff_p) - res["fee_cost"]
                    holdings[i] += sh
                    costs_incurred[t] += res["total_cost"]

            portfolio_values[t] = cash + np.sum(holdings * curr_prices)
            weights_history[t] = target_weights

        daily_returns = np.zeros(n_timesteps)
        daily_returns[1:] = (portfolio_values[1:] - portfolio_values[:-1]) / (portfolio_values[:-1] + 1e-9)

        df = pd.DataFrame({
            "portfolio_value": portfolio_values,
            "daily_return": daily_returns,
            "costs": costs_incurred,
        })
        for i in range(self.n_assets):
            df[f"weight_asset_{i}"] = weights_history[:, i]

        return df
