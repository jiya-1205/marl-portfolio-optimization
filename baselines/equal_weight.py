"""Equal-Weight (1/N) benchmark strategy."""

import numpy as np
import pandas as pd
from environments.market_frictions import MarketFrictionConfig, MarketFrictionModel


class EqualWeightStrategy:
    """Rebalances portfolio to equal weights (1/N) periodically under market frictions."""

    def __init__(
        self,
        n_assets: int,
        rebalance_interval: int = 1,
        friction_config: MarketFrictionConfig = None,
    ):
        self.n_assets = n_assets
        self.rebalance_interval = rebalance_interval
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

        for t in range(n_timesteps):
            curr_prices = prices[t]
            curr_vols = volumes[t]

            # Mark to market at start of step
            port_val = cash + np.sum(holdings * curr_prices)

            # Check if rebalancing day
            if t % self.rebalance_interval == 0:
                target_per_asset = (port_val * 0.99) / self.n_assets  # 1% cash buffer
                target_shares = target_per_asset / (curr_prices + 1e-9)
                delta_shares = target_shares - holdings

                sell_idx = np.where(delta_shares < 0)[0]
                buy_idx = np.where(delta_shares > 0)[0]

                for i in np.concatenate([sell_idx, buy_idx]):
                    sh_to_trade = delta_shares[i]
                    if abs(sh_to_trade) < 1e-4:
                        continue
                    res = self.friction_model.calculate_execution(
                        curr_prices[i], sh_to_trade, curr_vols[i]
                    )
                    sh = res["executed_shares"]
                    eff_p = res["effective_price"]
                    cash += -(sh * eff_p) - res["fee_cost"]
                    holdings[i] += sh
                    costs_incurred[t] += res["total_cost"]

            portfolio_values[t] = cash + np.sum(holdings * curr_prices)

        daily_returns = np.zeros(n_timesteps)
        daily_returns[1:] = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]

        return pd.DataFrame({
            "portfolio_value": portfolio_values,
            "daily_return": daily_returns,
            "costs": costs_incurred,
        })
