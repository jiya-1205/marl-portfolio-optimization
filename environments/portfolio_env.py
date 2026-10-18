"""Gymnasium environment for Portfolio Optimization."""

from typing import Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from environments.market_frictions import MarketFrictionConfig, MarketFrictionModel


class PortfolioOptimizationEnv(gym.Env):
    """Gymnasium environment for multi-asset portfolio allocation under market frictions."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        features: np.ndarray,
        asset_names: list,
        initial_capital: float = 100_000.0,
        rebalance_interval: int = 1,
        friction_config: Optional[MarketFrictionConfig] = None,
        risk_penalty: float = 0.5,
        turnover_penalty: float = 0.002,
        drawdown_penalty: float = 0.5,
    ):
        super().__init__()
        self.prices = prices  # shape (T, N)
        self.volumes = volumes  # shape (T, N)
        self.features = features  # shape (T, N, F)
        self.asset_names = asset_names
        self.n_timesteps, self.n_assets, self.n_features = features.shape

        self.initial_capital = float(initial_capital)
        self.rebalance_interval = rebalance_interval
        self.risk_penalty = risk_penalty
        self.turnover_penalty = turnover_penalty
        self.drawdown_penalty = drawdown_penalty

        self.friction_model = MarketFrictionModel(friction_config)

        # Action: continuous logits for N assets + 1 cash position
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.n_assets + 1,), dtype=np.float32
        )

        # Observation: flattened features (N * F) + current asset weights (N) + cash weight (1) + drawdown (1)
        obs_dim = (self.n_assets * self.n_features) + self.n_assets + 1 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Tracking state
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.holdings = np.zeros(self.n_assets, dtype=np.float64)  # shares per asset
        self.weights = np.zeros(self.n_assets + 1, dtype=np.float64)
        self.weights[-1] = 1.0  # initial 100% cash
        self.peak_value = self.initial_capital
        self.history = []

    def _softmax_action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """Converts raw continuous action into valid probability distribution over simplex."""
        # Stabilize exp
        exp_a = np.exp(action - np.max(action))
        weights = exp_a / (np.sum(exp_a) + 1e-9)
        return weights.astype(np.float64)

    def _get_observation(self) -> np.ndarray:
        current_feat = self.features[self.current_step].flatten()
        current_drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-9)
        obs = np.concatenate([
            current_feat,
            self.weights[:self.n_assets],
            [self.weights[-1]],
            [current_drawdown]
        ])
        return obs.astype(np.float32)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.holdings = np.zeros(self.n_assets, dtype=np.float64)
        self.weights = np.zeros(self.n_assets + 1, dtype=np.float64)
        self.weights[-1] = 1.0
        self.peak_value = self.initial_capital
        self.history = [{
            "step": 0,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "weights": self.weights.copy(),
            "costs": 0.0,
            "return": 0.0,
        }]
        return self._get_observation(), {"portfolio_value": self.portfolio_value}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        target_weights = self._softmax_action_to_weights(action)
        curr_prices = self.prices[self.current_step]
        curr_vols = self.volumes[self.current_step]

        # 1. Rebalancing execution
        total_costs = 0.0
        total_slippage = 0.0
        total_fees = 0.0
        turnover = float(np.sum(np.abs(target_weights - self.weights)))

        # Target dollar values
        target_asset_values = target_weights[:self.n_assets] * self.portfolio_value
        target_shares = target_asset_values / (curr_prices + 1e-9)
        delta_shares = target_shares - self.holdings

        # Execute sells first to free cash, then buys
        sell_indices = np.where(delta_shares < 0)[0]
        buy_indices = np.where(delta_shares > 0)[0]

        for i in np.concatenate([sell_indices, buy_indices]):
            shares_to_trade = delta_shares[i]
            if abs(shares_to_trade) < 1e-4:
                continue

            exec_info = self.friction_model.calculate_execution(
                arrival_price=curr_prices[i],
                shares=shares_to_trade,
                market_volume=curr_vols[i],
            )
            exec_shares = exec_info["executed_shares"]
            eff_price = exec_info["effective_price"]

            # Cash impact: buying uses cash, selling adds cash
            cash_change = -(exec_shares * eff_price) - exec_info["fee_cost"]
            self.cash += cash_change
            self.holdings[i] += exec_shares

            total_costs += exec_info["total_cost"]
            total_slippage += exec_info["slippage_cost"]
            total_fees += exec_info["fee_cost"]

        # Ensure non-negative cash through emergency liquidation if necessary
        if self.cash < 0:
            for i in range(self.n_assets):
                if self.holdings[i] > 0 and self.cash < 0:
                    needed = -self.cash
                    shares_liq = min(self.holdings[i], needed / curr_prices[i])
                    self.holdings[i] -= shares_liq
                    self.cash += shares_liq * curr_prices[i] * (1.0 - 0.002)

        # 2. Advance time step
        self.current_step += 1
        terminated = self.current_step >= (self.n_timesteps - 1)
        truncated = False

        next_prices = self.prices[self.current_step]
        asset_values = self.holdings * next_prices
        new_portfolio_value = float(self.cash + np.sum(asset_values))

        # Actual step return
        step_return = (new_portfolio_value - self.portfolio_value) / (self.portfolio_value + 1e-9)
        self.portfolio_value = new_portfolio_value
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value

        # Update current weights
        if self.portfolio_value > 0:
            self.weights[:self.n_assets] = asset_values / self.portfolio_value
            self.weights[-1] = max(0.0, self.cash / self.portfolio_value)
            # Normalize to sum exactly to 1
            self.weights /= np.sum(self.weights)

        # 3. Reward formulation
        # Log return penalized by portfolio variance, turnover, and drawdown
        drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-9)
        variance_penalty = (step_return ** 2) if step_return < 0 else 0.0
        
        reward = (
            step_return
            - (self.risk_penalty * variance_penalty)
            - (self.turnover_penalty * turnover)
            - (self.drawdown_penalty * (drawdown ** 2))
        )

        obs = self._get_observation()
        info = {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "step_return": step_return,
            "drawdown": drawdown,
            "turnover": turnover,
            "total_costs": total_costs,
            "slippage_cost": total_slippage,
            "fee_cost": total_fees,
            "weights": self.weights.copy(),
        }
        self.history.append(info)

        return obs, float(reward), terminated, truncated, info
