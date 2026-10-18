"""Gymnasium environment for Algorithmic Trade Execution (Almgren-Chriss framework)."""

from typing import Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from environments.market_frictions import MarketFrictionConfig, MarketFrictionModel


class ExecutionEnv(gym.Env):
    """Gymnasium environment for learning optimal order slicing policies.
    
    The agent receives a target block of shares Q_0 to trade over T sub-steps,
    balancing market impact/slippage against timing risk (price volatility).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        target_shares: float = 10_000.0,
        n_substeps: int = 10,
        arrival_price: float = 100.0,
        market_volume: float = 500_000.0,
        volatility: float = 0.25,
        friction_config: Optional[MarketFrictionConfig] = None,
        risk_aversion: float = 1e-4,
    ):
        super().__init__()
        self.default_target_shares = float(target_shares)
        self.n_substeps = n_substeps
        self.default_arrival_price = float(arrival_price)
        self.default_market_volume = float(market_volume)
        self.default_volatility = float(volatility)
        self.risk_aversion = risk_aversion

        self.friction_model = MarketFrictionModel(friction_config)

        # Action: fraction of remaining inventory to execute in this sub-step [0.0, 1.0]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation:
        # [remaining_shares_ratio, remaining_time_ratio, current_price_ratio, volatility, volume_ratio]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        # State variables
        self.current_substep = 0
        self.target_shares = self.default_target_shares
        self.remaining_shares = self.target_shares
        self.arrival_price = self.default_arrival_price
        self.current_price = self.arrival_price
        self.market_volume = self.default_market_volume
        self.volatility = self.default_volatility
        self.total_executed_shares = 0.0
        self.total_cost_incurred = 0.0
        self.execution_prices = []
        self.slice_sizes = []

    def _get_observation(self) -> np.ndarray:
        rem_shares_ratio = (
            self.remaining_shares / (abs(self.target_shares) + 1e-9)
        )
        rem_time_ratio = (
            (self.n_substeps - self.current_substep) / float(self.n_substeps)
        )
        price_ratio = (self.current_price - self.arrival_price) / (
            self.arrival_price + 1e-9
        )
        volume_ratio = 1.0  # Normalized sub-interval volume
        obs = np.array(
            [
                rem_shares_ratio,
                rem_time_ratio,
                price_ratio,
                self.volatility,
                volume_ratio,
            ],
            dtype=np.float32,
        )
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        options = options or {}
        self.target_shares = float(
            options.get("target_shares", self.default_target_shares)
        )
        self.arrival_price = float(
            options.get("arrival_price", self.default_arrival_price)
        )
        self.market_volume = float(
            options.get("market_volume", self.default_market_volume)
        )
        self.volatility = float(
            options.get("volatility", self.default_volatility)
        )

        self.current_substep = 0
        self.remaining_shares = self.target_shares
        self.current_price = self.arrival_price
        self.total_executed_shares = 0.0
        self.total_cost_incurred = 0.0
        self.execution_prices = []
        self.slice_sizes = []

        return self._get_observation(), {"target_shares": self.target_shares}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action_fraction = float(np.clip(action[0], 0.0, 1.0))
        is_last_step = self.current_substep >= (self.n_substeps - 1)

        # On the final step, force full liquidation/completion of order
        if is_last_step:
            shares_to_execute = self.remaining_shares
        else:
            shares_to_execute = self.remaining_shares * action_fraction

        # Sub-interval market volume
        sub_volume = self.market_volume / self.n_substeps

        exec_info = self.friction_model.calculate_execution(
            arrival_price=self.current_price,
            shares=shares_to_execute,
            market_volume=sub_volume,
            volatility=self.volatility,
        )

        executed_shares = exec_info["executed_shares"]
        effective_price = exec_info["effective_price"]
        shortfall = exec_info["implementation_shortfall"]

        self.remaining_shares -= executed_shares
        self.total_executed_shares += executed_shares
        self.total_cost_incurred += exec_info["total_cost"]
        self.execution_prices.append(effective_price)
        self.slice_sizes.append(executed_shares)

        # Advance price with stochastic drift + permanent impact
        daily_dt = 1.0 / (252.0 * self.n_substeps)
        random_shock = np.random.normal(0, 1)
        perm_impact = (
            self.friction_model.config.permanent_impact_gamma
            * (executed_shares / max(sub_volume, 1.0))
            * self.arrival_price
        )
        price_drift = (
            self.current_price * self.volatility * np.sqrt(daily_dt) * random_shock
            + perm_impact
        )
        self.current_price += price_drift

        # Reward:
        # Negative implementation shortfall normalized + inventory holding risk penalty
        direction = np.sign(self.target_shares)
        # Cost relative to benchmark arrival price
        trade_shortfall_pct = (
            direction * (effective_price - self.arrival_price) / self.arrival_price
        )
        # Inventory variance risk (Almgren-Chriss penalty for holding unexecuted risk)
        inventory_risk = (
            self.risk_aversion
            * (self.volatility ** 2)
            * ((self.remaining_shares / (abs(self.target_shares) + 1e-9)) ** 2)
        )

        reward = -trade_shortfall_pct - inventory_risk - (exec_info["fee_cost"] / (abs(self.target_shares * self.arrival_price) + 1e-9))

        self.current_substep += 1
        terminated = self.current_substep >= self.n_substeps or abs(self.remaining_shares) < 1e-4
        truncated = False

        obs = self._get_observation()
        info = {
            "substep": self.current_substep,
            "executed_shares": executed_shares,
            "effective_price": effective_price,
            "remaining_shares": self.remaining_shares,
            "total_costs": self.total_cost_incurred,
            "avg_execution_price": np.mean(self.execution_prices) if self.execution_prices else self.arrival_price,
        }

        return obs, float(reward), terminated, truncated, info
