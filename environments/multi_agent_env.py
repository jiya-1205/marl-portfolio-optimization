"""PettingZoo Multi-Agent Environment for Portfolio Optimization and Algorithmic Execution."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from environments.market_frictions import MarketFrictionConfig, MarketFrictionModel


class MultiAgentTradingEnv(ParallelEnv):
    """Multi-Agent PettingZoo ParallelEnv coordinating a Portfolio Agent and an Execution Agent.
    
    Agents:
      - 'portfolio_agent': Strategic asset allocation agent (simplex action over assets + cash).
      - 'execution_agent': Tactical order execution agent (slicing & slippage minimization).
    """

    metadata = {"render_modes": ["human"], "name": "multi_agent_trading_v1"}

    def __init__(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        features: np.ndarray,
        asset_names: List[str],
        initial_capital: float = 100_000.0,
        n_execution_slices: int = 5,
        friction_config: Optional[MarketFrictionConfig] = None,
        risk_penalty: float = 0.5,
    ):
        super().__init__()
        self.prices = prices  # (T, N)
        self.volumes = volumes  # (T, N)
        self.features = features  # (T, N, F)
        self.asset_names = asset_names
        self.n_timesteps, self.n_assets, self.n_features = features.shape
        self.initial_capital = float(initial_capital)
        self.n_slices = n_execution_slices
        self.risk_penalty = risk_penalty

        self.friction_model = MarketFrictionModel(friction_config)

        self.possible_agents = ["portfolio_agent", "execution_agent"]
        self.agents = self.possible_agents.copy()

        # Observation spaces
        # Portfolio agent: features (N*F) + weights (N) + cash (1) + drawdown (1)
        port_obs_dim = (self.n_assets * self.n_features) + self.n_assets + 1 + 1
        # Execution agent: total order size, current price level, market vol, time slice, execution urgency
        exec_obs_dim = self.n_assets + self.n_assets + 2  # (delta_shares_norm, prices_norm, time_slice, vol)

        self.observation_spaces = {
            "portfolio_agent": spaces.Box(
                low=-np.inf, high=np.inf, shape=(port_obs_dim,), dtype=np.float32
            ),
            "execution_agent": spaces.Box(
                low=-np.inf, high=np.inf, shape=(exec_obs_dim,), dtype=np.float32
            ),
        }

        # Action spaces
        # Portfolio: target weight logits for N assets + cash
        # Execution: urgency multiplier / slicing parameter for each asset [0.1, 1.0]
        self.action_spaces = {
            "portfolio_agent": spaces.Box(
                low=-5.0, high=5.0, shape=(self.n_assets + 1,), dtype=np.float32
            ),
            "execution_agent": spaces.Box(
                low=0.1, high=1.0, shape=(self.n_assets,), dtype=np.float32
            ),
        }

        # Simulation state
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.holdings = np.zeros(self.n_assets, dtype=np.float64)
        self.weights = np.zeros(self.n_assets + 1, dtype=np.float64)
        self.weights[-1] = 1.0
        self.peak_value = self.initial_capital
        self.history = []

    def _get_portfolio_obs(self) -> np.ndarray:
        current_feat = self.features[self.current_step].flatten()
        current_drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-9)
        return np.concatenate([
            current_feat,
            self.weights[:self.n_assets],
            [self.weights[-1]],
            [current_drawdown]
        ]).astype(np.float32)

    def _get_execution_obs(self, delta_shares: np.ndarray) -> np.ndarray:
        curr_prices = self.prices[self.current_step]
        # Normalized shares relative to portfolio value
        shares_norm = (delta_shares * curr_prices) / (self.portfolio_value + 1e-9)
        prices_norm = curr_prices / (np.mean(curr_prices) + 1e-9)
        time_step_norm = self.current_step / float(self.n_timesteps)
        avg_vol = float(np.mean(self.volumes[self.current_step]) / 1e6)
        return np.concatenate([
            shares_norm,
            prices_norm,
            [time_step_norm],
            [avg_vol]
        ]).astype(np.float32)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        self.agents = self.possible_agents.copy()
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.holdings = np.zeros(self.n_assets, dtype=np.float64)
        self.weights = np.zeros(self.n_assets + 1, dtype=np.float64)
        self.weights[-1] = 1.0
        self.peak_value = self.initial_capital

        port_obs = self._get_portfolio_obs()
        exec_obs = self._get_execution_obs(np.zeros(self.n_assets))

        observations = {
            "portfolio_agent": port_obs,
            "execution_agent": exec_obs,
        }
        infos = {
            "portfolio_agent": {"portfolio_value": self.portfolio_value},
            "execution_agent": {},
        }
        return observations, infos

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        port_action = actions["portfolio_agent"]
        exec_action = actions["execution_agent"]

        # 1. Strategic allocation
        exp_a = np.exp(port_action - np.max(port_action))
        target_weights = exp_a / (np.sum(exp_a) + 1e-9)

        curr_prices = self.prices[self.current_step]
        curr_vols = self.volumes[self.current_step]

        target_dollars = target_weights[:self.n_assets] * self.portfolio_value
        target_shares = target_dollars / (curr_prices + 1e-9)
        total_delta_shares = target_shares - self.holdings

        # 2. Tactical execution across slices
        total_costs = 0.0
        total_slippage = 0.0
        total_fees = 0.0
        execution_savings = 0.0

        # Execute orders modulated by execution agent's urgency factor
        for i in range(self.n_assets):
            order_shares = total_delta_shares[i]
            if abs(order_shares) < 1e-4:
                continue

            urgency = float(np.clip(exec_action[i], 0.1, 1.0))
            # Slice the order into smaller sub-blocks based on urgency
            # Higher urgency executes faster, lower urgency executes in smaller pieces reducing slippage
            n_suborders = max(1, int(round(self.n_slices * (1.1 - urgency))))
            slice_shares = order_shares / n_suborders
            slice_vol = curr_vols[i] / n_suborders

            for _ in range(n_suborders):
                res = self.friction_model.calculate_execution(
                    arrival_price=curr_prices[i],
                    shares=slice_shares,
                    market_volume=slice_vol,
                )
                exec_sh = res["executed_shares"]
                eff_pr = res["effective_price"]
                
                self.cash += -(exec_sh * eff_pr) - res["fee_cost"]
                self.holdings[i] += exec_sh
                total_costs += res["total_cost"]
                total_slippage += res["slippage_cost"]
                total_fees += res["fee_cost"]

            # Estimate baseline naive execution cost vs sliced execution
            naive_res = self.friction_model.calculate_execution(
                arrival_price=curr_prices[i],
                shares=order_shares,
                market_volume=curr_vols[i],
            )
            savings = naive_res["total_cost"] - (total_slippage + total_fees)
            execution_savings += max(0.0, savings)

        # Ensure cash non-negative
        if self.cash < 0:
            for i in range(self.n_assets):
                if self.holdings[i] > 0 and self.cash < 0:
                    needed = -self.cash
                    liq = min(self.holdings[i], needed / curr_prices[i])
                    self.holdings[i] -= liq
                    self.cash += liq * curr_prices[i] * 0.998

        # 3. Step forward
        self.current_step += 1
        terminated = self.current_step >= (self.n_timesteps - 1)
        truncated = False

        next_prices = self.prices[self.current_step]
        asset_vals = self.holdings * next_prices
        new_portfolio_val = float(self.cash + np.sum(asset_vals))
        step_return = (new_portfolio_val - self.portfolio_value) / (self.portfolio_value + 1e-9)
        self.portfolio_value = new_portfolio_val

        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value

        if self.portfolio_value > 0:
            self.weights[:self.n_assets] = asset_vals / self.portfolio_value
            self.weights[-1] = max(0.0, self.cash / self.portfolio_value)
            self.weights /= np.sum(self.weights)

        drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-9)
        turnover = float(np.sum(np.abs(target_weights - self.weights)))

        # Cooperative rewards
        # Strategic agent: risk-adjusted return minus drawdown and turnover
        port_reward = (
            step_return
            - (self.risk_penalty * (step_return ** 2 if step_return < 0 else 0.0))
            - (0.002 * turnover)
            - (0.5 * (drawdown ** 2))
        )
        # Tactical agent: execution cost savings reward + bonus for portfolio preservation
        exec_reward = (
            (execution_savings / (self.portfolio_value * 0.001 + 1e-9))
            - (total_slippage / (self.portfolio_value * 0.001 + 1e-9))
            + 0.1 * step_return
        )

        rewards = {
            "portfolio_agent": float(port_reward),
            "execution_agent": float(exec_reward),
        }
        terminations = {
            "portfolio_agent": terminated,
            "execution_agent": terminated,
        }
        truncations = {
            "portfolio_agent": truncated,
            "execution_agent": truncated,
        }

        port_obs = self._get_portfolio_obs()
        exec_obs = self._get_execution_obs(total_delta_shares)
        observations = {
            "portfolio_agent": port_obs,
            "execution_agent": exec_obs,
        }

        info_dict = {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "step_return": step_return,
            "drawdown": drawdown,
            "total_costs": total_costs,
            "slippage_cost": total_slippage,
            "fee_cost": total_fees,
            "execution_savings": execution_savings,
            "weights": self.weights.copy(),
        }
        infos = {
            "portfolio_agent": info_dict,
            "execution_agent": info_dict,
        }
        self.history.append(info_dict)

        if terminated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos
