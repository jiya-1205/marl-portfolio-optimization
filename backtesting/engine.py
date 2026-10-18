"""Comprehensive backtesting engine running all agents and benchmarks out-of-sample."""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtesting.metrics import calculate_portfolio_metrics, compute_drawdown_series
from backtesting.regime_analysis import MarketRegimeDetector, evaluate_regimes
from baselines.buy_and_hold import BuyAndHoldStrategy
from baselines.equal_weight import EqualWeightStrategy
from baselines.markowitz import MarkowitzOptimizationStrategy
from environments.market_frictions import MarketFrictionConfig, MarketFrictionModel
from environments.multi_agent_env import MultiAgentTradingEnv
from environments.portfolio_env import PortfolioOptimizationEnv


class BacktestEngine:
    """Orchestrates out-of-sample backtesting across models and benchmarks."""

    def __init__(
        self,
        dates: pd.DatetimeIndex,
        prices: np.ndarray,
        volumes: np.ndarray,
        features: np.ndarray,
        asset_names: List[str],
        benchmark_prices: np.ndarray,
        initial_capital: float = 100_000.0,
        friction_config: Optional[MarketFrictionConfig] = None,
    ):
        self.dates = dates
        self.prices = prices
        self.volumes = volumes
        self.features = features
        self.asset_names = asset_names
        self.benchmark_prices = benchmark_prices
        self.initial_capital = initial_capital
        self.friction_config = friction_config or MarketFrictionConfig()

        self.strategy_results: Dict[str, pd.DataFrame] = {}
        self.metrics_summary: Optional[pd.DataFrame] = None
        self.regime_summary: Optional[pd.DataFrame] = None

    def run_baselines(self):
        """Runs Buy-and-Hold, Equal-Weight, and Markowitz benchmarks."""
        n_assets = len(self.asset_names)

        # 1. Buy and Hold
        bnh = BuyAndHoldStrategy(n_assets, self.friction_config)
        self.strategy_results["Buy & Hold"] = bnh.run(
            self.prices, self.volumes, self.initial_capital
        )

        # 2. Equal Weight
        eq = EqualWeightStrategy(n_assets, rebalance_interval=5, friction_config=self.friction_config)
        self.strategy_results["Equal Weight (1/N)"] = eq.run(
            self.prices, self.volumes, self.initial_capital
        )

        # 3. Markowitz MPT
        mpt = MarkowitzOptimizationStrategy(
            n_assets, lookback_window=60, rebalance_interval=5, friction_config=self.friction_config
        )
        self.strategy_results["Markowitz MPT"] = mpt.run(
            self.prices, self.volumes, self.initial_capital
        )

    def run_rl_agent(self, agent_name: str, agent, env_type: str = "single"):
        """Evaluates a trained RL agent (PPO or SAC) in test environment."""
        if env_type == "single":
            env = PortfolioOptimizationEnv(
                self.prices,
                self.volumes,
                self.features,
                self.asset_names,
                initial_capital=self.initial_capital,
                friction_config=self.friction_config,
            )
            obs, _ = env.reset()
            done = False
            port_values = [env.portfolio_value]
            costs_incurred = [0.0]
            weights_history = [env.weights.copy()]

            while not done:
                # Select deterministic action for evaluation
                if hasattr(agent, "select_action"):
                    action = agent.select_action(obs, deterministic=True)
                    if isinstance(action, tuple):
                        action = action[0]
                else:
                    action = np.zeros(len(self.asset_names) + 1)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                port_values.append(info["portfolio_value"])
                costs_incurred.append(info["total_costs"])
                weights_history.append(info["weights"])

            p_vals = np.array(port_values[:len(self.prices)])
            ret = np.zeros(len(p_vals))
            ret[1:] = (p_vals[1:] - p_vals[:-1]) / (p_vals[:-1] + 1e-9)

            df = pd.DataFrame({
                "portfolio_value": p_vals,
                "daily_return": ret,
                "costs": costs_incurred[:len(self.prices)],
            })
            weights_arr = np.array(weights_history[:len(self.prices)])
            for i, name in enumerate(self.asset_names):
                df[f"weight_{name}"] = weights_arr[:, i]
            df["weight_cash"] = weights_arr[:, -1]
            self.strategy_results[agent_name] = df

        elif env_type == "multi":
            # Cooperative MARL
            env = MultiAgentTradingEnv(
                self.prices,
                self.volumes,
                self.features,
                self.asset_names,
                initial_capital=self.initial_capital,
                friction_config=self.friction_config,
            )
            obs_dict, _ = env.reset()
            done = False
            port_values = [env.portfolio_value]
            costs_incurred = [0.0]
            weights_history = [env.weights.copy()]

            port_agent, exec_agent = agent

            while not done:
                p_act = port_agent.select_action(obs_dict["portfolio_agent"], deterministic=True)
                if isinstance(p_act, tuple):
                    p_act = p_act[0]
                e_act = exec_agent.select_action(obs_dict["execution_agent"], deterministic=True)
                if isinstance(e_act, tuple):
                    e_act = e_act[0]

                actions = {
                    "portfolio_agent": p_act,
                    "execution_agent": e_act,
                }
                obs_dict, rew_dict, term_dict, trunc_dict, info_dict = env.step(actions)
                done = term_dict["portfolio_agent"] or trunc_dict["portfolio_agent"]

                info = info_dict["portfolio_agent"]
                port_values.append(info["portfolio_value"])
                costs_incurred.append(info["total_costs"])
                weights_history.append(info["weights"])

            p_vals = np.array(port_values[:len(self.prices)])
            ret = np.zeros(len(p_vals))
            ret[1:] = (p_vals[1:] - p_vals[:-1]) / (p_vals[:-1] + 1e-9)

            df = pd.DataFrame({
                "portfolio_value": p_vals,
                "daily_return": ret,
                "costs": costs_incurred[:len(self.prices)],
            })
            weights_arr = np.array(weights_history[:len(self.prices)])
            for i, name in enumerate(self.asset_names):
                df[f"weight_{name}"] = weights_arr[:, i]
            df["weight_cash"] = weights_arr[:, -1]
            self.strategy_results[agent_name] = df

    def compute_comparative_metrics(self) -> pd.DataFrame:
        """Calculates and ranks all strategies on financial risk and return metrics."""
        summary = []
        for name, df in self.strategy_results.items():
            metrics = calculate_portfolio_metrics(
                df["portfolio_value"].values,
                df["daily_return"].values,
                df["costs"].values,
            )
            metrics["Strategy"] = name
            summary.append(metrics)

        cols_order = [
            "Strategy",
            "cumulative_return_pct",
            "annualized_return_cagr_pct",
            "annualized_volatility_pct",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown_pct",
            "calmar_ratio",
            "total_friction_costs",
        ]
        self.metrics_summary = pd.DataFrame(summary)[cols_order].sort_values(
            by="sharpe_ratio", ascending=False
        ).reset_index(drop=True)
        return self.metrics_summary

    def run_regime_analysis(self) -> pd.DataFrame:
        """Runs market regime analysis for all strategies."""
        detector = MarketRegimeDetector(
            self.benchmark_prices, dates=self.dates[:len(self.benchmark_prices)]
        )
        regimes = detector.classify_regimes()
        self.regime_summary = evaluate_regimes(self.strategy_results, regimes)
        return self.regime_summary

    def plot_comparisons(self, save_dir: str = "results") -> str:
        """Generates comparative plots: Cumulative Returns & Underwater Drawdowns."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir) / "backtest_comparison.png"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

        dates_plot = self.dates[:len(self.prices)]

        for name, df in self.strategy_results.items():
            p_vals = df["portfolio_value"].values[:len(dates_plot)]
            cum_ret = (p_vals / p_vals[0] - 1.0) * 100.0
            ax1.plot(dates_plot, cum_ret, label=name, lw=2.0 if "MARL" in name else 1.5)

            # Drawdown
            dd = compute_drawdown_series(p_vals) * 100.0
            ax2.plot(dates_plot, dd, label=name, lw=1.2)

        ax1.set_title("Out-of-Sample Cumulative Portfolio Growth (%)", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Cumulative Return (%)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left")

        ax2.set_title("Underwater Drawdown (%)", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Drawdown (%)", fontsize=12)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        return str(save_path)
