"""Out-of-sample evaluation and benchmarking script."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from data.fetch_data import fetch_and_cache_data
from data.feature_pipeline import FeaturePipeline
from environments.market_frictions import MarketFrictionConfig
from environments.portfolio_env import PortfolioOptimizationEnv
from environments.multi_agent_env import MultiAgentTradingEnv
from models.ppo import PPOAgent
from models.sac import SACAgent
from backtesting.engine import BacktestEngine
from train import train_portfolio_ppo, train_portfolio_sac, train_multiagent_ppo


def run_evaluation(episodes_train_if_missing: int = 15):
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
    benchmark = "SPY"

    print("Step 1: Fetching and preprocessing market data...")
    data_dict = fetch_and_cache_data(tickers=tickers, benchmark=benchmark)
    pipeline = FeaturePipeline(tickers, benchmark)
    dates, prices, volumes, features = pipeline.fit_transform(data_dict)

    # 80% Train, 20% Test Split
    split_idx = int(0.8 * len(dates))
    train_dates, test_dates = dates[:split_idx], dates[split_idx:]
    train_prices, test_prices = prices[:split_idx], prices[split_idx:]
    train_vols, test_vols = volumes[:split_idx], volumes[split_idx:]
    train_feats, test_feats = features[:split_idx], features[split_idx:]

    bench_test_prices = data_dict[benchmark].loc[test_dates, "Close"].values

    friction_cfg = MarketFrictionConfig()
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. PPO Agent
    port_ppo_path = ckpt_dir / "portfolio_ppo.pt"
    port_env = PortfolioOptimizationEnv(train_prices, train_vols, train_feats, tickers, friction_config=friction_cfg)
    port_ppo_agent = PPOAgent(obs_dim=port_env.observation_space.shape[0], act_dim=port_env.action_space.shape[0])

    if port_ppo_path.exists():
        print(f"Loading trained PPO Agent from {port_ppo_path}...")
        port_ppo_agent.load(str(port_ppo_path))
    else:
        print("Training PPO Agent...")
        train_portfolio_ppo(port_env, episodes=episodes_train_if_missing, save_path=str(port_ppo_path))
        port_ppo_agent.load(str(port_ppo_path))

    # 2. SAC Agent
    port_sac_path = ckpt_dir / "portfolio_sac.pt"
    port_sac_agent = SACAgent(
        obs_dim=port_env.observation_space.shape[0],
        act_dim=port_env.action_space.shape[0],
        action_low=port_env.action_space.low,
        action_high=port_env.action_space.high,
    )
    if port_sac_path.exists():
        print(f"Loading trained SAC Agent from {port_sac_path}...")
        port_sac_agent.load(str(port_sac_path))
    else:
        print("Training SAC Agent...")
        train_portfolio_sac(port_env, episodes=episodes_train_if_missing, save_path=str(port_sac_path))
        port_sac_agent.load(str(port_sac_path))

    # 3. Multi-Agent System (MARL)
    marl_port_path = ckpt_dir / "marl_portfolio_agent.pt"
    marl_exec_path = ckpt_dir / "marl_execution_agent.pt"
    marl_env = MultiAgentTradingEnv(train_prices, train_vols, train_feats, tickers, friction_config=friction_cfg)

    marl_port_agent = PPOAgent(
        obs_dim=marl_env.observation_spaces["portfolio_agent"].shape[0],
        act_dim=marl_env.action_spaces["portfolio_agent"].shape[0],
    )
    marl_exec_agent = PPOAgent(
        obs_dim=marl_env.observation_spaces["execution_agent"].shape[0],
        act_dim=marl_env.action_spaces["execution_agent"].shape[0],
    )

    if marl_port_path.exists() and marl_exec_path.exists():
        print("Loading trained Multi-Agent models...")
        marl_port_agent.load(str(marl_port_path))
        marl_exec_agent.load(str(marl_exec_path))
    else:
        print("Training Multi-Agent System...")
        train_multiagent_ppo(marl_env, episodes=episodes_train_if_missing, save_dir=str(ckpt_dir))
        marl_port_agent.load(str(marl_port_path))
        marl_exec_agent.load(str(marl_exec_path))

    # Step 2: Out-of-sample Backtesting
    print("\nStep 2: Executing Out-of-Sample Backtesting on Test Set (20% unseen data)...")
    engine = BacktestEngine(
        dates=test_dates,
        prices=test_prices,
        volumes=test_vols,
        features=test_feats,
        asset_names=tickers,
        benchmark_prices=bench_test_prices,
        initial_capital=100_000.0,
        friction_config=friction_cfg,
    )

    print("Running classical benchmarks: Buy & Hold, Equal Weight, Markowitz MPT...")
    engine.run_baselines()

    print("Evaluating Single PPO Portfolio Agent...")
    engine.run_rl_agent("PPO Portfolio Agent", port_ppo_agent, env_type="single")

    print("Evaluating Single SAC Portfolio Agent...")
    engine.run_rl_agent("SAC Portfolio Agent", port_sac_agent, env_type="single")

    print("Evaluating Hierarchical Multi-Agent RL (Portfolio + Execution)...")
    engine.run_rl_agent("MARL (Portfolio + Execution)", (marl_port_agent, marl_exec_agent), env_type="multi")

    # Step 3: Comparative Analysis
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = engine.compute_comparative_metrics()
    regime_df = engine.run_regime_analysis()
    plot_file = engine.plot_comparisons(str(results_dir))

    # Save to CSV
    metrics_path = results_dir / "metrics_summary.csv"
    regime_path = results_dir / "regime_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    regime_df.to_csv(regime_path, index=False)

    print("\n" + "=" * 80)
    print("           OUT-OF-SAMPLE PERFORMANCE COMPARISON SUMMARY")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print("\n" + "=" * 80)
    print("           MARKET REGIME ROBUSTNESS BREAKDOWN")
    print("=" * 80)
    print(regime_df.to_string(index=False))
    print("\nComparative plot saved to:", plot_file)
    print("Metrics CSV saved to:", metrics_path)
    print("Regime CSV saved to:", regime_path)

    return metrics_df, regime_df


if __name__ == "__main__":
    run_evaluation()
