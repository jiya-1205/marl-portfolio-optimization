"""Unified training script for Portfolio, Execution, and Multi-Agent RL agents."""

import argparse
from pathlib import Path
import numpy as np
import torch

from data.fetch_data import fetch_and_cache_data
from data.feature_pipeline import FeaturePipeline
from environments.market_frictions import MarketFrictionConfig
from environments.portfolio_env import PortfolioOptimizationEnv
from environments.execution_env import ExecutionEnv
from environments.multi_agent_env import MultiAgentTradingEnv
from models.ppo import PPOAgent, PPOBuffer
from models.sac import SACAgent, ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Financial RL Agents")
    parser.add_argument("--mode", type=str, default="multiagent", choices=["portfolio", "execution", "multiagent"])
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    return parser.parse_args()


def train_portfolio_ppo(env: PortfolioOptimizationEnv, episodes: int, save_path: str):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, lr_actor=3e-4, lr_critic=1e-3)
    rollout_len = env.n_timesteps - 1

    print(f"--- Training Portfolio PPO Agent for {episodes} episodes ---", flush=True)
    for ep in range(episodes):
        obs, _ = env.reset()
        buffer = PPOBuffer(obs_dim, act_dim, rollout_len)
        total_reward = 0.0

        for t in range(rollout_len):
            act, val, logp = agent.select_action(obs)
            next_obs, rew, term, trunc, info = env.step(act)
            done = term or trunc
            buffer.store(obs, act, rew, done, val, logp)
            obs = next_obs
            total_reward += rew
            if done:
                break

        last_val = 0.0 if done else agent.select_action(obs)[1]
        buffer.finish_path(last_val)
        update_info = agent.update(buffer)

        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"Episode {ep + 1}/{episodes} | Total Reward: {total_reward:,.4f} | "
                f"Final Value: ${env.portfolio_value:,.2f} | Loss Pi: {update_info['policy_loss']:.4f}",
                flush=True,
            )

    agent.save(save_path)
    print(f"Portfolio PPO model saved to {save_path}", flush=True)
    return agent


def train_execution_ppo(env: ExecutionEnv, episodes: int, save_path: str):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, lr_actor=5e-4, lr_critic=1e-3)
    steps_per_rollout = env.n_substeps * 10

    print(f"--- Training Execution PPO Agent for {episodes} episodes ---")
    buffer = PPOBuffer(obs_dim, act_dim, steps_per_rollout)
    step_count = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_rew = 0.0

        while not done:
            act, val, logp = agent.select_action(obs)
            next_obs, rew, term, trunc, info = env.step(act)
            done = term or trunc
            buffer.store(obs, act, rew, done, val, logp)
            obs = next_obs
            ep_rew += rew
            step_count += 1

            if step_count >= steps_per_rollout:
                buffer.finish_path(0.0)
                agent.update(buffer)
                step_count = 0

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Execution Episode {ep + 1}/{episodes} | Reward: {ep_rew:.4f} | Avg Exec Price: ${info['avg_execution_price']:.2f}")

    agent.save(save_path)
    print(f"Execution PPO model saved to {save_path}")
    return agent


def train_multiagent_ppo(env: MultiAgentTradingEnv, episodes: int, save_dir: str):
    port_obs_dim = env.observation_spaces["portfolio_agent"].shape[0]
    port_act_dim = env.action_spaces["portfolio_agent"].shape[0]
    exec_obs_dim = env.observation_spaces["execution_agent"].shape[0]
    exec_act_dim = env.action_spaces["execution_agent"].shape[0]

    port_agent = PPOAgent(obs_dim=port_obs_dim, act_dim=port_act_dim)
    exec_agent = PPOAgent(obs_dim=exec_obs_dim, act_dim=exec_act_dim)

    rollout_len = env.n_timesteps - 1
    print(f"--- Training Multi-Agent (Portfolio + Execution) PPO for {episodes} episodes ---", flush=True)

    for ep in range(episodes):
        obs_dict, _ = env.reset()
        port_buf = PPOBuffer(port_obs_dim, port_act_dim, rollout_len)
        exec_buf = PPOBuffer(exec_obs_dim, exec_act_dim, rollout_len)
        tot_port_rew, tot_exec_rew = 0.0, 0.0

        for t in range(rollout_len):
            p_act, p_val, p_logp = port_agent.select_action(obs_dict["portfolio_agent"])
            e_act, e_val, e_logp = exec_agent.select_action(obs_dict["execution_agent"])

            actions = {"portfolio_agent": p_act, "execution_agent": e_act}
            next_obs_dict, rews, terms, truncs, infos = env.step(actions)
            done = terms["portfolio_agent"] or truncs["portfolio_agent"]

            port_buf.store(obs_dict["portfolio_agent"], p_act, rews["portfolio_agent"], done, p_val, p_logp)
            exec_buf.store(obs_dict["execution_agent"], e_act, rews["execution_agent"], done, e_val, e_logp)

            tot_port_rew += rews["portfolio_agent"]
            tot_exec_rew += rews["execution_agent"]
            obs_dict = next_obs_dict
            if done:
                break

        port_buf.finish_path(0.0)
        exec_buf.finish_path(0.0)
        port_agent.update(port_buf)
        exec_agent.update(exec_buf)

        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"MARL Episode {ep + 1}/{episodes} | Port Val: ${env.portfolio_value:,.2f} | "
                f"Port Rew: {tot_port_rew:.4f} | Exec Rew: {tot_exec_rew:.4f}",
                flush=True,
            )

    save_p = Path(save_dir)
    save_p.mkdir(parents=True, exist_ok=True)
    port_agent.save(str(save_p / "marl_portfolio_agent.pt"))
    exec_agent.save(str(save_p / "marl_execution_agent.pt"))
    print(f"Multi-Agent checkpoints saved to {save_dir}/", flush=True)
    return port_agent, exec_agent


def train_portfolio_sac(env: PortfolioOptimizationEnv, episodes: int, save_path: str):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim, action_low=act_low, action_high=act_high)
    replay_buffer = ReplayBuffer(obs_dim, act_dim, max_size=50_000)

    print(f"--- Training Portfolio SAC Agent for {episodes} episodes ---", flush=True)
    step_count = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_rew = 0.0
        done = False

        while not done:
            act = agent.select_action(obs, deterministic=False)
            next_obs, rew, term, trunc, info = env.step(act)
            done = term or trunc
            replay_buffer.store(obs, act, rew, next_obs, done)
            obs = next_obs
            ep_rew += rew
            step_count += 1

            if step_count % 4 == 0 and replay_buffer.size > 256:
                agent.update(replay_buffer, batch_size=128)

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"SAC Episode {ep + 1}/{episodes} | Port Val: ${env.portfolio_value:,.2f} | Ep Reward: {ep_rew:.4f} | Alpha: {agent.alpha:.4f}", flush=True)

    agent.save(save_path)
    print(f"Portfolio SAC model saved to {save_path}", flush=True)
    return agent


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
    benchmark = "SPY"
    data_dict = fetch_and_cache_data(tickers=tickers, benchmark=benchmark)

    pipeline = FeaturePipeline(tickers, benchmark)
    dates, prices, volumes, features = pipeline.fit_transform(data_dict)

    # 80% train / 20% test split
    split_idx = int(0.8 * len(dates))
    train_prices = prices[:split_idx]
    train_vols = volumes[:split_idx]
    train_features = features[:split_idx]

    friction_cfg = MarketFrictionConfig()

    if args.mode == "portfolio":
        env = PortfolioOptimizationEnv(train_prices, train_vols, train_features, tickers, friction_config=friction_cfg)
        if args.algo == "ppo":
            train_portfolio_ppo(env, args.episodes, f"{args.save_dir}/portfolio_ppo.pt")
        else:
            train_portfolio_sac(env, args.episodes, f"{args.save_dir}/portfolio_sac.pt")

    elif args.mode == "execution":
        env = ExecutionEnv(target_shares=5000.0, friction_config=friction_cfg)
        train_execution_ppo(env, args.episodes, f"{args.save_dir}/execution_ppo.pt")

    elif args.mode == "multiagent":
        env = MultiAgentTradingEnv(train_prices, train_vols, train_features, tickers, friction_config=friction_cfg)
        train_multiagent_ppo(env, args.episodes, args.save_dir)


if __name__ == "__main__":
    main()
