"""Proximal Policy Optimization (PPO) implementation for continuous action spaces."""

from typing import List, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.networks import MLPActor, MLPCritic


class PPOBuffer:
    """Stores trajectories collected during policy rollouts."""

    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        done: bool,
        val: float,
        logp: float,
    ):
        assert self.ptr < self.max_size
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = val
        self.log_probs[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val: float = 0.0, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Computes GAE-lambda advantage and Monte-Carlo returns."""
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rewards[path_slice], last_val)
        vals = np.append(self.values[path_slice], last_val)
        dones = np.append(self.dones[path_slice], 0.0)

        # GAE-lambda calculation
        deltas = rews[:-1] + gamma * vals[1:] * (1.0 - dones[:-1]) - vals[:-1]
        adv = 0.0
        advantages = np.zeros_like(deltas)
        for t in reversed(range(len(deltas))):
            adv = deltas[t] + gamma * gae_lambda * (1.0 - dones[t]) * adv
            advantages[t] = adv

        self.advantages[path_slice] = advantages
        self.returns[path_slice] = advantages + self.values[path_slice]
        self.path_start_idx = self.ptr

    def get(self) -> dict:
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        # Normalize advantages
        adv_mean, adv_std = np.mean(self.advantages), np.std(self.advantages) + 1e-8
        adv_norm = (self.advantages - adv_mean) / adv_std

        data = {
            "obs": torch.as_tensor(self.obs, dtype=torch.float32),
            "actions": torch.as_tensor(self.actions, dtype=torch.float32),
            "returns": torch.as_tensor(self.returns, dtype=torch.float32),
            "advantages": torch.as_tensor(adv_norm, dtype=torch.float32),
            "log_probs": torch.as_tensor(self.log_probs, dtype=torch.float32),
        }
        return data


class PPOAgent:
    """Proximal Policy Optimization (PPO) Agent."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        train_iters: int = 10,
        batch_size: int = 64,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        self.actor = MLPActor(obs_dim, act_dim).to(self.device)
        self.critic = MLPCritic(obs_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mean, std = self.actor(obs_t)
            if deterministic:
                action = mean
                log_prob = torch.tensor([0.0])
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(axis=-1)
            val = self.critic(obs_t)
            return action.cpu().numpy()[0], val.item(), log_prob.item()

    def update(self, buffer: PPOBuffer) -> dict:
        data = buffer.get()
        obs = data["obs"].to(self.device)
        actions = data["actions"].to(self.device)
        returns = data["returns"].to(self.device)
        advantages = data["advantages"].to(self.device)
        old_log_probs = data["log_probs"].to(self.device)

        n_samples = obs.size(0)
        policy_losses, value_losses, entropies = [], [], []

        for _ in range(self.train_iters):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                b_obs = obs[batch_idx]
                b_act = actions[batch_idx]
                b_adv = advantages[batch_idx]
                b_ret = returns[batch_idx]
                b_old_logp = old_log_probs[batch_idx]

                # Actor Loss
                _, log_prob, entropy = self.actor.get_action_and_log_prob(b_obs, b_act)
                ratio = torch.exp(log_prob - b_old_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy.mean()

                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic Loss
                values = self.critic(b_obs)
                value_loss = nn.MSELoss()(values, b_ret)

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies),
        }

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
