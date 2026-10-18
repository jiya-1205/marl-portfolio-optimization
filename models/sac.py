"""Soft Actor-Critic (SAC) implementation with automatic entropy tuning."""

import copy
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.networks import SACTanhActor, SACTwinQCritic


class ReplayBuffer:
    """Experience replay buffer for off-policy SAC learning."""

    def __init__(self, obs_dim: int, act_dim: int, max_size: int = 100_000):
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, max_size

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int = 128) -> dict:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs_buf[idxs], dtype=torch.float32),
            "next_obs": torch.as_tensor(self.next_obs_buf[idxs], dtype=torch.float32),
            "acts": torch.as_tensor(self.acts_buf[idxs], dtype=torch.float32),
            "rews": torch.as_tensor(self.rews_buf[idxs], dtype=torch.float32).unsqueeze(-1),
            "done": torch.as_tensor(self.done_buf[idxs], dtype=torch.float32).unsqueeze(-1),
        }


class SACAgent:
    """Soft Actor-Critic (SAC) Agent."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: float = None,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

        # Actor and Twin Critics
        self.actor = SACTanhActor(obs_dim, act_dim, action_low, action_high).to(self.device)
        self.critic = SACTwinQCritic(obs_dim, act_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Entropy temperature alpha
        self.target_entropy = target_entropy or -float(act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _, mean_action = self.actor.sample(obs_t)
            if deterministic:
                return mean_action.cpu().numpy()[0]
            return action.cpu().numpy()[0]

    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 128) -> dict:
        batch = replay_buffer.sample_batch(batch_size)
        obs = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device)
        done = batch["done"].to(self.device)

        # 1. Update Twin Critics
        with torch.no_grad():
            next_acts, next_log_pi, _ = self.actor.sample(next_obs)
            target_q1, target_q2 = self.critic_target(next_obs, next_acts)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_pi
            target_y = rews + self.gamma * (1.0 - done) * target_q

        current_q1, current_q2 = self.critic(obs, acts)
        q1_loss = nn.MSELoss()(current_q1, target_y)
        q2_loss = nn.MSELoss()(current_q2, target_y)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. Update Actor
        new_acts, log_pi, _ = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, new_acts)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_pi - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. Update Entropy Temperature alpha
        alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # 4. Polyak soft target update
        with torch.no_grad():
            for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_target.data.mul_(1.0 - self.tau)
                p_target.data.add_(self.tau * p.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
        }

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_alpha": self.log_alpha,
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target = copy.deepcopy(self.critic)
        self.log_alpha = checkpoint["log_alpha"]
        self.alpha = self.log_alpha.exp().item()
