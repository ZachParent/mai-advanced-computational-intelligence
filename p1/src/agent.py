from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from run_config import RunConfig
from torch.distributions import Normal

import gymnasium as gym


class Agent(ABC):
    env: gym.Env

    def __init__(self, env: gym.Env):
        self.env = env

    @abstractmethod
    def act(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        log_prob: float,
    ):
        pass

    @abstractmethod
    def update(
        self,
    ):
        pass

    @abstractmethod
    def save(self, path: Path):
        pass

    @staticmethod
    @abstractmethod
    def load(path: Path, env: gym.Env) -> "Agent":
        pass


class RandomAgent(Agent):
    def act(self, state: np.ndarray) -> np.ndarray:
        return self.env.action_space.sample()

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        log_prob: float,
    ):
        pass

    def update(
        self,
    ):
        pass

    def save(self, path: Path) -> None:
        pass

    @staticmethod
    def load(path: Path, env: gym.Env) -> "RandomAgent":
        return RandomAgent(env)


class ActorMean(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh(),  # Tanh activation often used for continuous actions bounded [-1, 1]
            # Pendulum action space is [-2, 2], we'll scale later
        )
        for param in self.network.parameters():
            nn.init.normal_(param, 0, 0.0001)

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        for param in self.network.parameters():
            nn.init.normal_(param, 0, 0.01)

    def forward(self, state):
        return self.network(state)


class PPOAgent(Agent):
    def __init__(
        self,
        run_config: RunConfig,
        env: gym.Env,
    ):
        super().__init__(env)
        self.run_config = run_config

        if self.run_config.ppo_hyperparams is None:
            raise ValueError("PPOAgent requires ppo_hyperparams in RunConfig")
        hp = self.run_config.ppo_hyperparams

        self.clip_epsilon = hp.clip_epsilon
        self.update_epochs = hp.update_epochs
        self.gae_lambda = hp.gae_lambda
        self.buffer_capacity = hp.buffer_capacity
        self.buffer = deque(maxlen=self.buffer_capacity)

        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(
                "PPOAgent implementation only supports Box (continuous) action spaces."
            )
        self.action_low = torch.tensor(env.action_space.low, dtype=torch.float32)
        self.action_high = torch.tensor(env.action_space.high, dtype=torch.float32)

        assert (
            env.observation_space.shape is not None
        ), "Observation space shape cannot be None"
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.actor_mean = ActorMean(state_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        nn.init.normal_(self.actor_log_std, 0, 0.0001)
        self.actor_optimizer = optim.Adam(
            list(self.actor_mean.parameters()) + [self.actor_log_std],
            lr=hp.actor_lr,
        )

        self.critic = Critic(state_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=hp.critic_lr)

        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

    def get_distribution(self, state: torch.Tensor) -> Normal:
        action_mean_normalized = self.actor_mean(state)
        log_std = self.actor_log_std.expand_as(action_mean_normalized)
        std = torch.exp(log_std)
        action_mean = action_mean_normalized * self.action_scale + self.action_bias
        return Normal(action_mean, std)

    def act(self, state: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist = self.get_distribution(state_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        action_clipped = torch.clamp(action, self.action_low, self.action_high)
        return action_clipped.squeeze(0).numpy(), log_prob

    def store_transition(self, state, action, reward, next_state, terminated, log_prob):
        self.buffer.append((state, action, reward, next_state, terminated, log_prob))

    def update(self) -> None:
        if len(self.buffer) == 0:
            return

        if self.run_config.ppo_hyperparams is None:
            raise ValueError(
                "PPOAgent requires ppo_hyperparams in RunConfig for update"
            )
        hp = self.run_config.ppo_hyperparams

        batch = list(self.buffer)
        states, actions, rewards, next_states, terminateds, log_probs_old = zip(*batch)

        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.FloatTensor(np.array(actions))
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_tensor = torch.FloatTensor(np.array(next_states))
        terminateds_tensor = torch.FloatTensor(terminateds).unsqueeze(1)
        log_probs_old_tensor = torch.stack(log_probs_old).detach().unsqueeze(1)

        with torch.no_grad():
            values = self.critic(states_tensor)
            next_values = self.critic(next_states_tensor)

            advantages = torch.zeros_like(rewards_tensor)
            last_advantage = 0
            for t in reversed(range(len(rewards_tensor))):
                mask = 1.0 - terminateds_tensor[t]
                delta = rewards_tensor[t] + hp.gamma * next_values[t] * mask - values[t]
                advantages[t] = delta + hp.gamma * hp.gae_lambda * last_advantage * mask
                last_advantage = advantages[t]

            returns_tensor = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_tensor = states_tensor.view(-1, states_tensor.shape[-1])
        actions_tensor = actions_tensor.view(-1, actions_tensor.shape[-1])
        log_probs_old_tensor = log_probs_old_tensor.view(-1)
        advantages = advantages.view(-1)
        returns_tensor = returns_tensor.view(-1)

        num_samples = states_tensor.shape[0]

        for epoch in range(hp.update_epochs):
            dist = self.get_distribution(states_tensor)
            log_probs_new = dist.log_prob(actions_tensor).sum(dim=-1)
            entropy = dist.entropy().mean()
            values_new = self.critic(states_tensor).view(-1)

            ratio = torch.exp(log_probs_new - log_probs_old_tensor)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - hp.clip_epsilon, 1.0 + hp.clip_epsilon)
                * advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            critic_loss = F.mse_loss(values_new, returns_tensor)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        self.buffer.clear()

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor_mean.state_dict(), path / "actor_mean.pth")
        torch.save(self.actor_log_std.data, path / "actor_log_std.pth")
        torch.save(self.critic.state_dict(), path / "critic.pth")

    @staticmethod
    def load(path: Path, run_config: RunConfig, env: gym.Env) -> "PPOAgent":
        agent = PPOAgent(run_config, env)
        agent.actor_mean.load_state_dict(torch.load(path / "actor_mean.pth"))
        agent.actor_log_std.data = torch.load(path / "actor_log_std.pth")
        agent.critic.load_state_dict(torch.load(path / "critic.pth"))
        return agent


def get_agent(run_config: RunConfig, env: gym.Env):
    if run_config.agent_name == "random":
        try:
            return RandomAgent(env)
        except TypeError:
            print("Warning: RandomAgent missing save/load, cannot be instantiated.")
            raise
    elif run_config.agent_name == "ppo":
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(f"Agent 'ppo' requires Box space")
        return PPOAgent(run_config=run_config, env=env)
    else:
        raise ValueError(f"Agent {run_config.agent_name} not found")
