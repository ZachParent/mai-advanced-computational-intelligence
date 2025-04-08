from abc import ABC, abstractmethod
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
    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        terminated: bool,
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

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        terminated: bool,
    ):
        pass

    def save(self, path: Path) -> None:
        pass

    @staticmethod
    def load(path: Path, env: gym.Env) -> "RandomAgent":
        return RandomAgent(env)


# --- Actor Network ---
# Outputs the mean of the action distribution
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),  # Tanh activation often used for continuous actions bounded [-1, 1]
            # Pendulum action space is [-2, 2], we'll scale later
        )

    def forward(self, state):
        return self.network(state)


# --- Critic Network ---
# Outputs the estimated value of a state
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Outputs a single value
        )

    def forward(self, state):
        return self.network(state)


class PPOAgent(Agent):
    # Renamed PPOAgent to ActorCriticAgent for clarity, but kept class name PPOAgent
    # as requested implicitly by the file structure. Note this isn't true PPO.
    def __init__(
        self,
        env: gym.Env,
        actor_lr: float = 1e-4,  # Learning rates might need tuning
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
    ):
        super().__init__(env)
        self.gamma = gamma

        # Ensure continuous action space
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

        # --- Actor ---
        self.actor = Actor(state_dim, action_dim)
        # Learnable log standard deviation for action distribution
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()) + [self.actor_log_std], lr=actor_lr
        )

        # --- Critic ---
        self.critic = Critic(state_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Store action scale/bias if needed (Pendulum needs scaling from Tanh's [-1,1])
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

    def get_distribution(self, state: torch.Tensor) -> Normal:
        """Gets the action distribution for a given state"""
        action_mean_normalized = self.actor(state)  # Output is in [-1, 1] due to Tanh
        log_std = self.actor_log_std.expand_as(action_mean_normalized)
        std = torch.exp(log_std)
        # Scale mean to environment's action space
        action_mean = action_mean_normalized * self.action_scale + self.action_bias
        return Normal(action_mean, std)

    def act(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist = self.get_distribution(state_tensor)
            action = dist.sample()  # Sample action from the distribution
        # Clip action to ensure it's within valid bounds
        action_clipped = torch.clamp(action, self.action_low, self.action_high)
        return action_clipped.squeeze(0).numpy()  # Return numpy array

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        terminated: bool,
    ) -> None:
        # Convert inputs to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward]).unsqueeze(0)
        terminated_tensor = torch.FloatTensor([1.0 if terminated else 0.0]).unsqueeze(0)

        # --- Critic Update ---
        with torch.no_grad():
            # Calculate target value V_target = r + gamma * V(s') * (1 - done)
            next_value = self.critic(next_state_tensor)
            target_value = reward_tensor + self.gamma * next_value * (
                1.0 - terminated_tensor
            )

        current_value = self.critic(state_tensor)
        critic_loss = F.mse_loss(current_value, target_value)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update ---
        # Calculate advantage A(s,a) = V_target - V(s)
        # Detach target_value and current_value to prevent gradients flowing into critic update from actor update
        advantage = (target_value - current_value).detach()

        # Get action distribution and log probability of the taken action
        dist = self.get_distribution(state_tensor)
        log_prob = dist.log_prob(action_tensor).sum(
            dim=-1, keepdim=True
        )  # Sum log_prob across action dimensions if > 1

        # Actor loss: -log_prob * advantage
        # Negative sign because we want to maximize log_prob * advantage (gradient ascent)
        # but optimizers perform gradient descent.
        actor_loss = (-log_prob * advantage).mean()  # Use mean if batching later

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path / "actor.pth")
        torch.save(self.critic.state_dict(), path / "critic.pth")

    @staticmethod
    def load(path: Path, env: gym.Env) -> "PPOAgent":
        agent = PPOAgent(env)
        agent.actor.load_state_dict(torch.load(path / "actor.pth"))
        agent.critic.load_state_dict(torch.load(path / "critic.pth"))
        return agent


def get_agent(run_config: RunConfig, env: gym.Env):
    if run_config.agent_name == "random":
        return RandomAgent(env)
    elif run_config.agent_name == "ppo":  # Keeping name 'ppo' for consistency
        # Check if env is suitable (continuous) before creating agent
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(
                f"Agent 'ppo' (Actor-Critic) requires a continuous (Box) action space, "
                f"but environment '{run_config.env_name}' has {type(env.action_space)}."
            )
        return PPOAgent(env)  # Actually returns our Actor-Critic agent
    else:
        raise ValueError(f"Agent {run_config.agent_name} not found")
