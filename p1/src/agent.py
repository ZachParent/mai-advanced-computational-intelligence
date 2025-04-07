from abc import ABC, abstractmethod

import numpy as np
from run_config import RunConfig

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
    ):
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
    ):
        pass


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PPOAgent(Agent):
    def __init__(self, env: gym.Env, learning_rate: float = 0.001, gamma: float = 0.99):
        super().__init__(env)
        self.policy_net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, env.action_space.n),
            nn.Softmax(dim=-1),
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma

    def act(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_net(state_tensor)
        action = np.random.choice(
            self.env.action_space.n, p=action_probs.detach().numpy()[0]
        )
        return action

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
    ) -> None:
        # Calculate the advantage
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward])

        # Get current action probabilities
        current_probs = self.policy_net(state_tensor)
        current_action_prob = current_probs[0, action]

        # Calculate the target value (using reward and next state)
        with torch.no_grad():
            next_action_probs = self.policy_net(next_state_tensor)
        target_value = reward_tensor + self.gamma * next_action_probs[0, action]

        # Calculate the loss
        loss = -torch.log(current_action_prob) * target_value

        # Backpropagate the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def get_agent(run_config: RunConfig, env: gym.Env):
    if run_config.agent_name == "random":
        return RandomAgent(env)
    elif run_config.agent_name == "ppo":
        return PPOAgent(env)
    else:
        raise ValueError(f"Agent {run_config.agent_name} not found")
