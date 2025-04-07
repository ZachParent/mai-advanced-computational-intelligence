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


def get_agent(run_config: RunConfig, env: gym.Env):
    if run_config.agent_name == "random":
        return RandomAgent(env)
    else:
        raise ValueError(f"Agent {run_config.agent_name} not found")
