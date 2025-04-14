from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from run_config import RunConfig

import gymnasium as gym


class Agent(ABC):
    env: gym.Env

    def __init__(self, run_config: RunConfig, env: gym.Env):
        self.run_config = run_config
        self.env = env

    @abstractmethod
    def act(self, state: np.ndarray) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
        """Returns: unclipped action tensor, clipped action numpy array, log prob tensor"""

    @abstractmethod
    def store_transition(
        self,
        state: np.ndarray,
        action_unclipped: torch.Tensor,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        log_prob: Optional[torch.Tensor],
    ):
        pass

    @abstractmethod
    def update(
        self,
        current_timestep: int,
        total_timesteps: int,
    ):
        pass

    @abstractmethod
    def save(self, path: Path):
        pass

    @staticmethod
    @abstractmethod
    def load(path: Path, run_config: RunConfig, env: gym.Env) -> "Agent":
        pass
