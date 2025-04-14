from pathlib import Path
from typing import Optional

import numpy as np
import torch
from agent import Agent
from run_config import RunConfig

import gymnasium as gym


class RandomAgent(Agent):
    def act(self, state: np.ndarray) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
        action_np = self.env.action_space.sample()
        action_tensor = torch.tensor(action_np, dtype=torch.float32)
        return (
            action_tensor,
            action_np,
            torch.tensor(0.0),
        )

    def store_transition(
        self,
        state: np.ndarray,
        action_unclipped: torch.Tensor,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        log_prob: Optional[torch.Tensor],
    ):
        pass  # Random agent doesn't store

    def update(
        self,
        current_timestep: int,
        total_timesteps: int,
    ):
        pass  # Random agent doesn't update

    def save(self, path: Path) -> None:
        pass  # Random agent doesn't save

    @staticmethod
    def load(path: Path, run_config: RunConfig, env: gym.Env) -> "RandomAgent":
        return RandomAgent(run_config=run_config, env=env)
