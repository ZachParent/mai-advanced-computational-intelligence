from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class RunConfig:
    id: int
    name: str
    env_name: Literal["CartPole-v1", "Pendulum-v1"]
    agent_name: Literal["random", "ppo"]
    num_episodes: int
    num_steps: int
    seed: Optional[int] = None
    render: Literal["rgb_array", "human", None] = None
