from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class RunConfig:
    env_name: Literal["CartPole-v1"]
    agent_name: Literal["random"]
    num_episodes: int
    num_steps: int
    seed: Optional[int] = None
    render: Literal["rgb_array", "human", None] = None
