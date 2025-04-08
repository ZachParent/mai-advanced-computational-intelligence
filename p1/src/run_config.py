from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class RunConfig:
    id: int
    name: str
    env_name: Literal["CartPole-v1", "Pendulum-v1"]
    agent_name: Literal["random", "ppo"]
    actor_lr: float
    critic_lr: float
    gamma: float
    num_episodes: int
    num_steps: int
    seed: Optional[int] = None
    record_episode_spacing: Optional[int] = None


CONFIGS = [
    RunConfig(
        id=0,
        name="Pendulum-v1-ppo low lr",
        env_name="Pendulum-v1",
        agent_name="ppo",
        num_episodes=400,
        num_steps=400,
        record_episode_spacing=20,
        gamma=0.99,
        actor_lr=1e-4,
        critic_lr=1e-3,
    ),
    RunConfig(
        id=1,
        name="Pendulum-v1-ppo high lr",
        env_name="Pendulum-v1",
        agent_name="ppo",
        num_episodes=400,
        num_steps=400,
        record_episode_spacing=20,
        gamma=0.99,
        actor_lr=1e-3,
        critic_lr=1e-2,
    ),
]
