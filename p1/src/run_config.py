from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class PPOHyperparams:
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    update_epochs: int = 10
    gae_lambda: float = 0.95
    buffer_capacity: int = 2000
    num_episodes_per_update: int = 100


@dataclass(frozen=True)
class RunConfig:
    id: int
    name: str
    env_name: Literal["CartPole-v1", "Pendulum-v1"]
    agent_name: Literal["random", "ppo"]
    num_episodes: int = 10000
    num_steps: int = 400
    seed: Optional[int] = None
    record_episode_spacing: Optional[int] = 100
    ppo_hyperparams: Optional[PPOHyperparams] = None

    def __post_init__(self):
        if self.agent_name == "ppo" and self.ppo_hyperparams is None:
            raise ValueError("ppo_hyperparams must be provided for PPO agent")
        if self.agent_name != "ppo" and self.ppo_hyperparams is not None:
            print(
                f"Warning: ppo_hyperparams provided for non-PPO agent '{self.agent_name}', will be ignored."
            )


CONFIGS = [
    # RunConfig(
    #     id=0,
    #     name="Pendulum-v1-ppo low lr",
    #     env_name="Pendulum-v1",
    #     agent_name="ppo",
    #     ppo_hyperparams=PPOHyperparams(
    #         actor_lr=1e-4,
    #         critic_lr=1e-3,
    #     ),
    # ),
    # RunConfig(
    #     id=1,
    #     name="Pendulum-v1-ppo high lr",
    #     env_name="Pendulum-v1",
    #     agent_name="ppo",
    #     ppo_hyperparams=PPOHyperparams(
    #         actor_lr=5e-3,
    #         critic_lr=5e-2,
    #     ),
    # ),
    RunConfig(
        id=1,
        name="Pendulum-v1-ppo balanced lr",
        env_name="Pendulum-v1",
        agent_name="ppo",
        ppo_hyperparams=PPOHyperparams(
            actor_lr=5e-3,
            critic_lr=1e-2,
        ),
    ),
]
