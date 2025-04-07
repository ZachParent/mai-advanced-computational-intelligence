import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

import gymnasium as gym

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class RunConfig:
    env_name: Literal["CartPole-v1"]
    agent_name: Literal["random"]
    num_episodes: int
    num_steps: int
    seed: Optional[int] = None
    render: Literal["rgb_array", "human", None] = None


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


def get_env(run_config: RunConfig):
    if run_config.env_name == "CartPole-v1":
        return gym.make("CartPole-v1", render_mode=run_config.render)
    # elif run_config.env_name == "MountainCar-v0":
    #     return gym.make("MountainCar-v0")
    # elif run_config.env_name == "Pendulum-v1":
    #     return gym.make("Pendulum-v1")
    else:
        raise ValueError(f"Environment {run_config.env_name} not found")


def get_agent(run_config: RunConfig, env: gym.Env):
    if run_config.agent_name == "random":
        return RandomAgent(env)
    else:
        raise ValueError(f"Agent {run_config.agent_name} not found")


def run_episode(
    run_config: RunConfig, env: gym.Env, agent: Agent, episode: int, num_steps: int
):
    state, _ = env.reset()
    step = 0
    terminated = False
    truncated = False
    while not terminated and not truncated and step < num_steps:
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.update(state, action, next_state, reward)
        state = next_state
        step += 1
        logger.info(f"Step {step} of episode {episode} of {run_config.num_episodes}")


def run_experiment(run_config: RunConfig):
    env = get_env(run_config)
    agent = get_agent(run_config, env)
    print(f"Running {run_config.num_episodes} episodes of {run_config.num_steps} steps")
    for episode in range(run_config.num_episodes):
        run_episode(run_config, env, agent, episode, run_config.num_steps)


def main():
    run_config = RunConfig(
        env_name="CartPole-v1",
        render="human",
        agent_name="random",
        num_episodes=10,
        num_steps=1000,
    )
    run_experiment(run_config)
    logger.error("Experiment finished")


if __name__ == "__main__":
    main()
