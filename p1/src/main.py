import logging

import numpy as np
import pandas as pd
from agent import Agent, get_agent
from config import RESULTS_DIR, VIDEO_DIR
from run_config import RunConfig
from tqdm import tqdm

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


class MetricsLogger:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config
        self.episode_rewards = []
        self.progress_bar = tqdm(total=run_config.num_episodes, desc="Running episodes")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def init_episode(self):
        self.episode_rewards.append(0)

    def end_episode(self):
        self.progress_bar.update(1)
        self.progress_bar.desc = f"Reward: {np.mean(self.episode_rewards)}"

    def update(self, reward: float):
        self.episode_rewards[-1] += reward

    def store_results(self):
        pd.DataFrame(self.episode_rewards).to_csv(
            RESULTS_DIR / f"{self.run_config.id}.csv", index=False
        )

    def close(self):
        self.progress_bar.close()


def wrap_env(env: gym.Env, run_config: RunConfig):
    if run_config.record_episode_spacing:
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=VIDEO_DIR / f"{run_config.id:03d}",
            episode_trigger=lambda episode: episode % run_config.record_episode_spacing
            == 0,
            name_prefix=f"run-{run_config.id:03d}",
        )
    return RecordEpisodeStatistics(env)


def get_env(run_config: RunConfig):
    if run_config.env_name == "CartPole-v1":
        return wrap_env(gym.make("CartPole-v1", render_mode="rgb_array"), run_config)
    elif run_config.env_name == "Pendulum-v1":
        return wrap_env(gym.make("Pendulum-v1", render_mode="rgb_array"), run_config)
    else:
        raise ValueError(f"Environment {run_config.env_name} not found")


def run_episode(
    run_config: RunConfig,
    env: gym.Env,
    agent: Agent,
    episode: int,
    num_steps: int,
    metrics_logger: MetricsLogger,
):
    state, _ = env.reset()
    step = 0
    terminated = False
    truncated = False
    metrics_logger.init_episode()
    while not terminated and not truncated and step < num_steps:
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.update(state, action, next_state, reward, terminated)
        state = next_state
        step += 1
        metrics_logger.update(reward)
    metrics_logger.end_episode()


def run_experiment(run_config: RunConfig):
    env = get_env(run_config)
    agent = get_agent(run_config, env)
    print(f"Running {run_config.num_episodes} episodes of {run_config.num_steps} steps")
    metrics_logger = MetricsLogger(run_config)
    for episode in range(run_config.num_episodes):
        run_episode(
            run_config, env, agent, episode, run_config.num_steps, metrics_logger
        )
    metrics_logger.store_results()
    metrics_logger.close()


def main():
    run_config = RunConfig(
        id=0,
        name="ppo",
        env_name="Pendulum-v1",
        agent_name="ppo",
        num_episodes=20,
        num_steps=1000,
        record_episode_spacing=10,
    )
    run_experiment(run_config)
    logger.error("Experiment finished")


if __name__ == "__main__":
    main()
