import random
import time  # For tracking total steps

import numpy as np
import pandas as pd
import torch  # Import torch
from agent import Agent, PPOAgent, RandomAgent
from config import AGENT_DIR, RESULTS_DIR, VIDEO_DIR
from run_config import CONFIGS, RunConfig
from tqdm import tqdm

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


class MetricsLogger:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config
        self.all_rewards = []
        self.all_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.progress_bar = tqdm(
            total=run_config.total_timesteps, desc="Training Steps"
        )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def start_episode(self):
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def end_episode(self):
        self.all_rewards.append(self.current_episode_reward)
        self.all_lengths.append(self.current_episode_length)
        avg_reward = np.mean(self.all_rewards[-10:]) if self.all_rewards else 0
        self.progress_bar.set_description(
            f"Steps | Avg Reward (last 10): {avg_reward:.2f}"
        )

    def step(self, reward: float, steps_added: int = 1):
        self.current_episode_reward += reward
        self.current_episode_length += steps_added
        self.progress_bar.update(steps_added)

    def store_results(self):
        df = pd.DataFrame({"rewards": self.all_rewards, "lengths": self.all_lengths})
        df.to_csv(RESULTS_DIR / f"{self.run_config.id:02d}.csv", index_label="episode")

    def close(self):
        self.progress_bar.close()


def wrap_env(env: gym.Env, run_config: RunConfig):
    env = RecordEpisodeStatistics(env)
    if (
        run_config.record_episode_spacing is not None
        and run_config.record_episode_spacing > 0
    ):
        episode_trigger = (
            lambda episode_id: episode_id % run_config.record_episode_spacing == 0
        )
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=str(VIDEO_DIR / f"{run_config.id:02d}"),
            episode_trigger=episode_trigger,
            name_prefix=f"run-{run_config.id:02d}",
        )
    return env


def get_env(run_config: RunConfig):
    env = gym.make(run_config.env_name, render_mode="rgb_array")

    if run_config.max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(
            env, max_episode_steps=run_config.max_episode_steps
        )

    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(
        env, lambda obs: np.clip(obs, -10, 10), observation_space=env.observation_space
    )
    env = gym.wrappers.NormalizeReward(
        env,
        gamma=run_config.ppo_hyperparams.gamma if run_config.ppo_hyperparams else 0.99,
    )
    env = gym.wrappers.TransformReward(
        env, lambda reward: np.clip(float(reward), -10, 10)
    )

    env = wrap_env(env, run_config)
    return env


def get_agent(run_config: RunConfig, env: gym.Env) -> Agent:
    if run_config.agent_name == "random":
        return RandomAgent(run_config=run_config, env=env)
    elif run_config.agent_name == "ppo":
        return PPOAgent(run_config=run_config, env=env)
    else:
        raise ValueError(f"Agent {run_config.agent_name} not found")


def run_experiment(run_config: RunConfig):
    start_time = time.time()

    if run_config.seed is not None:
        random.seed(run_config.seed)
        np.random.seed(run_config.seed)
        torch.manual_seed(run_config.seed)
        torch.backends.cudnn.deterministic = True

    env = get_env(run_config)
    agent = get_agent(run_config, env)
    is_trainable = isinstance(agent, PPOAgent)

    if is_trainable:
        hp = run_config.ppo_hyperparams
        if hp is None:
            raise ValueError("PPOAgent needs hyperparameters")
    else:
        print(f"  Agent: {run_config.agent_name} (Not training)")

    metrics_logger = MetricsLogger(run_config)
    global_step = 0
    state, info = env.reset(seed=run_config.seed)
    metrics_logger.start_episode()

    while global_step < run_config.total_timesteps:
        action_unclipped_tensor, action_clipped_np, log_prob_tensor = agent.act(state)

        next_state, reward, terminated, truncated, info = env.step(action_clipped_np)
        global_step += 1
        metrics_logger.step(float(reward))

        if is_trainable:
            agent.store_transition(
                state,
                action_unclipped_tensor,
                float(reward),
                next_state,
                terminated,
                log_prob_tensor,
            )

        state = next_state

        if terminated or truncated:
            metrics_logger.end_episode()
            state, info = env.reset()
            metrics_logger.start_episode()

        if is_trainable:
            agent.update(global_step, run_config.total_timesteps)

    metrics_logger.store_results()
    metrics_logger.close()
    env.close()

    if hasattr(agent, "save"):
        save_path = AGENT_DIR / f"{run_config.id:02d}"
        agent.save(save_path)
        print(f"Agent saved to {save_path}")

    print(f"Experiment finished in {(time.time() - start_time):.2f} seconds.")


def main():
    for run_config in CONFIGS:
        print("=" * 80)
        print(f"Experiment {run_config.id}: {run_config.name}".center(80))
        run_experiment(run_config)


if __name__ == "__main__":
    main()
