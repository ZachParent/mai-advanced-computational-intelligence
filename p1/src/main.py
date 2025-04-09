import logging

import pandas as pd
from agent import Agent, PPOAgent, get_agent
from config import AGENT_DIR, RESULTS_DIR, VIDEO_DIR
from run_config import CONFIGS, RunConfig
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
        self.progress_bar.desc = f"Reward: {self.episode_rewards[-1]}"

    def update(self, reward: float):
        self.episode_rewards[-1] += reward

    def store_results(self):
        pd.DataFrame(self.episode_rewards).to_csv(
            RESULTS_DIR / f"{self.run_config.id:02d}.csv", index=False
        )

    def close(self):
        self.progress_bar.close()


def wrap_env(env: gym.Env, run_config: RunConfig):
    if type(run_config.record_episode_spacing) == int:
        episode_trigger = (
            lambda episode: episode % run_config.record_episode_spacing == 0
        )
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=str(VIDEO_DIR / f"{run_config.id:02d}"),
            episode_trigger=episode_trigger,
            name_prefix=f"run-{run_config.id:02d}",
        )
    return RecordEpisodeStatistics(env)


def get_env(run_config: RunConfig):
    if run_config.env_name == "Pendulum-v1":
        return wrap_env(gym.make("Pendulum-v1", render_mode="rgb_array"), run_config)
    elif run_config.env_name == "InvertedPendulum-v5":
        return wrap_env(
            gym.make("InvertedPendulum-v5", render_mode="rgb_array"), run_config
        )
    elif run_config.env_name == "Ant-v5":
        # env = gym.make("Ant-v5", render_mode="rgb_array")
        # env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(
        #     env, lambda obs: np.clip(obs, -10, 10), env.observation_space
        # )
        # env = gym.wrappers.NormalizeReward(env)
        # env = gym.wrappers.TransformReward(
        #     env, lambda reward: np.clip(float(reward), -10, 10)
        # )
        return wrap_env(gym.make("Ant-v5", render_mode="rgb_array"), run_config)
    elif run_config.env_name == "Humanoid-v5":
        return wrap_env(gym.make("Humanoid-v5", render_mode="rgb_array"), run_config)
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
        action_np, log_prob_tensor = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action_np)
        agent.store_transition(
            state, action_np, float(reward), next_state, terminated, log_prob_tensor
        )
        state = next_state
        step += 1
        metrics_logger.update(float(reward))
    metrics_logger.end_episode()


def run_experiment(run_config: RunConfig):
    env = get_env(run_config)
    agent = get_agent(run_config, env)

    is_trainable = isinstance(agent, PPOAgent)
    if is_trainable:
        if run_config.ppo_hyperparams is None:
            raise ValueError("Trainable PPOAgent requires ppo_hyperparams in RunConfig")
        print(
            f"Running {run_config.num_episodes} episodes, updating every {run_config.ppo_hyperparams.num_episodes_per_update} episodes."
        )
    else:
        print(
            f"Running {run_config.num_episodes} episodes with non-trainable agent {run_config.agent_name}."
        )

    metrics_logger = MetricsLogger(run_config)

    for episode in range(run_config.num_episodes):
        run_episode(
            run_config, env, agent, episode, run_config.num_steps, metrics_logger
        )

        if run_config.ppo_hyperparams is None:
            raise ValueError("Trainable PPOAgent requires ppo_hyperparams in RunConfig")
        if (
            (episode + 1) % run_config.ppo_hyperparams.num_episodes_per_update == 0
            or episode == run_config.num_episodes - 1
        ):
            agent.update()

    metrics_logger.store_results()
    metrics_logger.close()
    save_path = AGENT_DIR / f"{run_config.id:02d}"
    agent.save(save_path)
    print(f"Agent saved to {save_path}")


def main():
    for run_config in CONFIGS:
        print("=" * 80)
        logger.info(f"Experiment {run_config.id}: {run_config.name}".center(80))
        run_experiment(run_config)


if __name__ == "__main__":
    main()
