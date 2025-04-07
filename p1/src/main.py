import logging

from agent import Agent, get_agent
from run_config import RunConfig

import gymnasium as gym

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_env(run_config: RunConfig):
    if run_config.env_name == "CartPole-v1":
        return gym.make("CartPole-v1", render_mode=run_config.render)
    # elif run_config.env_name == "MountainCar-v0":
    #     return gym.make("MountainCar-v0")
    # elif run_config.env_name == "Pendulum-v1":
    #     return gym.make("Pendulum-v1")
    else:
        raise ValueError(f"Environment {run_config.env_name} not found")


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
        render=None,
        agent_name="random",
        num_episodes=10,
        num_steps=1000,
    )
    run_experiment(run_config)
    logger.error("Experiment finished")


if __name__ == "__main__":
    main()
