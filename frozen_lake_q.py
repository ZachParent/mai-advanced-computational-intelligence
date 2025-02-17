import gymnasium as gym
import numpy as np
def main():
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode='human')
    state = env.reset()[0]
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print(next_state, reward, terminated, truncated, info)
        state = next_state
    env.close()

if __name__ == "__main__":
    main()