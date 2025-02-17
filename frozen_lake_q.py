import gymnasium as gym
import numpy as np
def main():
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode='human')
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    learning_rate = 0.9
    discount_rate = 0.9
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    num_episodes = 10
    for episode in range(num_episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if rng.random() < epsilon:
                action = rng.choice(env.action_space.n)
            else:
                action = np.argmax(q_table[state])
            next_state, reward, terminated, truncated, info = env.step(action)
            q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_rate * np.max(q_table[next_state]) - q_table[state, action])
            state = next_state
            epsilon = max(epsilon - epsilon_decay_rate, epsilon_min)
    env.close()

if __name__ == "__main__":
    main()