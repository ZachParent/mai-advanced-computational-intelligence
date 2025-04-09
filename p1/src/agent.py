from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from run_config import PPOHyperparams, RunConfig
from torch.distributions import Normal

import gymnasium as gym


# --- Helper for Orthogonal Initialization ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(ABC):
    env: gym.Env

    def __init__(self, env: gym.Env):
        self.env = env

    @abstractmethod
    def act(
        self, state: np.ndarray
    ) -> tuple[torch.Tensor, np.ndarray, Optional[torch.Tensor]]:
        """Returns unclipped action tensor, clipped action numpy array, optional log prob tensor."""

    @abstractmethod
    def store_transition(
        self,
        state: np.ndarray,
        action_unclipped: torch.Tensor,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        log_prob: Optional[torch.Tensor],
    ):
        pass

    @abstractmethod
    def update(
        self,
        current_timestep: int,
        total_timesteps: int,
    ):
        pass

    @abstractmethod
    def save(self, path: Path):
        pass

    @staticmethod
    @abstractmethod
    def load(path: Path, run_config: RunConfig, env: gym.Env) -> "Agent":
        pass


class RandomAgent(Agent):
    def act(
        self, state: np.ndarray
    ) -> tuple[torch.Tensor, np.ndarray, Optional[torch.Tensor]]:
        action_np = self.env.action_space.sample()
        # For random agent, clipped and unclipped are the same, return as tensor/numpy
        action_tensor = torch.tensor(action_np, dtype=torch.float32)
        return (
            action_tensor,
            action_np,
            None,
        )  # Unclipped Tensor, Clipped Numpy, No log_prob

    def store_transition(
        self,
        state: np.ndarray,
        action_unclipped: torch.Tensor,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        log_prob: Optional[torch.Tensor],
    ):
        pass  # Random agent doesn't store

    def update(
        self,
        current_timestep: int,
        total_timesteps: int,
    ):
        pass

    def save(self, path: Path) -> None:
        pass

    @staticmethod
    def load(path: Path, run_config: RunConfig, env: gym.Env) -> "RandomAgent":
        return RandomAgent(env)


class ActorMean(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Orthogonal initialization with gain sqrt(2) for hidden layers
        # Bias 0
        self.fc1 = layer_init(nn.Linear(state_dim, 64))
        self.fc2 = layer_init(nn.Linear(64, 64))
        # Orthogonal initialization with gain 0.01 for output layer
        # Bias 0
        self.fc_mean = layer_init(nn.Linear(64, action_dim), std=0.01)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        mean = self.fc_mean(x)  # No final activation here, Tanh applied later if needed
        return mean


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        # Orthogonal initialization with gain sqrt(2) for hidden layers
        # Bias 0
        self.fc1 = layer_init(nn.Linear(state_dim, 64))
        self.fc2 = layer_init(nn.Linear(64, 64))
        # Orthogonal initialization with gain 1.0 for output layer
        # Bias 0
        self.fc_value = layer_init(nn.Linear(64, 1), std=1.0)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))  # Using tanh based on CleanRL MuJoCo examples
        x = torch.tanh(self.fc2(x))
        value = self.fc_value(x)
        return value


class PPOAgent(Agent):
    def __init__(
        self,
        run_config: RunConfig,
        env: gym.Env,
    ):
        super().__init__(env)
        self.run_config = run_config

        if self.run_config.ppo_hyperparams is None:
            raise ValueError("PPOAgent requires ppo_hyperparams in RunConfig")
        hp: PPOHyperparams = self.run_config.ppo_hyperparams  # Type hint for clarity

        # --- Calculate buffer size based on RunConfig num_steps ---
        # Assuming 1 environment for now
        num_envs = 1
        self.buffer_capacity = num_envs * self.run_config.num_steps
        if self.buffer_capacity % hp.num_minibatches != 0:
            print(
                f"Warning: Buffer capacity ({self.buffer_capacity}) not divisible by num_minibatches ({hp.num_minibatches})"
            )
        self.minibatch_size = self.buffer_capacity // hp.num_minibatches

        # --- Store hyperparameters ---
        self.clip_epsilon = hp.clip_epsilon
        self.update_epochs = hp.update_epochs
        self.gae_lambda = hp.gae_lambda
        self.vf_coef = hp.vf_coef
        self.entropy_coef = hp.entropy_coef
        self.max_grad_norm = hp.max_grad_norm
        self.use_lr_annealing = hp.use_lr_annealing

        self.buffer = deque(maxlen=self.buffer_capacity)

        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("PPOAgent only supports Box spaces.")
        self.action_low = torch.tensor(env.action_space.low, dtype=torch.float32)
        self.action_high = torch.tensor(env.action_space.high, dtype=torch.float32)

        assert env.observation_space.shape is not None, "Obs shape cannot be None"
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.actor_mean = ActorMean(state_dim, action_dim)
        self.actor_log_std = nn.Parameter(
            torch.zeros(1, action_dim)
        )  # Initialize log_std to 0
        self.critic = Critic(state_dim)

        # --- Optimizers with Adam Epsilon ---
        self.actor_optimizer = optim.Adam(
            list(self.actor_mean.parameters()) + [self.actor_log_std],
            lr=hp.actor_lr,
            eps=hp.adam_epsilon,  # Adam Epsilon
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=hp.critic_lr,
            eps=hp.adam_epsilon,  # Adam Epsilon
        )

        # Action scaling depends on the environment's action space range
        # Using Tanh in actor mean output layer assumes initial range [-1, 1]
        # Scaling to environment range happens in get_distribution
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

    def get_distribution(self, state: torch.Tensor) -> Normal:
        # Actor outputs mean in range [-1, 1] due to init/layer structure
        action_mean_normalized = self.actor_mean(state)
        # Scale mean to environment's action space range
        action_mean = action_mean_normalized * self.action_scale + self.action_bias

        log_std = self.actor_log_std.expand_as(action_mean)  # Ensure shape matches mean
        std = torch.exp(log_std)
        return Normal(action_mean, std)

    def act(self, state: np.ndarray) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
        """Returns: unclipped action tensor, clipped action numpy array, log prob tensor"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist = self.get_distribution(state_tensor)
            action_unclipped = dist.sample()  # Get the original UNCLIPPED action
            log_prob = dist.log_prob(action_unclipped).sum(
                dim=-1
            )  # Log prob of UNCLIPPED action
        # Clip the action *only* for interacting with the environment
        action_clipped_tensor = torch.clamp(
            action_unclipped, self.action_low, self.action_high
        )
        action_clipped_np = action_clipped_tensor.squeeze(0).numpy()
        # Return the UNCLIPPED tensor, the CLIPPED numpy array, and the log_prob (of unclipped)
        return action_unclipped.squeeze(0), action_clipped_np, log_prob.squeeze(0)

    def store_transition(
        self, state, action_unclipped, reward, next_state, terminated, log_prob
    ):
        """Stores the UNCLIPPED action tensor along with other data"""
        # Ensure log_prob is not None before appending
        if log_prob is None:
            # This shouldn't happen if called correctly from main loop for PPOAgent
            print(
                "Warning: Trying to store transition with None log_prob for PPOAgent."
            )
            return
        self.buffer.append(
            (state, action_unclipped, reward, next_state, terminated, log_prob)
        )

    def update(self, current_timestep: int, total_timesteps: int) -> None:
        if len(self.buffer) < self.buffer_capacity:
            # Don't update if buffer isn't full yet (wait for full rollout)
            return
        if len(self.buffer) == 0:  # Should be caught by above, but safe check
            return

        hp = self.run_config.ppo_hyperparams
        if hp is None:
            raise ValueError("Missing ppo_hyperparams")

        # --- Learning Rate Annealing ---
        if self.use_lr_annealing:
            frac = 1.0 - (current_timestep / total_timesteps)
            lr_now = frac * hp.actor_lr
            self.actor_optimizer.param_groups[0]["lr"] = lr_now
            lr_now_critic = frac * hp.critic_lr
            self.critic_optimizer.param_groups[0]["lr"] = lr_now_critic

        # --- Prepare Batch Data ---
        batch = list(self.buffer)
        # Clear buffer now that we have the data for this update cycle
        self.buffer.clear()
        states, actions_unclipped, rewards, next_states, terminateds, log_probs_old = (
            zip(*batch)
        )

        states_tensor = torch.FloatTensor(np.array(states))
        # Stack the UNCLIPPED action tensors stored in the buffer
        actions_tensor = torch.stack(actions_unclipped)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_tensor = torch.FloatTensor(np.array(next_states))
        terminateds_tensor = torch.FloatTensor(terminateds).unsqueeze(1)
        log_probs_old_tensor = torch.stack(log_probs_old).detach().view(-1)  # Flatten

        # --- GAE Calculation ---
        with torch.no_grad():
            values_old = self.critic(states_tensor).view(-1)
            next_values = self.critic(next_states_tensor).view(-1)
            advantages = torch.zeros_like(rewards_tensor).view(-1)
            last_gae_lam = 0
            for t in reversed(range(self.buffer_capacity)):
                mask = 1.0 - terminateds_tensor[t].item()
                delta = (
                    rewards_tensor[t].item()
                    + hp.gamma * next_values[t] * mask
                    - values_old[t]
                )
                advantages[t] = last_gae_lam = (
                    delta + hp.gamma * hp.gae_lambda * last_gae_lam * mask
                )
            returns_tensor = advantages + values_old

        # --- PPO Update Loop with Mini-batches ---
        batch_indices = np.arange(self.buffer_capacity)
        for epoch in range(hp.update_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, self.buffer_capacity, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = batch_indices[start:end]

                mb_states = states_tensor[mb_indices]
                # Use the UNCLIPPED actions from the buffer for log_prob calculation
                mb_actions = actions_tensor[mb_indices]
                mb_log_probs_old = log_probs_old_tensor[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns_tensor[mb_indices]
                mb_values_old = values_old[mb_indices]

                # --- Normalize Advantages (Per Mini-batch) ---
                mb_advantages_normalized = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # --- Recalculate log probabilities (using UNCLIPPED actions) ---
                dist = self.get_distribution(mb_states)
                # Calculate log_prob of the *unclipped* actions stored in the buffer
                log_probs_new = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().mean()
                values_new = self.critic(mb_states).view(-1)

                # --- Actor Loss (Clipped Surrogate Objective + Entropy Bonus) ---
                logratio = log_probs_new - mb_log_probs_old
                ratio = torch.exp(logratio)
                surr1 = ratio * mb_advantages_normalized
                surr2 = (
                    torch.clamp(ratio, 1.0 - hp.clip_epsilon, 1.0 + hp.clip_epsilon)
                    * mb_advantages_normalized
                )
                actor_loss = -torch.min(surr1, surr2).mean() - hp.entropy_coef * entropy

                # --- Value Function Loss (Clipped) ---
                v_loss_unclipped = F.mse_loss(values_new, mb_returns, reduction="none")
                v_clipped = mb_values_old + torch.clamp(
                    values_new - mb_values_old, -hp.clip_epsilon, hp.clip_epsilon
                )
                v_loss_clipped = F.mse_loss(v_clipped, mb_returns, reduction="none")
                critic_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # --- Actor Update with Grad Clipping ---
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor_mean.parameters()) + [self.actor_log_std],
                    hp.max_grad_norm,
                )
                self.actor_optimizer.step()

                # --- Critic Update with Grad Clipping ---
                self.critic_optimizer.zero_grad()
                (critic_loss * hp.vf_coef).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), hp.max_grad_norm)
                self.critic_optimizer.step()

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor_mean.state_dict(), path / "actor_mean.pth")
        torch.save(self.actor_log_std, path / "actor_log_std.pth")
        torch.save(self.critic.state_dict(), path / "critic.pth")

    @staticmethod
    def load(path: Path, run_config: RunConfig, env: gym.Env) -> "PPOAgent":
        agent = PPOAgent(run_config, env)
        agent.actor_mean.load_state_dict(torch.load(path / "actor_mean.pth"))
        agent.actor_log_std = torch.load(path / "actor_log_std.pth")
        agent.critic.load_state_dict(torch.load(path / "critic.pth"))
        return agent


def get_agent(run_config: RunConfig, env: gym.Env):
    if run_config.agent_name == "random":
        try:
            return RandomAgent(env)
        except TypeError:
            print("Warning: RandomAgent potentially missing implementations.")
            raise
    elif run_config.agent_name == "ppo":
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(f"Agent 'ppo' requires Box space")
        return PPOAgent(run_config=run_config, env=env)
    else:
        raise ValueError(f"Agent {run_config.agent_name} not found")
