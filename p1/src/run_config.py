import itertools
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class PPOHyperparams:
    actor_lr: float = 3e-4  # Common starting point for MuJoCo
    critic_lr: float = 1e-3  # Often higher than actor
    gamma: float = 0.99
    clip_epsilon: float = 0.2  # PPO clip range
    update_epochs: int = 10  # Epochs per PPO update
    gae_lambda: float = 0.95  # Lambda for GAE
    num_minibatches: int = 32  # Number of minibatches per PPO update epoch
    buffer_capacity: int = (
        2048  # Steps collected per update (often num_envs * num_steps_per_env)
    )

    # New hyperparameters based on principles
    adam_epsilon: float = 1e-5  # Adam epsilon parameter
    use_lr_annealing: bool = True  # Enable/disable LR annealing
    entropy_coef: float = 0.0  # Entropy bonus coefficient (often 0.0 for continuous)
    vf_coef: float = 0.5  # Value function loss coefficient
    max_grad_norm: float = 0.5  # Global gradient clipping threshold
    # Note: buffer_capacity should ideally be divisible by num_minibatches


@dataclass(frozen=True)
class RunConfig:
    id: int
    name: str
    env_name: Literal["Pendulum-v1", "InvertedPendulum-v5", "Ant-v5", "Humanoid-v5"]
    agent_name: Literal["random", "ppo"]
    seed: Optional[int] = None
    total_timesteps: int = 1_000_000  # Define total steps for training completion
    num_steps: int = (
        2048  # Steps per environment per rollout (renamed from buffer_capacity in PPOHyperparams)
    )
    max_episode_steps: Optional[int] = None
    seed: Optional[int] = None
    record_episode_spacing: Optional[int] = None
    ppo_hyperparams: Optional[PPOHyperparams] = None

    def __post_init__(self):
        if self.agent_name == "ppo":
            if self.ppo_hyperparams is None:
                raise ValueError("ppo_hyperparams must be provided for PPO agent")
            # Set buffer capacity based on num_steps (assuming 1 environment for now)
            # If using vectorized envs, this would be num_envs * num_steps
            num_envs = 1
            actual_buffer_capacity = num_envs * self.num_steps
            if self.ppo_hyperparams.buffer_capacity != actual_buffer_capacity:
                print(
                    f"Warning: Overriding ppo_hyperparams.buffer_capacity ({self.ppo_hyperparams.buffer_capacity}) "
                    f"with calculated value based on RunConfig.num_steps: {actual_buffer_capacity}"
                )
                # Pydantic models are immutable, need to create new one or handle differently.
                # For simplicity here, we'll assume they match or the user sets buffer_capacity correctly.
                # A better way is to remove buffer_capacity from PPOHyperparams and calculate in agent.
            if actual_buffer_capacity % self.ppo_hyperparams.num_minibatches != 0:
                print(
                    f"Warning: Buffer capacity ({actual_buffer_capacity}) is not divisible by num_minibatches ({self.ppo_hyperparams.num_minibatches})."
                )

        if self.agent_name != "ppo" and self.ppo_hyperparams is not None:
            print(
                f"Warning: ppo_hyperparams provided for non-PPO agent '{self.agent_name}', will be ignored."
            )


id_start = 0
PENDULUM_CONFIGS = [
    RunConfig(
        id=id_start + i,
        name=f"Pendulum-v1-ppo actor_lr: {actor_lr} critic_lr: {critic_lr} seed: {seed}",
        env_name="Pendulum-v1",
        agent_name="ppo",
        record_episode_spacing=500,
        total_timesteps=200_000,
        max_episode_steps=200,
        seed=seed,
        ppo_hyperparams=PPOHyperparams(
            actor_lr=actor_lr,
            critic_lr=critic_lr,
        ),
    )
    for i, (actor_lr, critic_lr, seed) in enumerate(
        itertools.product(
            [1e-3, 3e-4, 1e-4],
            [3e-3, 1e-3, 3e-4],
            [42, 43, 44, 45, 46],
        )
    )
]

id_start += len(PENDULUM_CONFIGS)

INVERTED_PENDULUM_CONFIGS = [
    RunConfig(
        id=id_start + i,
        name=f"InvertedPendulum-v5-ppo actor_lr: {actor_lr} critic_lr: {critic_lr} seed: {seed}",
        env_name="InvertedPendulum-v5",
        agent_name="ppo",
        record_episode_spacing=500,
        total_timesteps=200_000,
        max_episode_steps=200,
        seed=seed,
        ppo_hyperparams=PPOHyperparams(
            actor_lr=actor_lr,
            critic_lr=critic_lr,
        ),
    )
    for i, (actor_lr, critic_lr, seed) in enumerate(
        itertools.product(
            [1e-3, 3e-4, 1e-4],
            [3e-3, 1e-3, 3e-4],
            [42, 43, 44, 45, 46],
        )
    )
]

id_start += len(INVERTED_PENDULUM_CONFIGS)

ANT_CONFIGS = [
    RunConfig(
        id=id_start + i,
        name=f"Ant-v5-ppo actor_lr: {actor_lr} critic_lr: {critic_lr} seed: {seed}",
        env_name="Ant-v5",
        agent_name="ppo",
        record_episode_spacing=500,
        total_timesteps=800_000,
        max_episode_steps=400,
        seed=seed,
        ppo_hyperparams=PPOHyperparams(
            actor_lr=actor_lr,
            critic_lr=critic_lr,
        ),
    )
    for i, (actor_lr, critic_lr, seed) in enumerate(
        itertools.product(
            [1e-3, 3e-4, 1e-4],
            [3e-3, 1e-3, 3e-4],
            [42, 43, 44, 45, 46],
        )
    )
]

id_start += len(ANT_CONFIGS)

HUMANOID_CONFIGS = [
    RunConfig(
        id=id_start + i,
        name=f"Humanoid-v5-ppo actor_lr: {actor_lr} critic_lr: {critic_lr} seed: {seed}",
        env_name="Humanoid-v5",
        agent_name="ppo",
        record_episode_spacing=500,
        total_timesteps=1_000_000,
        max_episode_steps=400,
        seed=seed,
        ppo_hyperparams=PPOHyperparams(
            actor_lr=actor_lr,
            critic_lr=critic_lr,
        ),
    )
    for i, (actor_lr, critic_lr, seed) in enumerate(
        itertools.product(
            [1e-3, 3e-4, 1e-4],
            [3e-3, 1e-3, 3e-4],
            [42, 43, 44, 45, 46],
        )
    )
]

CONFIGS = (
    PENDULUM_CONFIGS + INVERTED_PENDULUM_CONFIGS + ANT_CONFIGS
)  # + HUMANOID_CONFIGS

if __name__ == "__main__":
    print(f"PENDULUM_CONFIGS: {len(PENDULUM_CONFIGS)}")
    print(f"INVERTED_PENDULUM_CONFIGS: {len(INVERTED_PENDULUM_CONFIGS)}")
    print(f"ANT_CONFIGS: {len(ANT_CONFIGS)}")
    # print(f"HUMANOID_CONFIGS: {len(HUMANOID_CONFIGS)}")
    print()
    print(f"Total CONFIGS: {len(CONFIGS)}")
