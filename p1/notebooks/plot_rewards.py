# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from einops import rearrange

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import RESULTS_DIR
from src.run_config import PENDULUM_CONFIGS

plt.style.use("default")
WINDOW_SIZE = 20

# %%
# plot reward over episodes
n_rows = 5
n_cols = len(PENDULUM_CONFIGS) // n_rows
fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
for i, run_config in enumerate(PENDULUM_CONFIGS):
    experiment_id = run_config.id
    try:
        results = pd.read_csv(RESULTS_DIR / f"{experiment_id:03d}.csv")
    except FileNotFoundError:
        print(f"Experiment {experiment_id} not found")
        continue

    ax = axs[i % n_rows, i // n_rows]
    moving_average = results["rewards"].rolling(window=WINDOW_SIZE).mean()

    ax.plot(results["rewards"], alpha=0.4, label="Original Rewards", color="dodgerblue")
    ax.plot(moving_average, label="Moving Average", color="blueviolet")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(f"Experiment {experiment_id}")
    ax.legend(loc="lower right")
    ax.text(
        0.5,
        0.5,
        f"Final Reward: {moving_average.iloc[-1]:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="center",
    )
plt.tight_layout()
plt.show()
# %%
hist_dfs = {}
for run_config in PENDULUM_CONFIGS:
    experiment_id = run_config.id
    try:
        hist_dfs[experiment_id] = pd.read_csv(RESULTS_DIR / f"{experiment_id:03d}.csv")
    except FileNotFoundError:
        print(f"Experiment {experiment_id} not found")
        continue
# %%

sum_of_rewards = np.array([df["rewards"].sum() for df in hist_dfs.values()])
sum_of_final_rewards = np.array(
    [df["rewards"].iloc[-WINDOW_SIZE:].sum() for df in hist_dfs.values()]
)
# TODO: remove this truncation once all data is collected
sum_of_final_rewards = sum_of_final_rewards[: (len(sum_of_final_rewards) // 5) * 5]
sum_of_final_rewards = rearrange(
    sum_of_final_rewards, "(config seed) -> config seed", seed=5
)
# %%
pval_grid = np.ones((len(PENDULUM_CONFIGS) // 5, len(PENDULUM_CONFIGS) // 5))
for i in range(len(sum_of_final_rewards)):
    for j in range(len(sum_of_final_rewards)):
        if i == j:
            pval_grid[i, j] = 1
        else:
            ttest, pval = stats.ttest_rel(
                sum_of_final_rewards[i], sum_of_final_rewards[j]
            )
            pval_grid[i, j] = pval
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(pval_grid, annot=True, cmap="PuBuGn_r", vmin=0, vmax=1, ax=ax)
ax.set_title(f"P-values of Rewards from Final {WINDOW_SIZE} Episodes")
ax.set_xlabel("Experiment")
ax.set_ylabel("Experiment")
plt.tight_layout()
plt.show()
# %%
