# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from einops import rearrange
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import DATA_DIR, RESULTS_DIR
from src.run_config import ANT_CONFIGS

plt.style.use("default")
WINDOW_SIZE = 20
configs = ANT_CONFIGS

# %%
# plot reward over episodes
n_rows = 5
n_cols = len(configs) // n_rows
fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
for i, run_config in enumerate(configs):
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
for run_config in configs:
    experiment_id = run_config.id
    try:
        hist_dfs[experiment_id] = pd.read_csv(RESULTS_DIR / f"{experiment_id:03d}.csv")
    except FileNotFoundError:
        print(f"Experiment {experiment_id} not found")
        continue
# %%
# calculate reward statistics
mean_of_rewards = np.array([df["rewards"].mean() for df in hist_dfs.values()])
mean_of_final_rewards = np.array(
    [df["rewards"].iloc[-WINDOW_SIZE:].mean() for df in hist_dfs.values()]
)
# TODO: remove this truncation once all data is collected
mean_of_final_rewards = mean_of_final_rewards[: (len(mean_of_final_rewards) // 5) * 5]
mean_of_final_rewards = rearrange(
    mean_of_final_rewards, "(config seed) -> config seed", seed=5
)

# %%
# plot p-values and diff of means
alpha = 0.05
pval_grid = np.ones((len(configs) // 5, len(configs) // 5))
diff_of_means_grid = np.zeros((len(configs) // 5, len(configs) // 5))
significant_grid = np.zeros((len(configs) // 5, len(configs) // 5))
for i in range(len(mean_of_final_rewards)):
    for j in range(len(mean_of_final_rewards)):
        if i == j:
            pval_grid[i, j] = 1
        else:
            ttest, pval = stats.ttest_ind(
                mean_of_final_rewards[i], mean_of_final_rewards[j]
            )
            if not isinstance(pval, np.float64):
                raise ValueError(f"pval is not a float: {pval}")
            pval_grid[i, j] = pval
            if pval < alpha:
                significant_grid[i, j] = 1
            diff_of_means = np.mean(mean_of_final_rewards[i]) - np.mean(
                mean_of_final_rewards[j]
            )
            diff_of_means_grid[i, j] = diff_of_means
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    pval_grid,
    annot=True,
    cmap="PuBuGn_r",
    vmin=0,
    vmax=1,
    ax=ax,
    linewidths=0.5,
    linecolor="lightgrey",
)
for (i, j), val in np.ndenumerate(significant_grid):
    if val == 1:
        rect = Rectangle(
            (j + 0.02, i + 0.02),
            0.96,
            0.96,
            fill=False,
            edgecolor="mediumseagreen",
            linewidth=2,
        )
        ax.add_patch(rect)
ax.set_title(f"P-values of Rewards from Final {WINDOW_SIZE} Episodes")
ax.set_xlabel("Experiment")
ax.set_ylabel("Experiment")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(diff_of_means_grid, annot=True, cmap="bwr", ax=ax, center=0)
for (i, j), val in np.ndenumerate(significant_grid):
    if val == 1:
        rect = Rectangle(
            (j + 0.02, i + 0.02),
            0.96,
            0.96,
            fill=False,
            edgecolor="mediumseagreen",
            linewidth=2,
        )
        ax.add_patch(rect)
ax.set_title(
    f"Advantage of Exp A over Exp B's Rewards from Final {WINDOW_SIZE} Episodes"
)
ax.set_xlabel("Experiment B")
ax.set_ylabel("Experiment A")
plt.tight_layout()
plt.show()
# %%
all_configs_df = pd.read_csv(DATA_DIR / "configs.csv")
all_configs_df.set_index("id", inplace=True)
configs_df = all_configs_df.iloc[list(hist_dfs.keys())].copy()

configs_df["final_reward"] = [
    hist_dfs[id]["rewards"][-WINDOW_SIZE:].mean() for id in configs_df.index
]

independent_var = "actor_lr"
independent_vals = configs_df[independent_var].unique()
independent_vals_str = [str(val) for val in independent_vals]

pval_grid = np.ones((len(independent_vals), len(independent_vals)))
diff_of_means_grid = np.zeros((len(independent_vals), len(independent_vals)))
significant_grid = np.zeros((len(independent_vals), len(independent_vals)))

groups = {}
for val in independent_vals:
    groups[val] = configs_df[configs_df[independent_var] == val]

for i in range(len(independent_vals)):
    for j in range(len(independent_vals)):
        if i == j:
            pval_grid[i, j] = 1
        else:
            ttest, pval = stats.ttest_rel(
                groups[independent_vals[i]]["final_reward"],
                groups[independent_vals[j]]["final_reward"],
            )
            if not isinstance(pval, np.float64):
                raise ValueError(f"pval is not a float: {pval}")
            pval_grid[i, j] = pval
            if pval < alpha:
                significant_grid[i, j] = 1
            diff_of_means = np.mean(
                groups[independent_vals[i]]["final_reward"]
            ) - np.mean(groups[independent_vals[j]]["final_reward"])
            diff_of_means_grid[i, j] = diff_of_means

# %%
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    pval_grid,
    annot=True,
    cmap="PuBuGn_r",
    ax=ax,
    vmin=0,
    vmax=1,
    xticklabels=independent_vals_str,
    yticklabels=independent_vals_str,
)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    diff_of_means_grid,
    annot=True,
    cmap="bwr",
    ax=ax,
    center=0,
    xticklabels=independent_vals_str,
    yticklabels=independent_vals_str,
)
# %%
fig, ax = plt.subplots(figsize=(7, 6))
sns.boxenplot(
    data=configs_df,
    x="actor_lr",
    y="final_reward",
    hue="critic_lr",
    palette="Set2",
    gap=0.2,
    k_depth=4,
    ax=ax,
)
ax.set_title("Final Reward by Actor and Critic Learning Rates")
ax.set_xlabel("Actor Learning Rate")
ax.set_ylabel("Final Reward")
ax.legend(title="Critic Learning Rate")
plt.tight_layout()
plt.show()

# %%
