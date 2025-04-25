# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import RESULTS_DIR
from src.run_config import PENDULUM_CONFIGS

plt.style.use("default")

# %%
# plot reward over episodes
n_rows = 5
n_cols = len(PENDULUM_CONFIGS) // n_rows
fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
for i, run_config in enumerate(PENDULUM_CONFIGS):
    experiment_id = run_config.id
    try:
        results = pd.read_csv(RESULTS_DIR / f"{experiment_id:03d}.csv")
        results.columns = ["episode", "reward", "length"]
    except FileNotFoundError:
        print(f"Experiment {experiment_id} not found")
        continue

    ax = axs[i % n_rows, i // n_rows]
    window_size = 20  # You can adjust the window size for the moving average
    moving_average = results["reward"].rolling(window=window_size).mean()

    ax.plot(results["reward"], alpha=0.4, label="Original Rewards", color="dodgerblue")
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
