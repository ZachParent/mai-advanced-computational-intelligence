# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import RESULTS_DIR

# %%
experiment_id = 4

results = pd.read_csv(RESULTS_DIR / f"{experiment_id:02d}.csv")
results.columns = ["reward"]

window_size = 10  # You can adjust the window size for the moving average
moving_average = results["reward"].rolling(window=window_size).mean()
plt.plot(moving_average, label="Moving Average", color="orange")
plt.plot(results["reward"], alpha=0.5, label="Original Rewards", color="blue")
plt.legend()
plt.show()

# %%
