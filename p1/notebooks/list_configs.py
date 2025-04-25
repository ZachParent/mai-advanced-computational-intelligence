# %%
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import DATA_DIR
from src.run_config import CONFIGS

# %%
config_df = pd.DataFrame(CONFIGS)
config_df.set_index("id", inplace=True)

config_df["actor_lr"] = config_df["ppo_hyperparams"].map(lambda hp: hp["actor_lr"])
config_df["critic_lr"] = config_df["ppo_hyperparams"].map(lambda hp: hp["critic_lr"])

config_df = config_df[["env_name", "actor_lr", "critic_lr", "seed"]]
config_df.to_csv(DATA_DIR / "configs.csv")
config_df

# %%
