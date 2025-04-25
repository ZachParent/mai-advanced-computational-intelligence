from pathlib import Path

import torch

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = DATA_DIR / "01_results"
VIDEO_DIR = DATA_DIR / "02_videos"
AGENT_DIR = DATA_DIR / "03_agents"

USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
