from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "01_results"
VIDEO_DIR = DATA_DIR / "02_videos"
AGENT_DIR = DATA_DIR / "03_agents"

REPORT_DIR = PROJECT_ROOT / "report"
FIGURES_DIR = REPORT_DIR / "figures"

USE_GPU = False  # torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
