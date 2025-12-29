from pathlib import Path

import torch


PROJECT_NAME = "animals-detection"
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_PATH = PROJECT_ROOT / "results"
DATA_YAML_RELATIVE = PROJECT_ROOT / "data/data.yaml"

def get_device():
    if torch.cuda.is_available():
        return 0
    return "cpu"