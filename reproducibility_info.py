import sys
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost

try:
    import lightgbm
    lightgbm_version = lightgbm.__version__
except ImportError:
    lightgbm_version = "Not installed"

BASE_DIR = Path(__file__).resolve().parent

output_dir = BASE_DIR / "results_corrected"

output_dir.mkdir(exist_ok=True)

information = {
    "Python": sys.version,
    "Operating System": platform.platform(),
    "NumPy": np.__version__,
    "Pandas": pd.__version__,
    "Scikit-learn": sklearn.__version__,
    "XGBoost": xgboost.__version__,
    "LightGBM": lightgbm_version,
    "Random Seed": 42,
    "Outer CV Folds": 5,
    "Inner Stacking CV Folds": 5,
    "Calibration CV Folds": 3,
    "Prediction Horizons": "15%, 30%, 45%",
    "Fairness Objective": "Equalized Odds gap"
}

with open(
    output_dir / "reproducibility_info.txt",
    "w",
    encoding="utf-8"
) as file:

    for key, value in information.items():
        file.write(f"{key}: {value}\n")

with open(
    output_dir / "requirements_locked.txt",
    "w",
    encoding="utf-8"
) as file:

    subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        stdout=file,
        check=True
    )

print("Reproducibility information saved successfully.")