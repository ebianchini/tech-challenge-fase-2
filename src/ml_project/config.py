from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "online_shoppers_intention.csv"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
MLRUNS_DIR = ROOT_DIR / "mlruns"
PROCESSED_DATA_PATH = PROCESSED_DIR / "online_shoppers_processed.npz"
PROCESSED_METADATA_PATH = PROCESSED_DIR / "online_shoppers_metadata.json"

TARGET_COLUMN = "Revenue"
SEED = 42
RANDOM_STATE = SEED
TEST_SIZE = 0.2
CROSS_VALIDATION_FOLDS = 5
MIN_TARGET_CLASS_RATIO = 0.05


def set_global_seed(seed: int = SEED) -> int:
    """Define a seed global para garantir reprodutibilidade em numpy, random e sklearn."""
    random.seed(seed)
    np.random.seed(seed)
    return seed


set_global_seed(SEED)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "online-shoppers-purchasing-intention")

FEATURE_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    "TotalSessionTime",
    "TotalPagesVisited",
    "AdministrativeTimeRatio",
    "InformationalTimeRatio",
    "ProductRelatedTimeRatio",
    "AdministrativeRatio",
    "InformationalRatio",
    "ProductRelatedRatio",
]

RAW_DATASET_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    TARGET_COLUMN,
]
