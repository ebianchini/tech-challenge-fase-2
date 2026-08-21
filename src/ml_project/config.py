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
PROCESSED_PREPROCESSOR_PATH = PROCESSED_DIR / "preprocessor.joblib"
MODEL_PATH = MODELS_DIR / "model.joblib"
MODEL_FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"
MODEL_PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
MODEL_VERSION_INFO_PATH = MODELS_DIR / "version_info.json"
MODEL_REGISTRY_INFO_PATH = MODELS_DIR / "model_registry.json"
MODEL_REGISTRY_EVENTS_PATH = MODELS_DIR / "model_registry_events.json"

TARGET_COLUMN = "Revenue"
INFERENCE_CONTRACT_VERSION = "1.0"
MAX_PREDICTION_INSTANCES = 100
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
MLFLOW_ENABLE_MODEL_REGISTRY = os.getenv("MLFLOW_ENABLE_MODEL_REGISTRY", "true").lower() in {
    "1",
    "true",
    "yes",
}
MLFLOW_REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME")
MLFLOW_MODEL_INITIAL_STATUS = os.getenv("MLFLOW_MODEL_INITIAL_STATUS", "Staging")
MLFLOW_MODEL_APPROVAL_STATUS = os.getenv("MLFLOW_MODEL_APPROVAL_STATUS", "pending")
MLFLOW_MODEL_APPROVER = os.getenv("MLFLOW_MODEL_APPROVER", "")
MLFLOW_INFERENCE_MODEL_URI = os.getenv("MLFLOW_INFERENCE_MODEL_URI")
MLFLOW_INFERENCE_MODEL_ALIAS = os.getenv("MLFLOW_INFERENCE_MODEL_ALIAS", "Production")

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
