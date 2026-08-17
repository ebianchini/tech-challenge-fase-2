from src.ml_project.config import SEED, set_global_seed
from src.ml_project.logging import logger


def test_seed_is_fixed() -> None:
    assert SEED == 42
    assert callable(set_global_seed)


def test_logger_is_available() -> None:
    assert hasattr(logger, "info")
    assert hasattr(logger, "warning")
    assert hasattr(logger, "error")
