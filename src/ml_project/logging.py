from __future__ import annotations

from loguru import logger

logger.remove()
logger.add(
    "logs/project.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

__all__ = ["logger"]
