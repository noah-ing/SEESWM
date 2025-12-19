"""
Logging utilities for SEESWM.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "seeswm",
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files (None = no file logging)
        console: Whether to log to console

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(log_path / f"{name}_{timestamp}.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricsLogger:
    """Simple metrics logging for training."""

    def __init__(self, log_dir: Optional[str] = None):
        self.metrics: dict[str, list] = {}
        self.log_dir = Path(log_dir) if log_dir else None

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, **kwargs) -> None:
        """Log metrics for a step."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append((step, value))

    def get(self, key: str) -> list[tuple[int, float]]:
        """Get all values for a metric."""
        return self.metrics.get(key, [])

    def save(self, filename: str = "metrics.yaml") -> None:
        """Save metrics to file."""
        if self.log_dir:
            import yaml

            with open(self.log_dir / filename, "w") as f:
                yaml.dump(self.metrics, f)
