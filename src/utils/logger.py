"""Structured logger utility for Assistive Vision System."""

import logging
import sys
from typing import Optional


def setup_logger(name: str = "assistive_vision", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a structured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger
