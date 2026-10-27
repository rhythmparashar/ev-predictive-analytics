"""
Centralised logger factory for the EV analytics pipeline.

Every module gets a named logger:
    from configs.logging_config import get_logger
    logger = get_logger(__name__)

Log level is controlled by the LOG_LEVEL env var (default INFO).
In production, set LOG_LEVEL=WARNING to reduce verbosity.
"""

from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.propagate = False
    return logger
