"""DEEPVISION Logger Module"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    log_file: Optional[Path] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logger:
    """
    Set up the logger for DEEPVISION
    
    Args:
        log_file: Path to log file
        level: Logging level
        format_string: Custom format string
    
    Returns:
        Configured logger instance
    """
    # Remove default handler
    logger.remove()
    
    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Console output
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=True
    )
    
    # File output
    if log_file:
        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
    
    return logger


def get_logger(name: str = "deepvision") -> logger:
    """Get a logger instance with the given name"""
    return logger.bind(name=name)


# Default logger setup
default_logger = setup_logger()