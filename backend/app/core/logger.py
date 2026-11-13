import sys
from typing import Optional
from loguru import logger

from app.core.config import get_settings


def setup_logger(log_level: Optional[str] = None, log_format: Optional[str] = None):
    try:
        settings = get_settings()
        level = (log_level or settings.log_level).upper()
        fmt = (log_format or settings.log_format).lower()
    except Exception:
        level = (log_level or "INFO").upper()
        fmt = (log_format or "text").lower()

    logger.remove()
    if fmt == "json":
        logger.add(sys.stdout, serialize=True, level=level)
    else:
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=level,
        )
    return logger


log = setup_logger()
