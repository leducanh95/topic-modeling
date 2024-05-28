import datetime as dt
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union

from config.constants import LOG_DIR

LOG_FORMAT = logging.Formatter(
    fmt="Log entry for %(name)s: %(asctime)s %(levelname)s %(filename)s %(module)s - %(funcName)s(%(lineno)d): %(message)s",  # noqa E501
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(
    name: str = "app",
    logdir: Union[Path, str] = LOG_DIR,
) -> logging.Logger:
    """Create a logger with file and stream handlers.

    Args:
        name (str, optional): Name of logger. Defaults to "app".
        logdir (Union[Path, str]): Folder where log files will be stored.

    Returns:
        logging.Logger: Logger object with the given name.
    """
    if logdir != LOG_DIR:
        logdir = LOG_DIR / dt.datetime.today().strftime(format="%Y-%m-%d") / name
    else:
        logdir = LOG_DIR / dt.datetime.today().strftime(format="%Y-%m-%d")

    logdir.mkdir(parents=True, exist_ok=True)
    path: Path = logdir / name

    logger: logging.Logger = logging.getLogger(name=name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    if len(list(logger.handlers)) == 0:
        logger.info("Info logger initialized for %s", name)
        logger.error("Error logger initialized for %s", name)
        logger.critical("Critical logger initialized for %s", name)

    # Remove existing file handlers (if any)
    for handler in list(logger.handlers):
        if (
            isinstance(handler, logging.FileHandler)
            or isinstance(handler, RotatingFileHandler)
            or isinstance(handler, logging.StreamHandler)
        ):
            logger.removeHandler(handler)

    # Create a handler for printing to terminal
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(fmt=LOG_FORMAT)
    logger.addHandler(hdlr=stream_handler)

    # Create file handlers for different levels and add them to the logger
    for level in [logging.INFO, logging.ERROR, logging.CRITICAL]:
        filename = f"{path.suffix}.{logging.getLevelName(level)}.log".lower()
        file_handler = RotatingFileHandler(
            filename=path.with_suffix(filename),
            maxBytes=25 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(LOG_FORMAT)
        logger.addHandler(file_handler)

    return logger
