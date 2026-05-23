import logging
from pathlib import Path


def setup_logger(name: Path, mode: str = "a") -> logging.Logger:
    logger = logging.getLogger(name.stem)
    logger.setLevel(logging.INFO)

    # Create console handler and set level to info
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create file handler and set level to info
    fh = logging.FileHandler(f"{name}.log", mode=mode)
    fh.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add formatter to ch and fh
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Add ch and fh to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
