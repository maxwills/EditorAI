import logging
import sys
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "service.log"

_FMT = logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def _build() -> logging.Logger:
    _LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("cad_service")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setFormatter(_FMT)
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_FMT)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


#: Shared logger — import this everywhere instead of using print().
log = _build()
