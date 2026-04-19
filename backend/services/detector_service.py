"""Lazy-loaded FruitDetector singleton for the backend."""

from __future__ import annotations

import threading

from utils.detector import FruitDetector
from backend import config


_lock = threading.Lock()
_instance: FruitDetector | None = None
_current_conf: float | None = None


def get_detector(conf: float = config.DEFAULT_DET_CONF) -> FruitDetector:
    global _instance, _current_conf
    with _lock:
        if _instance is None:
            _instance = FruitDetector(
                weights=config.DOMAIN_DET_WEIGHTS,
                conf=conf,
            )
            _current_conf = conf
        elif _current_conf != conf:
            _instance.conf = conf
            _current_conf = conf
    return _instance
