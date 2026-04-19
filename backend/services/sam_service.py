"""Lazy-loaded SAM predictor cache.

The SAMPredictor object is expensive to create (loads a multi-GB ViT checkpoint),
so one instance per model_type is kept alive. `set_image` state is not cached
across images here; refinement endpoints re-run set_image on demand.
"""

from __future__ import annotations

import threading

from utils.sam_predictor import SAMPredictor
from backend import config


_lock = threading.Lock()
_predictors: dict[str, SAMPredictor] = {}


def get_sam(model_type: str = config.DEFAULT_SAM_MODEL) -> SAMPredictor:
    if model_type not in config.SAM_CHECKPOINTS:
        raise ValueError(f"unknown SAM model_type: {model_type}")
    with _lock:
        pred = _predictors.get(model_type)
        if pred is None:
            pred = SAMPredictor(
                model_type=model_type,
                checkpoint_path=str(config.SAM_CHECKPOINTS[model_type]),
            )
            _predictors[model_type] = pred
    return pred
