"""
Uncertainty scoring for SAM-generated labels.

Combines multi-mask score spread and best-score threshold to decide whether
an image should be routed to the human review queue. Intentionally cheap —
no extra SAM passes required; scores come straight from predict_with_box
multimask output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class UncertaintyThresholds:
    """Image is flagged uncertain when ANY of these are true for any label."""
    min_score: float = 0.85          # best SAM score below this → uncertain
    max_variance: float = 0.05       # multimask score variance above this → uncertain
    min_det_conf: float = 0.35       # YOLO bbox confidence below this → uncertain


def mask_score_variance(scores: Iterable[float] | np.ndarray) -> float:
    arr = np.asarray(list(scores), dtype=np.float32)
    if arr.size < 2:
        return 0.0
    return float(arr.var())


def label_uncertainty(
    *,
    sam_scores: Iterable[float] | None,
    det_conf: float | None,
    thresholds: UncertaintyThresholds,
) -> tuple[bool, str | None]:
    """
    Decide whether a single label is uncertain. Returns (flag, reason).
    `sam_scores` is the full multimask score vector (length 3 for SAM1).
    """
    if sam_scores is not None:
        arr = np.asarray(list(sam_scores), dtype=np.float32)
        if arr.size > 0:
            best = float(arr.max())
            if best < thresholds.min_score:
                return True, f"low_sam_score={best:.2f}"
            var = mask_score_variance(arr)
            if var > thresholds.max_variance:
                return True, f"high_variance={var:.3f}"
    if det_conf is not None and det_conf < thresholds.min_det_conf:
        return True, f"low_det_conf={det_conf:.2f}"
    return False, None


def image_priority(n_uncertain: int, total_labels: int, any_reason: bool) -> float:
    """
    Higher score = higher review priority. Used to sort the review queue.
    Empty images (no labels) sink to the bottom; images with many uncertain
    labels float to the top.
    """
    if total_labels == 0:
        return -1.0
    if not any_reason:
        return 0.0
    return n_uncertain / max(total_labels, 1)


def sort_queue(records: list[dict], priority_key: str = "_priority") -> list[dict]:
    return sorted(records, key=lambda r: r.get(priority_key, 0.0), reverse=True)
