"""Unit tests for utils.uncertainty."""

from __future__ import annotations

import pytest

from utils.uncertainty import (
    UncertaintyThresholds,
    image_priority,
    label_uncertainty,
    mask_score_variance,
    sort_queue,
)


def test_variance_single_score():
    assert mask_score_variance([0.9]) == 0.0


def test_variance_spread():
    assert mask_score_variance([0.1, 0.9]) > 0.1


@pytest.fixture
def thr() -> UncertaintyThresholds:
    return UncertaintyThresholds(min_score=0.85, max_variance=0.05, min_det_conf=0.35)


def test_high_score_low_variance_is_certain(thr):
    flag, reason = label_uncertainty(
        sam_scores=[0.95, 0.94, 0.93],
        det_conf=0.8,
        thresholds=thr,
    )
    assert flag is False
    assert reason is None


def test_low_best_score_flags_uncertain(thr):
    flag, reason = label_uncertainty(
        sam_scores=[0.7, 0.72, 0.74],
        det_conf=0.8,
        thresholds=thr,
    )
    assert flag is True
    assert "low_sam_score" in reason


def test_high_variance_flags_uncertain(thr):
    flag, reason = label_uncertainty(
        sam_scores=[0.4, 0.9, 0.95],
        det_conf=0.8,
        thresholds=thr,
    )
    assert flag is True
    assert "high_variance" in reason


def test_low_det_conf_flags_uncertain(thr):
    flag, reason = label_uncertainty(
        sam_scores=[0.95],
        det_conf=0.25,
        thresholds=thr,
    )
    assert flag is True
    assert "low_det_conf" in reason


def test_image_priority_empty_is_negative():
    assert image_priority(0, 0, False) < 0


def test_image_priority_all_certain_is_zero():
    assert image_priority(0, 5, False) == 0.0


def test_image_priority_ratio():
    assert image_priority(2, 4, True) == 0.5


def test_sort_queue_orders_highest_priority_first():
    records = [
        {"name": "a", "_priority": 0.0},
        {"name": "b", "_priority": 0.8},
        {"name": "c", "_priority": 0.3},
    ]
    out = sort_queue(records)
    assert [r["name"] for r in out] == ["b", "c", "a"]
