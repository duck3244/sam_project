"""
Unit tests for utils.export.

Validates YOLO-seg polygon conversion round-trip on synthetic masks:
round, rectangular, donut (with hole), and degenerate inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from utils.export import (
    mask_to_yolo_polygon,
    mask_to_yolo_polygons,
    write_yolo_seg,
    write_dataset_yaml,
    normalize_iscrowd,
)


IMG_W, IMG_H = 640, 480


def _circle_mask(cx: int, cy: int, r: int) -> np.ndarray:
    mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 1, thickness=-1)
    return mask


def _rect_mask(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def _donut_mask() -> np.ndarray:
    mask = _circle_mask(320, 240, 100)
    cv2.circle(mask, (320, 240), 40, 0, thickness=-1)
    return mask


def test_circle_mask_produces_normalized_polygon():
    mask = _circle_mask(320, 240, 80)
    poly = mask_to_yolo_polygon(mask, IMG_W, IMG_H)
    assert poly is not None
    assert len(poly) % 2 == 0
    assert len(poly) >= 6
    assert all(0.0 <= v <= 1.0 for v in poly)


def test_rect_mask_approximates_to_four_corners():
    mask = _rect_mask(100, 100, 300, 200)
    poly = mask_to_yolo_polygon(mask, IMG_W, IMG_H, epsilon=0.02)
    assert poly is not None
    # A rectangle should reduce to ~4 points after approxPolyDP
    assert 8 <= len(poly) <= 12


def test_donut_keeps_outer_contour():
    mask = _donut_mask()
    poly = mask_to_yolo_polygon(mask, IMG_W, IMG_H)
    assert poly is not None
    # Outer radius ~100 → normalized points should span a wide area
    xs = poly[::2]
    ys = poly[1::2]
    assert max(xs) - min(xs) > 0.25
    assert max(ys) - min(ys) > 0.35


def test_empty_mask_returns_none():
    mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    assert mask_to_yolo_polygon(mask, IMG_W, IMG_H) is None


def test_single_pixel_mask_returns_none():
    mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    mask[10, 10] = 1
    assert mask_to_yolo_polygon(mask, IMG_W, IMG_H) is None


def test_write_yolo_seg_roundtrip(tmp_path):
    entries = [
        {"cls_id": 0, "mask": _circle_mask(200, 200, 60)},
        {"cls_id": 3, "mask": _rect_mask(400, 100, 550, 300)},
    ]
    out = tmp_path / "img001.txt"
    n = write_yolo_seg(entries, out, IMG_W, IMG_H)
    assert n == 2
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    for line, expected_cls in zip(lines, [0, 3]):
        tokens = line.split()
        assert int(tokens[0]) == expected_cls
        coords = [float(t) for t in tokens[1:]]
        assert len(coords) % 2 == 0
        assert all(0.0 <= c <= 1.0 for c in coords)


def test_write_yolo_seg_skips_empty_entries(tmp_path):
    entries = [
        {"cls_id": 0, "mask": _circle_mask(320, 240, 50)},
        {"cls_id": 1, "mask": np.zeros((IMG_H, IMG_W), dtype=np.uint8)},
    ]
    out = tmp_path / "img002.txt"
    n = write_yolo_seg(entries, out, IMG_W, IMG_H)
    assert n == 1


def test_write_yolo_seg_empty_file_when_no_entries(tmp_path):
    out = tmp_path / "img003.txt"
    n = write_yolo_seg([], out, IMG_W, IMG_H)
    assert n == 0
    assert out.exists()
    assert out.read_text() == ""


def test_write_dataset_yaml(tmp_path):
    classes = ["apple", "banana", "orange"]
    path = write_dataset_yaml(tmp_path, classes)
    assert path.exists()
    content = path.read_text()
    assert "nc: 3" in content
    assert "apple" in content and "banana" in content and "orange" in content


def test_normalize_iscrowd():
    coco = {
        "annotations": [
            {"id": 1, "iscrowd": 1},
            {"id": 2, "iscrowd": 0},
            {"id": 3, "iscrowd": 1},
        ]
    }
    out = normalize_iscrowd(coco)
    assert all(ann["iscrowd"] == 0 for ann in out["annotations"])


def test_polygon_accepts_precomputed_polygon(tmp_path):
    entries = [
        {"cls_id": 5, "polygon": [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]},
    ]
    out = tmp_path / "img004.txt"
    n = write_yolo_seg(entries, out, IMG_W, IMG_H)
    assert n == 1
    line = out.read_text().strip()
    assert line.startswith("5 ")


def _two_circle_mask() -> np.ndarray:
    mask = _circle_mask(160, 240, 70)
    cv2.circle(mask, (480, 240), 60, 1, thickness=-1)
    return mask


def test_multi_polygons_returns_both_blobs():
    mask = _two_circle_mask()
    polys = mask_to_yolo_polygons(mask, IMG_W, IMG_H, min_area_ratio=0.1)
    assert len(polys) == 2
    for p in polys:
        assert len(p) >= 6
        assert all(0.0 <= v <= 1.0 for v in p)


def test_multi_polygons_drops_tiny_blob():
    mask = _circle_mask(160, 240, 70)
    cv2.circle(mask, (480, 240), 5, 1, thickness=-1)  # tiny secondary
    polys = mask_to_yolo_polygons(mask, IMG_W, IMG_H, min_area_ratio=0.1)
    assert len(polys) == 1


def test_singular_polygon_keeps_only_largest():
    mask = _two_circle_mask()
    poly = mask_to_yolo_polygon(mask, IMG_W, IMG_H)
    assert poly is not None
    xs = poly[::2]
    # The larger blob is centered at x=160 (norm 0.25). All points should
    # cluster near it, not span across to x=480 (norm 0.75).
    assert max(xs) < 0.5


def test_write_yolo_seg_multi_contour_writes_multiple_lines(tmp_path):
    entries = [
        {"cls_id": 7, "mask": _two_circle_mask()},
    ]
    out = tmp_path / "img005.txt"
    n = write_yolo_seg(entries, out, IMG_W, IMG_H, multi_contour=True)
    assert n == 2
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        assert line.startswith("7 ")


def test_write_yolo_seg_default_keeps_single_line_for_disjoint_mask(tmp_path):
    entries = [
        {"cls_id": 7, "mask": _two_circle_mask()},
    ]
    out = tmp_path / "img006.txt"
    n = write_yolo_seg(entries, out, IMG_W, IMG_H)  # multi_contour defaults False
    assert n == 1


def test_write_yolo_seg_accepts_polygons_plural(tmp_path):
    entries = [
        {
            "cls_id": 2,
            "polygons": [
                [0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2],
                [0.7, 0.7, 0.8, 0.7, 0.8, 0.8, 0.7, 0.8],
            ],
        },
    ]
    out = tmp_path / "img007.txt"
    n = write_yolo_seg(entries, out, IMG_W, IMG_H)
    assert n == 2
    lines = out.read_text().strip().splitlines()
    assert all(line.startswith("2 ") for line in lines)


def test_mask_to_yolo_polygons_empty_mask_returns_empty_list():
    mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    assert mask_to_yolo_polygons(mask, IMG_W, IMG_H) == []
