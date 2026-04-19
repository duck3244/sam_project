"""
YOLO-seg export utilities.

Converts binary segmentation masks to YOLOv8 segmentation polygon format
and writes per-image .txt labels + dataset.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import yaml


DEFAULT_EPSILON = 0.005
MIN_POLYGON_POINTS = 3
DEFAULT_MIN_AREA_RATIO = 0.1


def _contour_to_polygon(
    contour: np.ndarray, img_w: int, img_h: int, epsilon: float
) -> list[float] | None:
    if len(contour) < MIN_POLYGON_POINTS:
        return None
    perimeter = cv2.arcLength(contour, closed=True)
    approx = cv2.approxPolyDP(contour, epsilon * perimeter, closed=True)
    if len(approx) < MIN_POLYGON_POINTS:
        return None
    points = approx.reshape(-1, 2).astype(np.float64)
    points[:, 0] = np.clip(points[:, 0] / img_w, 0.0, 1.0)
    points[:, 1] = np.clip(points[:, 1] / img_h, 0.0, 1.0)
    return points.flatten().tolist()


def mask_to_yolo_polygon(
    mask: np.ndarray,
    img_w: int,
    img_h: int,
    epsilon: float = DEFAULT_EPSILON,
) -> list[float] | None:
    """
    Convert a binary mask into a single normalized YOLO-seg polygon
    using only the largest external contour.
    """
    if mask is None or mask.size == 0:
        return None

    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return None

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    return _contour_to_polygon(contour, img_w, img_h, epsilon)


def mask_to_yolo_polygons(
    mask: np.ndarray,
    img_w: int,
    img_h: int,
    epsilon: float = DEFAULT_EPSILON,
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
) -> list[list[float]]:
    """
    Convert a binary mask into one or more normalized YOLO-seg polygons,
    keeping every external contour whose area is at least
    ``min_area_ratio`` of the largest contour's area.

    Always returns a list (possibly empty). Use this when an object may be
    fragmented across disjoint regions (occlusion, partial visibility).
    """
    if mask is None or mask.size == 0:
        return []

    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas) if areas else 0.0
    if max_area <= 0:
        return []
    threshold = max_area * min_area_ratio

    polygons: list[list[float]] = []
    paired = sorted(zip(areas, contours), key=lambda p: p[0], reverse=True)
    for area, contour in paired:
        if area < threshold:
            break
        poly = _contour_to_polygon(contour, img_w, img_h, epsilon)
        if poly is not None:
            polygons.append(poly)
    return polygons


def write_yolo_seg(
    entries: Sequence[dict],
    out_path: Path | str,
    img_w: int,
    img_h: int,
    epsilon: float = DEFAULT_EPSILON,
    multi_contour: bool = False,
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
) -> int:
    """
    Write YOLO-seg label file for a single image.

    Each entry must contain one of:
      - {"cls_id": int, "polygon": [x1,y1,...]}        # already normalized, single
      - {"cls_id": int, "polygons": [[...], [...]]}    # already normalized, multi
      - {"cls_id": int, "mask": np.ndarray}            # converted from mask

    When ``multi_contour=True`` and a mask entry is given, every contour with
    area >= max_area * min_area_ratio is written as its own line under the
    same class id. Returns the total number of written lines.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for entry in entries:
        cls_id = int(entry["cls_id"])
        polygons: list[list[float]] = []

        if entry.get("polygons"):
            polygons = [list(p) for p in entry["polygons"]]
        elif entry.get("polygon") is not None:
            polygons = [list(entry["polygon"])]
        else:
            mask = entry.get("mask")
            if multi_contour:
                polygons = mask_to_yolo_polygons(
                    mask, img_w, img_h, epsilon, min_area_ratio
                )
            else:
                single = mask_to_yolo_polygon(mask, img_w, img_h, epsilon)
                if single is not None:
                    polygons = [single]

        for poly in polygons:
            if poly is None or len(poly) < MIN_POLYGON_POINTS * 2:
                continue
            coords = " ".join(f"{v:.6f}" for v in poly)
            lines.append(f"{cls_id} {coords}")

    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def write_dataset_yaml(
    out_dir: Path | str,
    class_names: Sequence[str],
    dataset_root: Path | str | None = None,
) -> Path:
    """
    Write an Ultralytics YOLO-seg dataset.yaml alongside the exported labels.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_dir / "dataset.yaml"

    root = Path(dataset_root) if dataset_root is not None else out_dir
    content = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
        "nc": len(class_names),
    }
    yaml_path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return yaml_path


def normalize_iscrowd(coco_dict: dict) -> dict:
    """
    Force iscrowd=0 on all annotations. SAM-generated COCO sometimes marks
    masks as crowd, which breaks Ultralytics convert_coco() conversion.
    """
    for ann in coco_dict.get("annotations", []):
        ann["iscrowd"] = 0
    return coco_dict
