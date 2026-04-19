"""
End-to-end auto-labeling pipeline:
    image → YOLO detect → SAM box prompt → best mask → YOLO-seg label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .detector import FruitDetector, FRUIT_CLASSES
from .export import (
    DEFAULT_EPSILON,
    DEFAULT_MIN_AREA_RATIO,
    mask_to_yolo_polygon,
    mask_to_yolo_polygons,
    write_yolo_seg,
)
from .image_utils import load_image, resize_image
from .sam_predictor import SAMPredictor


@dataclass
class LabelResult:
    image_path: Path
    label_path: Path | None
    n_labels: int
    n_detections: int
    per_label: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class FruitLabelPipeline:
    """Orchestrates detector + SAM + YOLO-seg export for a single domain."""

    def __init__(
        self,
        sam: SAMPredictor,
        detector: FruitDetector,
        class_names: tuple[str, ...] = FRUIT_CLASSES,
        epsilon: float = DEFAULT_EPSILON,
        max_image_size: int = 1024,
        multi_contour: bool = False,
        min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
    ) -> None:
        self.sam = sam
        self.detector = detector
        self.class_names = class_names
        self.epsilon = epsilon
        self.max_image_size = max_image_size
        self.multi_contour = multi_contour
        self.min_area_ratio = min_area_ratio

    def process(self, image_path: str | Path, out_dir: str | Path) -> LabelResult:
        image_path = Path(image_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        label_path = out_dir / f"{image_path.stem}.txt"

        try:
            image = load_image(str(image_path))
        except Exception as exc:
            return LabelResult(image_path, None, 0, 0, error=f"load_image: {exc}")

        if self.max_image_size:
            image = resize_image(image, max_size=self.max_image_size)
        h, w = image.shape[:2]

        try:
            detections = self.detector.detect(image)
        except Exception as exc:
            return LabelResult(image_path, None, 0, 0, error=f"detect: {exc}")

        if not detections:
            label_path.write_text("", encoding="utf-8")
            return LabelResult(image_path, label_path, 0, 0)

        try:
            self.sam.set_image(image)
        except Exception as exc:
            return LabelResult(image_path, None, 0, len(detections),
                               error=f"sam.set_image: {exc}")

        entries: list[dict] = []
        per_label: list[dict] = []
        for det in detections:
            try:
                masks, scores, _ = self.sam.predict_with_box(det["bbox"])
                best_mask, best_score = self.sam.get_best_mask(masks, scores)
                if self.multi_contour:
                    polygons = mask_to_yolo_polygons(
                        best_mask, w, h,
                        epsilon=self.epsilon,
                        min_area_ratio=self.min_area_ratio,
                    )
                else:
                    single = mask_to_yolo_polygon(best_mask, w, h, epsilon=self.epsilon)
                    polygons = [single] if single is not None else []
            except Exception as exc:
                per_label.append({**det, "sam_score": None, "error": str(exc)})
                continue
            if not polygons:
                per_label.append({**det, "sam_score": float(best_score),
                                  "sam_scores": [float(s) for s in scores],
                                  "error": "empty_polygon"})
                continue
            entries.append({"cls_id": det["cls_id"], "polygons": polygons})
            per_label.append({
                **det,
                "sam_score": float(best_score),
                "sam_scores": [float(s) for s in scores],
                "polygon_points": sum(len(p) // 2 for p in polygons),
                "n_contours": len(polygons),
                "polygons": polygons,
            })

        n = write_yolo_seg(entries, label_path, w, h, epsilon=self.epsilon)
        return LabelResult(image_path, label_path, n, len(detections), per_label=per_label)

    def process_batch(
        self,
        image_paths: list[Path],
        out_dir: str | Path,
        progress_callback=None,
    ) -> list[LabelResult]:
        out_dir = Path(out_dir)
        results: list[LabelResult] = []
        for i, p in enumerate(image_paths):
            res = self.process(p, out_dir)
            results.append(res)
            if progress_callback is not None:
                progress_callback(i + 1, len(image_paths), res)
        return results
