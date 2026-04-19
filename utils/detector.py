"""
YOLOv8 pre-detector wrapper for the fruit auto-labeling pipeline.

Loads the fruit_detection best.pt from the sibling project by default
and returns bounding boxes ready to be fed into SAM as box prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_FRUIT_WEIGHTS = Path(
    "/home/duck/PycharmProjects/sam_project"
    "/runs/detect/runs/detect/fruit10_merged_v2b_imgsz1280/weights/best.pt"
)

FRUIT_CLASSES: tuple[str, ...] = (
    "apple", "banana", "orange", "strawberry", "grape",
    "pineapple", "watermelon", "mango", "peach", "cherry",
)


class FruitDetector:
    """Thin wrapper around Ultralytics YOLO for fruit detection."""

    CLASSES = FRUIT_CLASSES

    def __init__(
        self,
        weights: str | Path = DEFAULT_FRUIT_WEIGHTS,
        conf: float = 0.15,
        iou: float = 0.5,
        device: str | None = None,
        imgsz: int = 640,
    ) -> None:
        from ultralytics import YOLO
        import torch

        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {weights_path}. "
                "Train fruit_detection or pass a valid path."
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.model = YOLO(str(weights_path))

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        Run detection on an RGB numpy image.

        Returns a list of dicts:
            {"bbox": [x1,y1,x2,y2], "cls_id": int, "cls_name": str, "conf": float}
        Coordinates are in absolute pixel space of the input image.
        """
        if image is None or image.size == 0:
            return []

        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        detections: list[dict] = []
        for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, cls_ids, confs):
            name = self.CLASSES[cls_id] if cls_id < len(self.CLASSES) else str(cls_id)
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "cls_id": int(cls_id),
                "cls_name": name,
                "conf": float(conf),
            })
        return detections

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[dict]]:
        return [self.detect(img) for img in images]
