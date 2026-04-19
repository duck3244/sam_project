"""Pydantic DTOs for the auto-labeling API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SessionCreateResponse(BaseModel):
    session_id: str
    image_count: int = 0


class ImageUploadResponse(BaseModel):
    session_id: str
    image_ids: list[str]


class RunRequest(BaseModel):
    sam_model: str = Field(default="vit_b")
    det_conf: float = Field(default=0.15, ge=0.0, le=1.0)
    epsilon: float = Field(default=0.005, gt=0.0, lt=0.1)
    max_image_size: int = Field(default=1024, ge=256, le=4096)
    multi_contour: bool = Field(default=False)
    min_area_ratio: float = Field(default=0.1, gt=0.0, le=1.0)


class RunResponse(BaseModel):
    task_id: str


class LabelEntry(BaseModel):
    cls_id: int
    cls_name: str
    polygon: list[float]
    sam_score: float | None = None
    det_conf: float | None = None
    bbox: list[float] | None = None


class ImageLabels(BaseModel):
    session_id: str
    image_id: str
    width: int
    height: int
    labels: list[LabelEntry]


class RefineRequest(BaseModel):
    session_id: str
    image_id: str
    label_index: int | None = None
    bbox: list[float] | None = None
    points: list[list[float]] = Field(default_factory=list)
    point_labels: list[int] = Field(default_factory=list)
    cls_id: int | None = None


class RefineResponse(BaseModel):
    polygon: list[float]
    sam_score: float


class SessionStats(BaseModel):
    session_id: str
    image_count: int
    labeled_count: int
    total_labels: int
    class_distribution: dict[str, int]
    uncertain_queue: int = 0


class ProgressEvent(BaseModel):
    task_id: str
    done: int
    total: int
    current_image: str | None = None
    uncertain_count: int = 0
    status: str = "running"  # running | completed | failed
    error: str | None = None
