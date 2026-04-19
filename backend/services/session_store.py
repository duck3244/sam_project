"""In-memory session + task store.

Persists nothing across restarts — sessions live under outputs/sessions on disk
for image files and labels, but mutable per-label state is kept in memory.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend import config


@dataclass
class ImageRecord:
    image_id: str
    path: Path
    width: int = 0
    height: int = 0
    labels: list[dict] = field(default_factory=list)
    uncertain: bool = False
    reviewed: bool = False
    priority: float = 0.0


@dataclass
class Session:
    session_id: str
    root: Path
    images: dict[str, ImageRecord] = field(default_factory=dict)
    created_at: float = 0.0

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels"


@dataclass
class Task:
    task_id: str
    session_id: str
    total: int = 0
    done: int = 0
    current_image: str | None = None
    uncertain_count: int = 0
    status: str = "pending"
    error: str | None = None


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._tasks: dict[str, Task] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._lock = threading.Lock()

    def create_session(self) -> Session:
        import time
        sid = uuid.uuid4().hex[:12]
        root = config.SESSIONS_DIR / sid
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)
        with self._lock:
            session = Session(session_id=sid, root=root, created_at=time.time())
            self._sessions[sid] = session
        return session

    def get_session(self, sid: str) -> Session | None:
        return self._sessions.get(sid)

    def require_session(self, sid: str) -> Session:
        s = self.get_session(sid)
        if s is None:
            raise KeyError(f"session not found: {sid}")
        return s

    def add_image(self, sid: str, path: Path, width: int, height: int) -> ImageRecord:
        session = self.require_session(sid)
        image_id = uuid.uuid4().hex[:10]
        rec = ImageRecord(image_id=image_id, path=path, width=width, height=height)
        with self._lock:
            session.images[image_id] = rec
        return rec

    def list_images(self, sid: str, sort_by_priority: bool = False) -> list[ImageRecord]:
        session = self.require_session(sid)
        records = list(session.images.values())
        if sort_by_priority:
            records.sort(key=lambda r: (r.priority, r.uncertain), reverse=True)
        return records

    def get_image(self, sid: str, image_id: str) -> ImageRecord:
        session = self.require_session(sid)
        rec = session.images.get(image_id)
        if rec is None:
            raise KeyError(f"image not found: {image_id}")
        return rec

    def set_labels(self, sid: str, image_id: str, labels: list[dict]) -> None:
        rec = self.get_image(sid, image_id)
        with self._lock:
            rec.labels = labels

    def mark_uncertain(self, sid: str, image_id: str, uncertain: bool) -> None:
        rec = self.get_image(sid, image_id)
        with self._lock:
            rec.uncertain = uncertain

    def create_task(self, sid: str, total: int) -> Task:
        tid = uuid.uuid4().hex[:12]
        task = Task(task_id=tid, session_id=sid, total=total, status="running")
        with self._lock:
            self._tasks[tid] = task
        return task

    def get_task(self, tid: str) -> Task | None:
        return self._tasks.get(tid)

    def subscribe(self, tid: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers[tid].append(q)
        return q

    def unsubscribe(self, tid: str, q: asyncio.Queue) -> None:
        if tid in self._subscribers and q in self._subscribers[tid]:
            self._subscribers[tid].remove(q)

    async def publish(self, tid: str, payload: dict[str, Any]) -> None:
        for q in list(self._subscribers.get(tid, [])):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass


store = SessionStore()
