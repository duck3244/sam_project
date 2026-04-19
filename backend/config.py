"""Backend configuration constants."""

from __future__ import annotations

from pathlib import Path

from utils.detector import DEFAULT_FRUIT_WEIGHTS, FRUIT_CLASSES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SESSIONS_DIR = OUTPUTS_DIR / "sessions"
MODELS_DIR = PROJECT_ROOT / "models"

SAM_CHECKPOINTS = {
    "vit_h": MODELS_DIR / "sam_vit_h_4b8939.pth",
    "vit_b": MODELS_DIR / "sam_vit_b_01ec64.pth",
}

DEFAULT_SAM_MODEL = "vit_b"
DEFAULT_DET_CONF = 0.15
DEFAULT_MAX_IMAGE_SIZE = 1024
DEFAULT_EPSILON = 0.005

DOMAIN_NAME = "fruit"
DOMAIN_CLASSES = list(FRUIT_CLASSES)
DOMAIN_DET_WEIGHTS = DEFAULT_FRUIT_WEIGHTS

ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def ensure_dirs() -> None:
    for d in (OUTPUTS_DIR, SESSIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)
