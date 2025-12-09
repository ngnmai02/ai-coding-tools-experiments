"""Project-wide configuration constants and helper paths."""
from __future__ import annotations

from pathlib import Path

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

for directory in (MODELS_DIR, FIGURES_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
