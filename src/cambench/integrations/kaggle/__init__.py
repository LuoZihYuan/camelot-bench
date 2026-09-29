"""Kaggle Benchmarks integration (install with ``uv sync --extra kaggle``)."""

from .client import CallBudget, KaggleClient
from .episode import SeatSpec, evaluate_episode

__all__ = ["CallBudget", "KaggleClient", "SeatSpec", "evaluate_episode"]
