"""Pipeline layer: the between-games training loop, persistence, and runner."""

from .runner import run
from .store import RunStore, new_run_id

__all__ = ["run", "RunStore", "new_run_id"]
