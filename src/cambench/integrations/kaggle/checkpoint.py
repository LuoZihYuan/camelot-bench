"""Atomic per-game commits and derived files compatible with cambench.eval."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def atomic_text(path: Path, text: str):
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as out:
      out.write(text)
      out.flush()
      os.fsync(out.fileno())
    os.replace(temporary, path)
  finally:
    if os.path.exists(temporary):
      os.unlink(temporary)


def atomic_json(path: Path, value):
  atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


@contextmanager
def exclusive_run(path: Path):
  """Advisory lock on Linux (Kaggle) and macOS; released even after a crash."""
  import fcntl

  path.mkdir(parents=True, exist_ok=True)
  with (path / ".lock").open("a") as handle:
    try:
      fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
      raise RuntimeError("Another evaluation is using this run directory") from exc
    try:
      yield
    finally:
      fcntl.flock(handle, fcntl.LOCK_UN)


class Checkpoints:
  def __init__(self, path: Path, manifest: dict, resume: bool):
    self.path = path
    self.manifest = manifest
    self.directory = path / "checkpoints"
    self.directory.mkdir(exist_ok=True)
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
      if not resume:
        raise ValueError("Run already exists; use resume=True or a new run directory")
      if json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Resume configuration/model/memory differs from the original run")
    else:
      if resume:
        raise ValueError("Cannot resume: no manifest exists")
      atomic_json(manifest_path, manifest)
    self.games = []
    for index, file in enumerate(sorted(self.directory.glob("game_*.json")), 1):
      if file.name != f"game_{index:04d}.json":
        raise ValueError("Checkpoint sequence has a gap")
      self.games.append(json.loads(file.read_text()))
    self.export()

  def usage(self) -> int:
    file = self.path / "usage.json"
    used = json.loads(file.read_text())["calls"] if file.exists() else 0
    if type(used) is not int or used < 0:
      raise ValueError("Invalid saved call usage")
    return used

  def save_usage(self, calls: int):
    atomic_json(self.path / "usage.json", {"calls": calls})

  def commit(self, game: dict):
    atomic_json(self.directory / f"game_{len(self.games) + 1:04d}.json", game)
    self.games.append(game)
    self.export()

  def export(self):
    """Recover derived files from authoritative atomic commits on every resume."""
    rows = [self.manifest] + [game["result"] for game in self.games]
    atomic_text(self.path / "results.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
    for index, game in enumerate(self.games, 1):
      atomic_text(self.path / "records" / f"game_{index:04d}.jsonl", game["record"])
    if self.games:
      for name, memory in zip(self.manifest["names"], self.games[-1]["memories"]):
        atomic_json(self.path / "memory" / f"{name}.json", memory)
