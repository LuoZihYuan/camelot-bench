"""Per-run persistence: results ledger (manifest on line 1), records archive, agent memory."""

from __future__ import annotations

import json
import pathlib
from datetime import datetime


def new_run_id(label: str = "") -> str:
  """A sortable local-time id (with UTC offset), optionally suffixed with a label."""
  ts = datetime.now().astimezone().strftime("%y%m%d%H%M%S%z")
  return f"{ts}_{label}" if label else ts


class RunStore:
  """Owns the on-disk layout for one pipeline run under data/runs/<run_id>/."""

  def __init__(self, run_id: str, root: str = "data/runs"):
    self.run_id = run_id
    self.root = root
    self.dir = pathlib.Path(root) / run_id
    self.records_dir = self.dir / "records"
    self.memory_dir = self.dir / "memory"
    self.results_path = self.dir / "results.jsonl"
    for d in (self.records_dir, self.memory_dir):
      d.mkdir(parents=True, exist_ok=True)

  # --- manifest = line 1 of results.jsonl (immutable: roster, seed, params) ---

  def write_manifest(self, manifest: dict) -> None:
    """Write the run manifest as the first line; results append after it."""
    self.results_path.write_text(json.dumps(manifest) + "\n")

  def read_manifest(self) -> dict:
    if not self.results_path.exists():
      return {}
    with self.results_path.open(encoding="utf-8") as f:
      first = f.readline()
    return json.loads(first) if first.strip() else {}

  # --- games ---

  def save_record(self, index: int, record) -> None:
    """Write one game's full transcript to records/game_NNNN.jsonl (event per line)."""
    (self.records_dir / f"game_{index:04d}.jsonl").write_text(record.to_jsonl())

  def games_done(self) -> int:
    """Count completed games = the durable resume point / completion check."""
    return len(list(self.records_dir.glob("game_*.jsonl")))

  def append_result(self, row: dict) -> None:
    with self.results_path.open("a", encoding="utf-8") as f:
      f.write(json.dumps(row) + "\n")

  def read_results(self) -> list:
    """Per-game outcome rows (skips the manifest on line 1)."""
    if not self.results_path.exists():
      return []
    with self.results_path.open(encoding="utf-8") as f:
      lines = [ln for ln in f if ln.strip()]
    return [json.loads(ln) for ln in lines[1:]]

  # --- memory (per-seat, named by the seat's display name; overwritten each game) ---

  def save_memory(self, name: str, memory_dict: dict) -> None:
    (self.memory_dir / f"{name}.json").write_text(json.dumps(memory_dict, indent=2))

  def load_memory(self, name: str) -> dict | None:
    """This seat's saved memory, or None if it hasn't been saved yet."""
    p = self.memory_dir / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_memory_file(path: str) -> dict:
  """Load a saved Memory dict from any run's memory file (for continue-training)."""
  return json.loads(pathlib.Path(path).read_text())


def latest_incomplete_run(root: str = "data/runs") -> str | None:
  """The newest run whose completed games < target (for lazy --resume)."""
  base = pathlib.Path(root)
  if not base.exists():
    return None
  for d in sorted((d for d in base.iterdir() if d.is_dir()), reverse=True):
    results = d / "results.jsonl"
    if not results.exists():
      continue
    with results.open(encoding="utf-8") as f:
      first = f.readline()
    if not first.strip():
      continue
    target = json.loads(first).get("target_games", 0)
    done = len(list((d / "records").glob("game_*.jsonl")))
    if done < target:
      return d.name
  return None
