"""Per-run persistence: records archive, results ledger, and portable agent memory."""

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
    self.dir = pathlib.Path(root) / run_id
    self.records_dir = self.dir / "records"
    self.memory_dir = self.dir / "memory"
    self.checkpoints_dir = self.dir / "checkpoints"
    self.results_path = self.dir / "results.jsonl"
    for d in (self.records_dir, self.memory_dir, self.checkpoints_dir):
      d.mkdir(parents=True, exist_ok=True)

  def checkpoint_db(self) -> str:
    return str(self.checkpoints_dir / "run.sqlite")

  def save_record(self, index: int, record) -> None:
    """Write one game's full transcript to records/game_NNNN.jsonl (event per line)."""
    path = self.records_dir / f"game_{index:04d}.jsonl"
    path.write_text(record.to_jsonl())

  def append_result(self, row: dict) -> None:
    """Append one game's outcome to the results ledger (crash-safe, one line)."""
    with self.results_path.open("a", encoding="utf-8") as f:
      f.write(json.dumps(row) + "\n")

  def read_results(self) -> list:
    """Read the whole outcome ledger back (empty if none yet)."""
    if not self.results_path.exists():
      return []
    with self.results_path.open(encoding="utf-8") as f:
      return [json.loads(line) for line in f if line.strip()]

  def save_memory(self, agent: str, memory_dict: dict) -> None:
    """Write one agent's Memory (notes + window + results) to memory/<agent>.json."""
    (self.memory_dir / f"{agent}.json").write_text(json.dumps(memory_dict, indent=2))

  def load_memory(self, path: str) -> dict:
    """Load a saved Memory dict from any run's memory file (for continue-training)."""
    return json.loads(pathlib.Path(path).read_text())
