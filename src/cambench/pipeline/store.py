"""Per-run persistence: manifest, records archive, results ledger, and portable agent memory."""

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
    self.manifest_path = self.dir / "run.json"
    for d in (self.records_dir, self.memory_dir):
      d.mkdir(parents=True, exist_ok=True)

  # --- manifest (run-level params + status) ---

  def write_manifest(self, manifest: dict) -> None:
    self.manifest_path.write_text(json.dumps(manifest, indent=2))

  def read_manifest(self) -> dict:
    return json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}

  def set_status(self, status: str) -> None:
    m = self.read_manifest()
    m["status"] = status
    self.write_manifest(m)

  # --- games ---

  def save_record(self, index: int, record) -> None:
    """Write one game's full transcript to records/game_NNNN.jsonl (event per line)."""
    (self.records_dir / f"game_{index:04d}.jsonl").write_text(record.to_jsonl())

  def games_done(self) -> int:
    """Count completed games = the durable resume point."""
    return len(list(self.records_dir.glob("game_*.jsonl")))

  def append_result(self, row: dict) -> None:
    with self.results_path.open("a", encoding="utf-8") as f:
      f.write(json.dumps(row) + "\n")

  def read_results(self) -> list:
    if not self.results_path.exists():
      return []
    with self.results_path.open(encoding="utf-8") as f:
      return [json.loads(line) for line in f if line.strip()]

  # --- memory (per-seat, overwritten each game; also the resume roster source) ---

  def save_memory(self, agent: str, memory_dict: dict) -> None:
    (self.memory_dir / f"{agent}.json").write_text(json.dumps(memory_dict, indent=2))

  def load_all_memory(self) -> dict:
    """All saved seat memories, keyed by seat letter (for resume)."""
    out = {}
    for p in sorted(self.memory_dir.glob("*.json")):
      out[p.stem] = json.loads(p.read_text())
    return out


def load_memory_file(path: str) -> dict:
  """Load a saved Memory dict from any run's memory file (for continue-training)."""
  return json.loads(pathlib.Path(path).read_text())


def latest_incomplete_run(root: str = "data/runs") -> str | None:
  """The newest run whose manifest status is not 'complete' (for lazy --resume)."""
  base = pathlib.Path(root)
  if not base.exists():
    return None
  runs = sorted((d for d in base.iterdir() if d.is_dir()), reverse=True)
  for d in runs:
    m = d / "run.json"
    if m.exists() and json.loads(m.read_text()).get("status") != "complete":
      return d.name
  return None
