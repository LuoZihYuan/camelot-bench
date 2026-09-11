"""Shared readers and small helpers for the eval metrics (seats + games)."""

from __future__ import annotations

import json
import pathlib

from ..engine.roles import alignment_of, Role


def read_ledger(run_dir: str) -> list:
  """Per-game result rows from results.jsonl (skips the manifest on line 1)."""
  path = pathlib.Path(run_dir) / "results.jsonl"
  if not path.exists():
    return []
  with path.open(encoding="utf-8") as f:
    lines = [ln for ln in f if ln.strip()]
  return [json.loads(ln) for ln in lines[1:]]


def read_records(run_dir: str) -> list:
  """(game_index, parsed_record) for every records/game_*.jsonl, in game order."""
  from ..engine.records import read_jsonl

  out = []
  for p in sorted((pathlib.Path(run_dir) / "records").glob("game_*.jsonl")):
    out.append((int(p.stem.split("_")[1]), read_jsonl(p.read_text())))
  return out


def seat_key(name: str, info: dict) -> str:
  """A run-stable seat identity: model@effort#seat."""
  return f"{info['model']}@{info.get('reasoning_effort') or 'none'}#{name}"


def seat_keys(run_dir: str) -> dict:
  """{game_index: {seat_name: "model@effort#seat"}} from the ledger."""
  out = {}
  for row in read_ledger(run_dir):
    idx = int(row["game_id"].split("_")[1])
    out[idx] = {name: seat_key(name, info) for name, info in row["seats"].items()}
  return out


def side_of(role_value: str) -> str:
  return alignment_of(Role(role_value)).value


def groups(role: str, by: str | None) -> list:
  if by == "side":
    return [side_of(role)]
  if by == "role":
    return [role]
  return ["overall"]


def wilson(wins: int, n: int, z: float = 1.96) -> tuple:
  """Wilson score interval for a proportion (lower, upper); (0, 0) if n == 0."""
  if n == 0:
    return (0.0, 0.0)
  p = wins / n
  denom = 1 + z * z / n
  center = (p + z * z / (2 * n)) / denom
  half = (z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / denom
  return (max(0.0, center - half), min(1.0, center + half))


def rolling(series: list, window: int) -> list:
  """Expanding-then-sliding mean over `series`, oldest to newest."""
  out = []
  for i in range(1, len(series) + 1):
    chunk = series[max(0, i - window) : i]
    out.append(sum(chunk) / len(chunk))
  return out
