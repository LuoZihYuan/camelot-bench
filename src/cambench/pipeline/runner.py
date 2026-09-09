"""Run a training pipeline: build players from resolved seats, play N games, persist."""

from __future__ import annotations

import json
import pathlib

from ..agents import LLMPlayer
from ..agents.memory import Memory
from ..config import settings
from ..engine.labels import seat_label
from ..llm import LLMClient
from .loop_graph import build_pipeline
from .store import RunStore, new_run_id


def _build_memory(seat: dict, memory_window: int) -> Memory:
  """Load the seat's memory file if given (else blank), and stamp its provenance."""
  if seat.get("memory"):
    mem = Memory.from_dict(json.loads(pathlib.Path(seat["memory"]).read_text()))
  else:
    mem = Memory(window_size=memory_window)
  mem.provenance = {
    "model": seat["model"],
    "reasoning_effort": seat.get("reasoning_effort", ""),
    "learn": seat.get("learn", True),
    "memory": seat.get("memory"),
  }
  return mem


def build_players(seats: list, memory_window: int, seed_base: int = 0) -> list:
  """seats: resolved per-seat dicts {model, reasoning_effort, memory, learn}."""
  players = []
  for i, seat in enumerate(seats):
    client = LLMClient(model=seat["model"], reasoning_effort=seat.get("reasoning_effort"))
    players.append(
      LLMPlayer(client=client, memory=_build_memory(seat, memory_window), name=seat["model"], seed=seed_base * 100 + i)
    )
  return players


def run(
  seats: list,
  config,
  games: int,
  label: str = "",
  root: str = "data/runs",
  win_rate_window: int | None = None,
  memory_window: int | None = None,
  keep_checkpoint: bool = False,
  verbose: bool = False,
):
  """Play `games` games with the resolved seats; persist records, ledger, and memory."""
  wrw = win_rate_window if win_rate_window is not None else settings.win_rate_window
  mw = memory_window if memory_window is not None else settings.memory_window

  store = RunStore(new_run_id(label), root=root)
  players = build_players(seats, mw)
  learn_flags = [{"learn": s.get("learn", True)} for s in seats]

  graph = build_pipeline(players, config, store, win_rate_window=wrw, roster=learn_flags, verbose=verbose)
  graph.invoke({"index": 0, "target": games}, {"configurable": {"thread_id": store.run_id}})

  # durable outputs first, then dispose the checkpoint
  for i, p in enumerate(players):
    store.save_memory(seat_label(i), p.memory.to_dict())
  if not keep_checkpoint:
    pathlib.Path(store.checkpoint_db()).unlink(missing_ok=True)

  return store
