"""Run a training pipeline: build players from a roster, play N games, persist, resume-safe."""

from __future__ import annotations

import pathlib

from ..agents import LLMPlayer
from ..agents.memory import Memory
from ..config import settings
from ..engine.labels import seat_label
from ..llm import LLMClient
from .loop_graph import build_pipeline
from .store import RunStore, new_run_id


def build_players(roster: list, seed_base: int = 0) -> list:
  """roster: per-seat dicts {model, memory (path or None), learn}. Returns LLMPlayers."""
  players = []
  for i, seat in enumerate(roster):
    client = LLMClient(model=seat.get("model"))
    mem_path = seat.get("memory")
    memory = Memory.from_dict(_read_json(mem_path)) if mem_path else Memory()
    players.append(
      LLMPlayer(client=client, memory=memory, name=seat.get("model") or settings.default_model, seed=seed_base * 100 + i)
    )
  return players


def run(
  roster: list,
  config,
  games: int,
  label: str = "",
  root: str = "data/runs",
  keep_checkpoint: bool = False,
  verbose: bool = False,
):
  """Play `games` games with the roster; persist records, ledger, and agent memory."""
  store = RunStore(new_run_id(label), root=root)
  players = build_players(roster)
  learn_flags = [{"learn": s.get("learn", True)} for s in roster]

  graph = build_pipeline(players, config, store, win_rate_window=settings.win_rate_window, roster=learn_flags, verbose=verbose)
  cfg = {"configurable": {"thread_id": store.run_id}}

  graph.invoke({"index": 0, "target": games}, cfg)

  # durable outputs first, then dispose the checkpoint
  for i, p in enumerate(players):
    store.save_memory(seat_label(i), p.memory.to_dict())
  if not keep_checkpoint:
    pathlib.Path(store.checkpoint_db()).unlink(missing_ok=True)

  return store


def _read_json(path):
  import json

  return json.loads(pathlib.Path(path).read_text())
