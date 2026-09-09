"""Run or resume a training pipeline: build players from seats, play games, persist per game."""

from __future__ import annotations

import random

from ..agents import LLMPlayer
from ..agents.memory import Memory
from ..config import settings
from ..engine.labels import seat_label, make_names, set_names
from ..llm import LLMClient
from .loop_graph import build_pipeline
from .store import RunStore, new_run_id, load_memory_file


def _stamp_provenance(mem: Memory, seat: dict) -> None:
  mem.provenance = {
    "model": seat["model"],
    "reasoning_effort": seat.get("reasoning_effort", ""),
    "learn": seat.get("learn", True),
    "memory": seat.get("memory"),
  }


def _player_from_seat(i: int, seat: dict, memory: Memory) -> LLMPlayer:
  client = LLMClient(model=seat["model"], reasoning_effort=seat.get("reasoning_effort"))
  return LLMPlayer(client=client, memory=memory, name=seat["model"], seed=i)


def build_players(seats: list, memory_window: int) -> list:
  """Fresh players from resolved seats {model, reasoning_effort, memory, learn}."""
  players = []
  for i, seat in enumerate(seats):
    if seat.get("memory"):
      mem = Memory.from_dict(load_memory_file(seat["memory"]))
    else:
      mem = Memory(window_size=memory_window)
    _stamp_provenance(mem, seat)
    players.append(_player_from_seat(i, seat, mem))
  return players


def run(
  seats: list,
  config,
  games: int,
  label: str = "",
  root: str = "data/runs",
  win_rate_window: int | None = None,
  memory_window: int | None = None,
  verbose: bool = False,
):
  """Start a new run: play `games` games with the resolved seats; persist per game."""
  wrw = win_rate_window if win_rate_window is not None else settings.win_rate_window
  mw = memory_window if memory_window is not None else settings.memory_window

  store = RunStore(new_run_id(label), root=root)
  seed_base = random.randrange(1_000_000)  # per-run: makes runs independent samples
  store.write_manifest(
    {
      "target_games": games,
      "win_rate_window": wrw,
      "memory_window": mw,
      "seed_base": seed_base,
      "seats": seats,
    }
  )
  set_names(make_names(len(seats), seed_base))  # per-run seat identities (LDDL, from seed)
  players = build_players(seats, mw)
  return _drive(store, players, config, seats, games, wrw, seed_base, start_index=0, verbose=verbose)


def resume(run_id: str, config, root: str = "data/runs", verbose: bool = False):
  """Resume an interrupted run from its manifest (roster) + saved memory + records."""
  store = RunStore(run_id, root=root)
  m = store.read_manifest()
  games = m["target_games"]
  wrw = m.get("win_rate_window", settings.win_rate_window)
  mw = m.get("memory_window", settings.memory_window)
  seed_base = m.get("seed_base", 0)
  seats = m["seats"]  # authoritative roster (ordered)

  set_names(make_names(len(seats), seed_base))  # same names as the original run
  names = [seat_label(i) for i in range(len(seats))]
  players = []
  for i, seat in enumerate(seats):
    d = store.load_memory(names[i])  # this seat's saved memory, if any
    mem = Memory.from_dict(d) if d else Memory(window_size=mw)
    _stamp_provenance(mem, seat)
    players.append(_player_from_seat(i, seat, mem))

  done = store.games_done()
  print(f"[cambench] resuming {run_id} at game {done + 1}/{games}", flush=True)
  return _drive(store, players, config, seats, games, wrw, seed_base, start_index=done, verbose=verbose)


def _drive(store, players, config, seats, games, wrw, seed_base, start_index, verbose):
  """Invoke the graph from start_index (names already installed by caller)."""
  learn_flags = [{"learn": s.get("learn", True)} for s in seats]
  graph = build_pipeline(players, config, store, win_rate_window=wrw, roster=learn_flags, seed_base=seed_base, verbose=verbose)
  graph.invoke({"index": start_index, "target": games})
  for i, p in enumerate(players):
    store.save_memory(seat_label(i), p.memory.to_dict())
  return store
