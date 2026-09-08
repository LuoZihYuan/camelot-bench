"""LangGraph between-games supervisor loop: setup -> play -> learn -> record -> route."""

from __future__ import annotations

import sqlite3
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph

from ..engine import AvalonGame
from ..engine.records import LiveStream
from ..agents.prompts import GAME_SEPARATOR
from ..engine.labels import seat_label
from ..engine.roles import alignment_of


class PipelineState(TypedDict, total=False):
  index: int  # games completed (the only checkpointed control data)
  target: int  # total games to play


def rolling_win_series(wins: list, window: int) -> list:
  """Rolling win-rate (expanding then sliding over `window`), oldest to newest."""
  series = []
  for i in range(1, len(wins) + 1):
    chunk = wins[max(0, i - window) : i]
    series.append(sum(chunk) / len(chunk))
  return series


def _reflection(record, seat: int) -> str:
  """This seat's debrief reflection from the record (empty if none)."""
  for ev in record.events:
    if ev.type.value == "debrief" and seat in ev.private:
      return ev.private[seat].get("reasoning", "")
  return ""


def build_pipeline(players, config, store, win_rate_window=10, roster=None, seed_base=0, verbose=False):
  """Compile the between-games graph.

  Live objects (players, store, config) are captured in closures -- never in
  graph state, which stays serializable (just index/target) for checkpointing.
  Agent memory is persisted separately by the runner (memory/*.json), not the
  checkpoint. `roster[seat]["learn"]` gates whether that seat's memory updates.
  `verbose` also prints each finished game's full log.
  """
  n = len(players)
  roster = roster or [{"learn": True} for _ in range(n)]

  def setup(state: PipelineState) -> dict:
    for p in players:
      p.win_rate_series = rolling_win_series(p.memory.wins(), win_rate_window)
    return {}

  def play_learn_record(state: PipelineState) -> dict:
    idx = state["index"] + 1
    if verbose and idx > 1:
      print(f"\n{GAME_SEPARATOR}\n", flush=True)
    live = LiveStream() if verbose else None
    game = AvalonGame(players, config=config, seed=seed_base + idx, on_event=live)
    game.play()
    game.run_debrief()
    rec = game.record
    winner = rec.outcome["winner"]

    # per-seat learning (atomic freeze: skip the whole update for frozen seats)
    for seat, p in enumerate(players):
      role = game.assignment[seat]
      side = alignment_of(role).value
      result = {"won": side == winner, "role": role.value, "side": side}
      if roster[seat].get("learn", True):
        p.memory.update(rec.render(seat), _reflection(rec, seat), result)

    # global, unconditional persistence
    store.save_record(idx, rec)
    store.append_result(
      {
        "game_id": f"game_{idx:04d}",
        "winner": winner,
        "reason": rec.outcome["reason"],
        "roles": rec.outcome["roles"],
        "seat_to_model": {seat_label(i): p.name for i, p in enumerate(players)},
      }
    )

    # live output
    if verbose:
      print(f"\n[game {idx}/{state['target']}] {winner} wins ({rec.outcome['reason']})", flush=True)
    else:
      print(f"[game {idx}/{state['target']}] {winner} wins ({rec.outcome['reason']})", flush=True)
    return {"index": idx}

  def route(state: PipelineState) -> str:
    return "done" if state["index"] >= state["target"] else "loop"

  g = StateGraph(PipelineState)
  g.add_node("setup", setup)
  g.add_node("game", play_learn_record)
  g.add_edge(START, "setup")
  g.add_edge("setup", "game")
  g.add_conditional_edges("game", route, {"loop": "setup", "done": END})

  conn = sqlite3.connect(store.checkpoint_db(), check_same_thread=False)
  return g.compile(checkpointer=SqliteSaver(conn))
