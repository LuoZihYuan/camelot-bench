"""LangGraph between-games supervisor loop: setup -> play -> learn -> record -> route."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import START, END, StateGraph

from ..engine import AvalonGame
from ..engine.records import LiveStream
from ..agents.prompts import GAME_SEPARATOR
from ..engine.labels import seat_label
from ..engine.roles import alignment_of


class PipelineState(TypedDict, total=False):
  index: int  # games completed so far
  target: int  # total games to play


def rolling_win_series(wins: list, window: int) -> list:
  """Rolling win-rate (expanding then sliding over `window`), oldest to newest."""
  series = []
  for i in range(1, len(wins) + 1):
    chunk = wins[max(0, i - window) : i]
    series.append(sum(chunk) / len(chunk))
  return series


def build_pipeline(players, config, store, win_rate_window=10, roster=None, seed_base=0, verbose=False):
  """Compile the between-games graph.

  Live objects (players, store, config) are captured in closures. State is just
  {index, target}. Each learning seat's memory is saved to memory/*.json after
  every game, so a stopped run can resume from the records + memory on disk.
  `roster[seat]["learn"]` gates whether that seat's memory updates.
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

    # per-seat learning (atomic freeze: skip the whole update for frozen seats).
    # run_debrief already added each seat's reflection to the record, so
    # rec.render(seat) includes it. Order: record result -> recompute win-rate
    # (so it includes THIS game) -> revise notes (sees fresh trajectory +
    # reflection) -> store window + curated.
    for seat, p in enumerate(players):
      if not roster[seat].get("learn", True):
        continue
      role = game.assignment[seat]
      side = alignment_of(role).value
      result = {"won": side == winner, "role": role.value, "side": side}
      log_with_reflection = rec.render(seat)

      p.memory.record_result(result)
      p.win_rate_series = rolling_win_series(p.memory.wins(), win_rate_window)
      new_notes = p.revise_notes(log_with_reflection, game._ctx(seat))
      p.memory.update_after_revise(log_with_reflection, new_notes)

    # global, unconditional persistence
    store.save_record(idx, rec)
    store.append_result(
      {
        "game_id": f"game_{idx:04d}",
        "winner": winner,
        "reason": rec.outcome["reason"],
        "roles": rec.outcome["roles"],
        "seats": {
          seat_label(i): {
            "model": p.name,
            "reasoning_effort": p.client.reasoning_effort,
          }
          for i, p in enumerate(players)
        },
      }
    )
    # per-game memory save -> resume point survives an interruption
    for seat, p in enumerate(players):
      store.save_memory(seat_label(seat), p.memory.to_dict())

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
  return g.compile()
