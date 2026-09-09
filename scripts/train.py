"""Run a self-play training pipeline. Roster from --config YAML, or homogeneous from flags."""

from __future__ import annotations

import argparse

from cambench import recommended_config
from cambench.config import settings
from cambench.pipeline.roster import load_roster, resolve_seats, seat_field_present, homogeneous_seats
from cambench.pipeline.runner import run, resume
from cambench.pipeline.store import RunStore, latest_incomplete_run


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", help="YAML roster file (per-seat models etc.)")
  ap.add_argument(
    "--resume",
    nargs="?",
    const="",
    default=None,
    help="resume a run: --resume <run_id>, or bare --resume for the latest unfinished",
  )
  ap.add_argument("--model", default=settings.default_model)
  ap.add_argument("--reasoning-effort", default=settings.default_reasoning_effort)
  ap.add_argument("--players", type=int, default=settings.default_players)
  ap.add_argument("--memory", help="load this memory file into every seat")
  ap.add_argument("--no-learn", action="store_true", help="freeze memory for all seats")
  ap.add_argument("--games", type=int, default=settings.default_games)
  ap.add_argument("--label", default="")
  ap.add_argument("--win-rate-window", type=int, default=settings.win_rate_window)
  ap.add_argument("--memory-window", type=int, default=settings.memory_window)
  ap.add_argument("--verbose", action="store_true", help="stream each game live (god view)")
  args = ap.parse_args()

  # resume path: reconstruct everything from the saved run on disk
  if args.resume is not None:
    # resume takes all config from disk; only --verbose may accompany it
    stray = argv_flags() - {"--resume", "--verbose"}
    if stray:
      ap.error("on --resume the run's config comes from disk; remove: " + ", ".join(sorted(stray)))
    run_id = args.resume or latest_incomplete_run()
    if not run_id:
      ap.error("no unfinished run found to resume")
    store = resume(run_id, recommended_config(_seat_count(run_id)), verbose=args.verbose)
    _summary(store)
    return

  flags = argv_flags()
  roster = load_roster(args.config) if args.config else {}
  raw_seats = roster.get("raw_seats") or []

  if raw_seats:
    # strict, none-or-all: a CLI value conflicts if the roster also sets it.
    # per-seat flags conflict if that field appears on ANY seat (partial included).
    flag_to_field = {
      "--model": "model",
      "--reasoning-effort": "reasoning_effort",
      "--memory": "memory",
      "--no-learn": "learn",
    }
    conflicts = [field for flag, field in flag_to_field.items() if flag in flags and seat_field_present(raw_seats, field)]
    # --players conflicts with an explicit seats block (seat count comes from it)
    if "--players" in flags:
      conflicts.append("players/seats")
    # run-level
    if "--games" in flags and roster.get("games") is not None:
      conflicts.append("games")
    if "--label" in flags and roster.get("label") is not None:
      conflicts.append("label")
    if "--win-rate-window" in flags and roster.get("win_rate_window") is not None:
      conflicts.append("win_rate_window")
    if "--memory-window" in flags and roster.get("memory_window") is not None:
      conflicts.append("memory_window")
    if conflicts:
      ap.error("set in both --config and CLI (choose one place per setting): " + ", ".join(conflicts))

    flag_defaults = {
      "model": args.model if "--model" in flags else None,
      "reasoning_effort": args.reasoning_effort if "--reasoning-effort" in flags else None,
      "memory": args.memory if "--memory" in flags else None,
      "learn": (not args.no_learn) if "--no-learn" in flags else None,
    }
    flag_defaults = {k: v for k, v in flag_defaults.items() if v is not None}
    seats = resolve_seats(raw_seats, flag_defaults)
  else:
    seats = homogeneous_seats(args.model, args.players, args.memory, not args.no_learn, args.reasoning_effort)

  games = args.games if "--games" in flags else (roster.get("games") or settings.default_games)
  label = args.label if "--label" in flags else (roster.get("label") or "")
  wrw = args.win_rate_window if "--win-rate-window" in flags else roster.get("win_rate_window")
  mw = args.memory_window if "--memory-window" in flags else roster.get("memory_window")

  store = run(
    seats, recommended_config(len(seats)), games=games, label=label, win_rate_window=wrw, memory_window=mw, verbose=args.verbose
  )
  _summary(store)


def _seat_count(run_id: str) -> int:
  return len(RunStore(run_id).read_manifest().get("seats", []))


def _summary(store) -> None:
  results = store.read_results()
  wins = sum(1 for r in results if r["winner"] == "good")
  print(f"run: {store.run_id}")
  print(f"games: {len(results)}  |  good {wins}  evil {len(results) - wins}")
  print(f"records: {store.records_dir}")
  print(f"memory:  {store.memory_dir}")


def argv_flags() -> set:
  import sys

  return {a for a in sys.argv if a.startswith("--")}


if __name__ == "__main__":
  main()
