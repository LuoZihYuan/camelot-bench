"""Run a self-play training pipeline of N games (all learning, single model)."""

from __future__ import annotations

import argparse

from cambench import recommended_config
from cambench.config import settings
from cambench.pipeline.runner import run


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default=settings.default_model)
  ap.add_argument("--players", type=int, default=5)
  ap.add_argument("--games", type=int, default=10)
  ap.add_argument("--label", default="")
  ap.add_argument("--verbose", action="store_true", help="print each game's full log")
  args = ap.parse_args()

  roster = [{"model": args.model, "memory": None, "learn": True} for _ in range(args.players)]
  store = run(roster, recommended_config(args.players), games=args.games, label=args.label, verbose=args.verbose)

  results = store.read_results()
  wins = sum(1 for r in results if r["winner"] == "good")
  print(f"run: {store.run_id}")
  print(f"games: {len(results)}  |  good {wins}  evil {len(results) - wins}")
  print(f"records: {store.records_dir}")
  print(f"memory:  {store.memory_dir}")


if __name__ == "__main__":
  main()
