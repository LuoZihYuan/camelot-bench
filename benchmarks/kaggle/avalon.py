"""Kaggle Task entry points. Install camelot-bench[kaggle] before importing.

  Cheap pilot: one candidate against four random legal-move opponents.
  Research task: supply fixed Kaggle opponent slugs, shared across candidates.
  Each result belongs to the task's llm, never all models at the table.
"""

import kaggle_benchmarks as kbench

from cambench.integrations.kaggle import SeatSpec, evaluate_episode


@kbench.task(name="camelot-random-baseline", description="Avalon candidate win rate against seeded random opponents; smoke-test baseline.")
def camelot_random_baseline(llm, seed: int = 0, candidate_seat: int = 0) -> float:
  seats = [SeatSpec(model="random") for _ in range(5)]
  if not 0 <= candidate_seat < len(seats):
    raise ValueError("candidate_seat must be between 0 and 4")
  seats[candidate_seat] = SeatSpec()
  result = evaluate_episode(llm, seats=seats, candidate_seat=candidate_seat, seed=seed, max_calls=120)
  print(f"Camelot records and metrics: {result.run_dir}")
  return result.win_rate


@kbench.task(name="camelot-fixed-roster", description="Avalon candidate win rate against an explicitly fixed Kaggle model roster.")
def camelot_fixed_roster(
  llm,
  opponent_models: list[str],
  seed: int = 0,
  candidate_seat: int = 0,
  games: int = 1,
  learn: bool = False,
  opponents_learn: bool = False,
  max_calls: int = 600,
) -> float:
  if not 4 <= len(opponent_models) <= 9:
    raise ValueError("Provide 4–9 fixed opponent model slugs for 5–10 players")
  if not 0 <= candidate_seat <= len(opponent_models):
    raise ValueError("Invalid candidate seat")
  seats = [SeatSpec(model=model, learn=opponents_learn) for model in opponent_models]
  if any(model is None or not model.strip() or model == "random" for model in opponent_models):
    raise ValueError("Fixed-roster opponents must be explicit Kaggle model slugs")
  seats.insert(candidate_seat, SeatSpec(learn=learn))
  result = evaluate_episode(llm, seats=seats, candidate_seat=candidate_seat, seed=seed,
                            games=games, max_calls=max_calls)
  print(f"Camelot records and metrics: {result.run_dir}")
  return result.win_rate
