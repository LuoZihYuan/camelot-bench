"""Sequential Avalon trajectories, with one explicitly scored candidate seat."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from ...agents import LLMPlayer
from ...agents.memory import Memory
from ...engine import AvalonGame, RandomBot
from ...engine.labels import make_names, using_names
from ...engine.roles import GameConfig, alignment_of, recommended_config
from ...eval import seats as metrics
from ...eval._io import rolling, seat_key
from .checkpoint import Checkpoints, atomic_json, exclusive_run
from .client import CallBudget, KaggleClient, require_kaggle_proxy


@dataclass(frozen=True)
class SeatSpec:
  """None uses the evaluated LLM; 'random' uses an offline legal-move baseline.

  Other model strings are Kaggle model slugs. Memory is a saved Memory.to_dict()
  value, copied per seat. No personal provider API keys are accepted.
  """

  model: str | None = None
  learn: bool = False
  reasoning_effort: str = ""
  memory: dict | None = None

  def __post_init__(self):
    if self.model == "random" and (self.learn or self.memory or self.reasoning_effort):
      raise ValueError("Random baseline seats cannot learn or load LLM memory/settings")
    if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
      raise ValueError("model must be None, 'random', or a Kaggle model slug")
    if type(self.learn) is not bool:
      raise ValueError("learn must be a bool")
    if self.reasoning_effort not in ("", "low", "medium", "high"):
      raise ValueError("Invalid reasoning_effort")


@dataclass
class EpisodeResult:
  run_dir: Path
  report: dict

  @property
  def win_rate(self) -> float:
    return self.report["candidate"]["win_rate"]["overall"]["win_rate"]


def _tuple_tree(value):
  return tuple(_tuple_tree(v) for v in value) if isinstance(value, list) else value


def _source_digest():
  """Invalidate checkpoints after a change to rules, agents, metrics, or this harness."""
  root = Path(__file__).resolve().parents[2]
  digest = hashlib.sha256()
  for folder in ("engine", "agents", "eval", "integrations/kaggle"):
    for file in sorted((root / folder).glob("*.py")):
      digest.update(str(file.relative_to(root)).encode())
      digest.update(file.read_bytes())
  return digest.hexdigest()


def _make_players(llm, seats, names, memory_window, budget, max_tokens):
  from kaggle_benchmarks.kaggle.models import load_model

  players = []
  for index, (spec, name) in enumerate(zip(seats, names)):
    memory = Memory.from_dict(copy.deepcopy(spec.memory)) if spec.memory is not None else Memory(window_size=memory_window)
    if memory.window_size < 1 or memory.notes_chars < 1:
      raise ValueError("Memory window and notes length must be positive")
    if spec.model == "random":
      player = RandomBot(seed=index)
      player.memory = memory
      player.name = "random"
      player.client = SimpleNamespace(reasoning_effort="")
    else:
      model = llm if spec.model is None else load_model(spec.model)
      client = KaggleClient(model, budget, spec.reasoning_effort, max_tokens)
      player = LLMPlayer(client=client, memory=memory, name=client.model, seed=index)
    memory.provenance = {
      "model": player.name, "reasoning_effort": spec.reasoning_effort,
      "learn": spec.learn, "memory": "embedded" if spec.memory is not None else None,
    }
    player.win_rate_series = []
    players.append(player)
  return players


def _report(path, candidate_key, calls):
  values = {
    "win_rate": metrics.win_rate(str(path)),
    "win_rate_by_role": metrics.win_rate(str(path), by="role"),
    "belief_accuracy": metrics.belief_accuracy(str(path)),
    "hidden_rate": metrics.hidden_rate(str(path)),
    "assassination": metrics.assassination(str(path)),
    "role_counts": metrics.role_counts(str(path)),
    "win_rate_trajectory": metrics.win_rate_trajectory(str(path)),
    "belief_accuracy_trajectory": metrics.belief_accuracy_trajectory(str(path)),
    "hidden_rate_trajectory": metrics.hidden_rate_trajectory(str(path)),
  }
  report = {
    "candidate_key": candidate_key, "calls_attempted": calls,
    "candidate": {name: value.get(candidate_key, {}) for name, value in values.items()},
    "all_seats": values,
  }
  atomic_json(path / "metrics.json", report)
  return report


def evaluate_episode(
  llm,
  *,
  seats: list[SeatSpec] | None = None,
  candidate_seat: int = 0,
  config: GameConfig | None = None,
  games: int = 1,
  seed: int = 0,
  memory_window: int = 3,
  win_rate_window: int = 10,
  max_calls: int = 600,
  max_tokens: int = 2048,
  run_dir: str | Path | None = None,
  resume: bool = False,
) -> EpisodeResult:
  """Run complete games; errors abort without manufacturing a partial score.

  Default: one candidate vs four RandomBots (a smoke-test baseline, not a
  social-deduction league). Pass explicit seats for mixed models or self-play.
  Games within a trajectory are sequential. Independent trajectories may run
  in separate SDK contexts/threads and must use different run directories.

  A crash replays the unfinished game from the last committed game boundary;
  model responses need not repeat. Completed games, memory, fallback RNG, and
  call usage survive. Resume requires identical experimental configuration;
  only max_calls may be increased. Keep automatic SDK evaluation retries off.
  """
  require_kaggle_proxy(llm)
  seats = list(seats) if seats is not None else [SeatSpec()] + [SeatSpec(model="random")] * 4
  config = config or recommended_config(len(seats))
  if len(seats) != config.num_players:
    raise ValueError("Seat count does not match GameConfig")
  for name, value in (("games", games), ("memory_window", memory_window), ("win_rate_window", win_rate_window)):
    if type(value) is not int or value < 1:
      raise ValueError(f"{name} must be a positive integer")
  if type(seed) is not int or type(candidate_seat) is not int or not 0 <= candidate_seat < len(seats):
    raise ValueError("Invalid seed or candidate seat")
  if seats[candidate_seat].model is not None:
    raise ValueError("The scored candidate seat must use the task's evaluated llm (model=None)")
  if resume and run_dir is None:
    raise ValueError("Resume requires an explicit run_dir")
  budget = CallBudget(max_calls)
  if run_dir is None:
    root = Path("data/kaggle")
    root.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="episode-", dir=root))
  else:
    path = Path(run_dir)
  names = make_names(len(seats), seed)
  manifest = {
    "protocol": "camelot-kaggle-v1", "source_digest": _source_digest(),
    "sdk_version": version("kaggle-benchmarks"), "candidate_model": llm.model,
    "target_games": games, "seed_base": seed, "candidate_seat": candidate_seat,
    "names": names, "memory_window": memory_window, "win_rate_window": win_rate_window,
    "max_tokens": max_tokens,
    "config": {"num_players": config.num_players, "specials": sorted(role.value for role in config.specials)},
    "seats": [asdict(spec) | {"model": llm.model if spec.model is None else spec.model} for spec in seats],
    "candidate_positions": [i for i, spec in enumerate(seats) if spec.model is None],
  }
  # Round-trip normalizes saved tuples and validates serializability before work.
  manifest = json.loads(json.dumps(manifest, allow_nan=False))
  with exclusive_run(path), using_names(names):
    store = Checkpoints(path, manifest, resume)
    budget.used = store.usage()
    budget.persist = store.save_usage
    players = _make_players(llm, seats, names, memory_window, budget, max_tokens)
    if store.games:
      for player, memory, rng in zip(players, store.games[-1]["memories"], store.games[-1]["rng_states"]):
        player.memory = Memory.from_dict(memory)
        player._rng.setstate(_tuple_tree(rng))
    for index in range(len(store.games) + 1, games + 1):
      for player in players:
        player.win_rate_series = rolling(player.memory.wins(), win_rate_window)
      game = AvalonGame(players, config=config, seed=seed + index, game_id=f"game_{index:04d}")
      game.play()
      game.run_debrief()
      winner = game.record.outcome["winner"]
      for seat, (player, spec) in enumerate(zip(players, seats)):
        if not spec.learn:
          continue
        role = game.assignment[seat]
        side = alignment_of(role).value
        player.memory.record_result({"won": side == winner, "role": role.value, "side": side})
        player.win_rate_series = rolling(player.memory.wins(), win_rate_window)
        log = game.record.render(seat)
        notes = player.revise_notes(log, game._ctx(seat))
        player.memory.update_after_revise(log, notes)
      store.commit({
        "record": game.record.to_jsonl(),
        "result": {
          "game_id": f"game_{index:04d}", "winner": winner,
          "reason": game.record.outcome["reason"], "roles": game.record.outcome["roles"],
          "seats": {name: {"model": p.name, "reasoning_effort": p.client.reasoning_effort}
                    for name, p in zip(names, players)},
        },
        "memories": [p.memory.to_dict() for p in players],
        "rng_states": [p._rng.getstate() for p in players],
      })
    candidate = players[candidate_seat]
    key = seat_key(names[candidate_seat], {"model": candidate.name, "reasoning_effort": candidate.client.reasoning_effort})
    return EpisodeResult(path, _report(path, key, budget.used))
