"""LLMPlayer: a Player backed by an injected client, deciding via a LangGraph subgraph."""

from __future__ import annotations

import random

from ..labels import label_to_seat
from ..player import Decision, Player, Seat, Utterance, Vote
from . import prompts, schemas
from .decision_graph import build_decision_graph
from .memory import Memory


class LLMPlayer(Player):
  """Plays via an injected client; owns its Memory. Each decision runs the decision graph."""

  def __init__(self, client, memory: Memory | None = None, name: str = "", seed: int | None = None):
    self.client = client
    self.memory = memory or Memory()
    self.name = name  # model identity, for the pipeline's seat->model map
    self.win_rate_series: list = []  # set by the pipeline before each game
    self._rng = random.Random(seed)

  # --- prompt plumbing ---

  def _system(self) -> str:
    return prompts.build_system_prompt(
      notes=self.memory.curated,
      recent_games=self.memory.recent_games(),
      win_rate_series=self.win_rate_series,
    )

  def _decide(self, log: str, ask: str, schema, validate, fallback):
    """Build a decision graph bound to this decision, run it, return the result."""
    graph = build_decision_graph(self.client, schema, validate, fallback)
    out = graph.invoke(
      {
        "system": self._system(),
        "user": prompts.build_user_prompt(log, ask),
        "attempts": 0,
        "max_attempts": 2,
      }
    )
    return out["result"]

  # --- seat/letter helpers ---

  def _one_seat(self, letter, n: int):
    try:
      s = label_to_seat(letter)
    except Exception:
      return None
    return s if 0 <= s < n else None

  def _to_seats(self, letters, n: int) -> list:
    seats = []
    for x in letters or []:
      s = self._one_seat(x, n)
      if s is not None and s not in seats:
        seats.append(s)
    return seats

  def _parse_belief(self, guesses, seat: int, n: int) -> dict:
    belief = {}
    for g in guesses or []:
      s = self._one_seat(g.seat, n)
      if s is not None and s != seat:
        belief[s] = g.role
    return belief

  # --- decisions (each supplies validate + fallback closures) ---

  def propose_team(self, log: str, ctx: Seat, team_size: int) -> Decision:
    def validate(raw):
      team = self._to_seats(raw.team, ctx.num_players)
      return (len(team) == team_size, Decision(tuple(team), raw.reasoning))

    def fallback():
      return Decision(tuple(self._rng.sample(range(ctx.num_players), team_size)), "(random pick)")

    return self._decide(log, prompts.ask_propose(team_size), schemas.ProposeDecision, validate, fallback)

  def speak(self, log: str, ctx: Seat, can_nominate: list) -> Utterance:
    def validate(raw):
      nxt = self._one_seat(raw.next_speaker, ctx.num_players)
      return (True, Utterance(raw.statement, raw.reasoning, nxt))

    def fallback():
      nxt = self._rng.choice(can_nominate) if can_nominate else None
      return Utterance("", "(random pick)", nxt)

    return self._decide(log, prompts.ask_speak(can_nominate), schemas.SpeakDecision, validate, fallback)

  def vote(self, log: str, ctx: Seat, proposed_team: tuple) -> Vote:
    def validate(raw):
      belief = self._parse_belief(raw.belief, ctx.seat, ctx.num_players)
      return (True, Vote(Decision(bool(raw.approve), raw.reasoning), belief=belief))

    def fallback():
      return Vote(Decision(self._rng.random() < 0.5, "(random pick)"))

    return self._decide(log, prompts.ask_vote(proposed_team), schemas.VoteDecision, validate, fallback)

  def quest_card(self, log: str, ctx: Seat) -> Decision:
    def validate(raw):
      return (True, Decision(bool(raw.success), raw.reasoning))

    def fallback():
      return Decision(self._rng.random() < 0.5, "(random pick)")

    return self._decide(log, prompts.ask_quest(), schemas.QuestDecision, validate, fallback)

  def assassinate(self, log: str, ctx: Seat) -> Decision:
    def validate(raw):
      t = self._one_seat(raw.target, ctx.num_players)
      return (t is not None, Decision(t, raw.reasoning) if t is not None else None)

    def fallback():
      cands = [s for s in range(ctx.num_players) if s != ctx.seat]
      return Decision(self._rng.choice(cands), "(random pick)")

    return self._decide(log, prompts.ask_assassinate(), schemas.AssassinateDecision, validate, fallback)

  def debrief(self, log: str, ctx: Seat) -> str:
    def validate(raw):
      return (True, raw.reflection)

    def fallback():
      return ""

    return self._decide(log, prompts.ask_debrief(), schemas.DebriefOutput, validate, fallback)
