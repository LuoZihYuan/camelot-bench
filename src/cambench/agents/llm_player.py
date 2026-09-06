"""LLMPlayer: a Player backed by an injected LLM client, with its own memory."""

from __future__ import annotations

import random

from ..labels import label_to_seat
from ..player import Decision, Player, Seat, Utterance, Vote
from . import prompts, schemas
from .memory import Memory


class LLMPlayer(Player):
  """Plays via an injected client; owns its Memory. Failed calls fall back to random."""

  def __init__(self, client, memory: Memory | None = None, name: str = "", seed: int | None = None):
    self.client = client
    self.memory = memory or Memory()
    self.name = name  # model identity, for the pipeline's seat->model map
    self.win_rate_series: list = []  # set by the pipeline before each game
    self._rng = random.Random(seed)

  # --- prompt + call plumbing ---

  def _system(self) -> str:
    return prompts.build_system_prompt(
      notes=self.memory.curated,
      recent_games=self.memory.recent_games(),
      win_rate_series=self.win_rate_series,
    )

  def _call(self, log: str, ask: str, schema):
    """One structured call; returns the schema instance, or None on failure."""
    try:
      return self.client.call(self._system(), prompts.build_user_prompt(log, ask), schema)
    except Exception:
      return None

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

  # --- decisions ---

  def propose_team(self, log: str, ctx: Seat, team_size: int) -> Decision:
    out = self._call(log, prompts.ask_propose(team_size), schemas.ProposeDecision)
    if out is not None:
      team = self._to_seats(out.team, ctx.num_players)
      if len(team) == team_size:
        return Decision(tuple(team), out.reasoning)
    return Decision(tuple(self._rng.sample(range(ctx.num_players), team_size)), "(random pick)")

  def speak(self, log: str, ctx: Seat, can_nominate: list) -> Utterance:
    out = self._call(log, prompts.ask_speak(can_nominate), schemas.SpeakDecision)
    if out is not None:
      nxt = self._one_seat(out.next_speaker, ctx.num_players)
      return Utterance(out.statement, out.reasoning, nxt)
    nxt = self._rng.choice(can_nominate) if can_nominate else None
    return Utterance("", "(random pick)", nxt)

  def vote(self, log: str, ctx: Seat, proposed_team: tuple) -> Vote:
    out = self._call(log, prompts.ask_vote(proposed_team), schemas.VoteDecision)
    if out is not None:
      belief = self._parse_belief(out.belief, ctx.seat, ctx.num_players)
      return Vote(Decision(bool(out.approve), out.reasoning), belief=belief)
    return Vote(Decision(self._rng.random() < 0.5, "(random pick)"))

  def quest_card(self, log: str, ctx: Seat) -> Decision:
    out = self._call(log, prompts.ask_quest(), schemas.QuestDecision)
    if out is not None:
      return Decision(bool(out.success), out.reasoning)
    return Decision(self._rng.random() < 0.5, "(random pick)")

  def assassinate(self, log: str, ctx: Seat) -> Decision:
    out = self._call(log, prompts.ask_assassinate(), schemas.AssassinateDecision)
    if out is not None:
      t = self._one_seat(out.target, ctx.num_players)
      if t is not None:
        return Decision(t, out.reasoning)
    cands = [s for s in range(ctx.num_players) if s != ctx.seat]
    return Decision(self._rng.choice(cands), "(random pick)")

  def debrief(self, log: str, ctx: Seat) -> str:
    out = self._call(log, prompts.ask_debrief(), schemas.DebriefOutput)
    return out.reflection if out is not None else ""
