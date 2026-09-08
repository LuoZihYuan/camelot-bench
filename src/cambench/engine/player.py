"""The Player interface (the engine's extension seam), its value objects, and RandomBot."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .roles import Alignment, Knowledge, Role


@dataclass(frozen=True)
class Utterance:
  """A speech: `statement` is public, `reasoning` private; `next_speaker` (a seat
  id) nominates who speaks next -- public, and carries the 'addressing' signal."""

  statement: str = ""
  reasoning: str = ""
  next_speaker: int | None = None


@dataclass(frozen=True)
class Decision:
  """A non-speech action: `value` is the public choice, `reasoning` is private."""

  value: object
  reasoning: str = ""


@dataclass(frozen=True)
class Vote:
  """A vote: the underlying Decision plus a private belief snapshot (role guesses
  for the other seats). `belief` maps seat id -> guessed Role; filled by LLM
  players at the vote, left empty by simple players."""

  decision: Decision
  belief: dict = field(default_factory=dict)

  @property
  def value(self):
    return self.decision.value

  @property
  def reasoning(self) -> str:
    return self.decision.reasoning


@dataclass(frozen=True)
class Seat:
  """One seat's identity + private knowledge. No game history (that's the log)."""

  seat: int
  num_players: int
  role: Role
  alignment: Alignment
  knowledge: Knowledge

  @property
  def known_evil(self) -> frozenset:
    return self.knowledge.known_evil

  @property
  def merlin_candidates(self) -> frozenset:
    return self.knowledge.merlin_candidates

  def is_evil(self) -> bool:
    return self.alignment is Alignment.EVIL


class Player(ABC):
  @abstractmethod
  def propose_team(self, log: str, ctx: Seat, team_size: int) -> Decision:
    """Decision.value = list of `team_size` distinct seat ids."""

  @abstractmethod
  def speak(self, log: str, ctx: Seat, can_nominate: list) -> Utterance:
    """Public statement + private reasoning + nomination of the next speaker.
    `can_nominate` = seats still eligible to be nominated (under cap, not self)."""

  @abstractmethod
  def vote(self, log: str, ctx: Seat, proposed_team: tuple) -> Vote:
    """Vote.value = True to approve, False to reject; Vote.belief = role guesses."""

  @abstractmethod
  def quest_card(self, log: str, ctx: Seat) -> Decision:
    """Decision.value = True for SUCCESS, False for FAIL. Only evil are asked
    (good must succeed by rule, and are not called)."""

  @abstractmethod
  def assassinate(self, log: str, ctx: Seat) -> Decision:
    """Decision.value = the seat the assassin believes is Merlin."""

  def debrief(self, log: str, ctx: Seat) -> str:
    """Post-game private reflection, after the reveal. Default: none."""
    return ""


class RandomBot(Player):
  """Legal, seedable random player for exercising the engine. No strategy."""

  def __init__(self, seed: int | None = None):
    self._rng = random.Random(seed)

  def propose_team(self, log: str, ctx: Seat, team_size: int) -> Decision:
    return Decision(self._rng.sample(range(ctx.num_players), team_size))

  def speak(self, log: str, ctx: Seat, can_nominate: list) -> Utterance:
    return Utterance()

  def vote(self, log: str, ctx: Seat, proposed_team: tuple) -> Vote:
    return Vote(Decision(self._rng.random() < 0.5))

  def quest_card(self, log: str, ctx: Seat) -> Decision:
    # only evil are asked; ~50% fail
    return Decision(self._rng.random() < 0.5)

  def assassinate(self, log: str, ctx: Seat) -> Decision:
    candidates = [s for s in range(ctx.num_players) if s != ctx.seat and s not in ctx.known_evil]
    return Decision(self._rng.choice(candidates or [ctx.seat]))
