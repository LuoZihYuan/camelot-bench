"""Roles, per-count game config, knowledge rules, and setup for 5-10 player Avalon."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Alignment(str, Enum):
  GOOD = "good"
  EVIL = "evil"


class Role(str, Enum):
  MERLIN = "merlin"  # good; sees evil except Mordred
  PERCIVAL = "percival"  # good; sees Merlin+Morgana, unlabeled
  LOYAL_SERVANT = "loyal"  # good; no knowledge
  ASSASSIN = "assassin"  # evil; guesses Merlin at the end
  MORGANA = "morgana"  # evil; looks like Merlin to Percival
  MORDRED = "mordred"  # evil; hidden from Merlin
  OBERON = "oberon"  # evil; blind to evil and vice versa
  MINION = "minion"  # evil; ordinary


ROLE_ALIGNMENT: dict[Role, Alignment] = {
  Role.MERLIN: Alignment.GOOD,
  Role.PERCIVAL: Alignment.GOOD,
  Role.LOYAL_SERVANT: Alignment.GOOD,
  Role.ASSASSIN: Alignment.EVIL,
  Role.MORGANA: Alignment.EVIL,
  Role.MORDRED: Alignment.EVIL,
  Role.OBERON: Alignment.EVIL,
  Role.MINION: Alignment.EVIL,
}

GOOD_SPECIALS = frozenset({Role.PERCIVAL})
EVIL_SPECIALS = frozenset({Role.MORGANA, Role.MORDRED, Role.OBERON})
OPTIONAL_SPECIALS = GOOD_SPECIALS | EVIL_SPECIALS

MIN_PLAYERS = 5
MAX_PLAYERS = 10

# (good, evil) counts per player count
SPLIT: dict[int, tuple[int, int]] = {
  5: (3, 2),
  6: (4, 2),
  7: (4, 3),
  8: (5, 3),
  9: (6, 3),
  10: (6, 4),
}

# team size per quest, per player count
TEAM_SIZES: dict[int, tuple[int, ...]] = {
  5: (2, 3, 2, 3, 3),
  6: (2, 3, 4, 3, 4),
  7: (2, 3, 3, 4, 4),
  8: (3, 4, 4, 5, 5),
  9: (3, 4, 4, 5, 5),
  10: (3, 4, 4, 5, 5),
}


def _fails_for(n: int) -> tuple[int, ...]:
  # 7+ players: quest 4 needs 2 fails
  return (1, 1, 1, 2, 1) if n >= 7 else (1, 1, 1, 1, 1)


FAILS_REQUIRED: dict[int, tuple[int, ...]] = {n: _fails_for(n) for n in SPLIT}

QUESTS_TO_WIN = 3
QUESTS_TO_LOSE = 3
MAX_CONSECUTIVE_REJECTS = 5
TURNS_PER_SEAT = 3  # discussion: max speeches per seat per proposal


def alignment_of(role: Role) -> Alignment:
  return ROLE_ALIGNMENT[role]


def is_evil(role: Role) -> bool:
  return ROLE_ALIGNMENT[role] is Alignment.EVIL


def is_good(role: Role) -> bool:
  return ROLE_ALIGNMENT[role] is Alignment.GOOD


# Knowledge: what a seat privately learns at setup.                           #
# `known_evil` = seats believed evil (Merlin / Evil).                         #
# `merlin_candidates` = Percival's UNLABELED {Merlin, Morgana} pair.          #


@dataclass(frozen=True)
class Knowledge:
  known_evil: frozenset = frozenset()
  merlin_candidates: frozenset = frozenset()

  def to_dict(self) -> dict:
    d: dict = {"known_evil": sorted(self.known_evil)}
    if self.merlin_candidates:
      d["merlin_candidates"] = sorted(self.merlin_candidates)
    return d


def compute_knowledge(seat: int, assignment: dict) -> Knowledge:
  role = assignment[seat]
  evil_seats = {s for s, r in assignment.items() if is_evil(r)}

  if role is Role.MERLIN:
    known = {s for s in evil_seats if assignment[s] is not Role.MORDRED}
    return Knowledge(known_evil=frozenset(known))

  if role is Role.PERCIVAL:
    cands = {s for s, r in assignment.items() if r in (Role.MERLIN, Role.MORGANA)}
    return Knowledge(merlin_candidates=frozenset(cands))

  if role is Role.OBERON:
    return Knowledge()

  if is_evil(role):
    known = {s for s in evil_seats if s != seat and assignment[s] is not Role.OBERON}
    return Knowledge(known_evil=frozenset(known))

  return Knowledge()  # loyal servant


# GameConfig: player count + optional specials, validated       #


@dataclass(frozen=True)
class GameConfig:
  num_players: int
  specials: frozenset = frozenset()

  def __post_init__(self):
    n = self.num_players
    if not (MIN_PLAYERS <= n <= MAX_PLAYERS):
      raise ValueError(f"num_players must be {MIN_PLAYERS}-{MAX_PLAYERS}, got {n}")

    bad = self.specials - OPTIONAL_SPECIALS
    if bad:
      raise ValueError(f"not optional specials: {sorted(r.value for r in bad)}")

    has_perc = Role.PERCIVAL in self.specials
    has_morg = Role.MORGANA in self.specials
    if has_perc != has_morg:
      raise ValueError("Percival and Morgana are a package deal: include both or neither")

    good_slots, evil_slots = SPLIT[n]
    n_good_specials = 1 + (1 if has_perc else 0)  # Merlin (+ Percival)
    n_evil_specials = 1 + len(self.specials & EVIL_SPECIALS)  # Assassin (+ evil specials)
    if n_good_specials > good_slots:
      raise ValueError(f"too many good specials for {n} players ({n_good_specials} > {good_slots})")
    if n_evil_specials > evil_slots:
      raise ValueError(f"too many evil specials for {n} players ({n_evil_specials} > {evil_slots})")

  def role_list(self) -> list:
    good_slots, evil_slots = SPLIT[self.num_players]

    good = [Role.MERLIN]
    if Role.PERCIVAL in self.specials:
      good.append(Role.PERCIVAL)
    good += [Role.LOYAL_SERVANT] * (good_slots - len(good))

    evil = [Role.ASSASSIN]
    for r in (Role.MORGANA, Role.MORDRED, Role.OBERON):
      if r in self.specials:
        evil.append(r)
    evil += [Role.MINION] * (evil_slots - len(evil))

    assert len(good) == good_slots and len(evil) == evil_slots
    return good + evil

  @property
  def team_sizes(self) -> tuple:
    return TEAM_SIZES[self.num_players]

  @property
  def fails_required(self) -> tuple:
    return FAILS_REQUIRED[self.num_players]


_RECOMMENDED_SPECIALS: dict = {
  5: frozenset({Role.PERCIVAL, Role.MORGANA}),
  6: frozenset({Role.PERCIVAL, Role.MORGANA}),
  7: frozenset({Role.PERCIVAL, Role.MORGANA, Role.MORDRED}),
  8: frozenset({Role.PERCIVAL, Role.MORGANA, Role.MORDRED}),
  9: frozenset({Role.PERCIVAL, Role.MORGANA, Role.MORDRED}),
  10: frozenset({Role.PERCIVAL, Role.MORGANA, Role.MORDRED, Role.OBERON}),
}


def recommended_config(num_players: int) -> GameConfig:
  return GameConfig(num_players, _RECOMMENDED_SPECIALS[num_players])


def barebones_config(num_players: int) -> GameConfig:
  """Merlin + Assassin only; everyone else generic."""
  return GameConfig(num_players, frozenset())


# Display names + a stable good-then-evil ordering for rendering.
ROLE_DISPLAY = {
  "merlin": "Merlin",
  "percival": "Percival",
  "loyal": "Loyal Servant",
  "assassin": "Assassin",
  "morgana": "Morgana",
  "mordred": "Mordred",
  "oberon": "Oberon",
  "minion": "Minion",
}
ROLE_ORDER = ["merlin", "percival", "loyal", "assassin", "morgana", "mordred", "oberon", "minion"]

REASON_TEXT = {
  "three_failed_quests": "3 failed quests",
  "quests_won_merlin_safe": "3 successful quests",
  "merlin_assassinated": "Merlin assassinated",
  "five_consecutive_rejects": "5 rejected proposals in a row",
}
