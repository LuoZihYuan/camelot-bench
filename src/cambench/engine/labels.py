"""Seat<->name labels: seats are ints inside the engine; names are the display identity.

Default names are letters (A, B, ...). A run can install per-run random names via
set_names() so agents reference opponents by a stable-within-run, random-across-runs
identity -- enabling opponent-modeling while keeping stale cross-run notes unmatchable.
"""

from __future__ import annotations

import random
import string

_DEFAULT = string.ascii_uppercase  # A..Z (letters, the default identity)
_names: list[str] = list(_DEFAULT)  # current mapping: seat index -> name


def seat_label(seat: int) -> str:
  return _names[seat]


def label_to_seat(label: str) -> int:
  return _names.index(label)


def labels(seats) -> list:
  return [seat_label(s) for s in seats]


def set_names(names: list) -> None:
  """Install the per-run seat names (index -> name). Pass letters to reset."""
  global _names
  _names = list(names)


def reset_names() -> None:
  set_names(list(_DEFAULT))


def make_names(n: int, seed: int) -> list:
  """n meaningless, distinct seat names (Letter-Digit-Digit-Letter), seeded per run.

  LDDL (e.g. K83V) structurally avoids words, leetspeak, and numeronyms; dropping
  I/O/0/1 removes confusables and the common numeronym characters.
  """
  rng = random.Random(seed)
  L = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # 24 letters (no I, O)
  D = "23456789"  # 8 digits (no 0, 1)
  names: list[str] = []
  while len(names) < n:
    name = rng.choice(L) + rng.choice(D) + rng.choice(D) + rng.choice(L)
    if name not in names:
      names.append(name)
  return names
