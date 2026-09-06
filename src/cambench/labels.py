"""Seat<->letter labels (A=0, B=1, ...): letters in logs, ints inside the engine."""

from __future__ import annotations

import string

_LETTERS = string.ascii_uppercase  # A..Z (we only use up to J for 10 players)


def seat_label(seat: int) -> str:
  return _LETTERS[seat]


def label_to_seat(label: str) -> int:
  return _LETTERS.index(label)


def labels(seats) -> list:
  return [seat_label(s) for s in seats]
