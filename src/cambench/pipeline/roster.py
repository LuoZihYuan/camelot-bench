"""Resolve a run's per-seat config from a YAML roster and/or CLI flags, over .env defaults."""

from __future__ import annotations

import pathlib

from ..config import settings

SEAT_FIELDS = ("model", "reasoning_effort", "memory", "learn")


def load_roster(path: str) -> dict:
  """Parse a YAML roster, keeping seats RAW (only fields the user actually set)."""
  import yaml

  data = yaml.safe_load(pathlib.Path(path).read_text()) or {}
  return {
    "raw_seats": list(data.get("seats") or []),  # untouched; explicit fields only
    "games": data.get("games"),
    "label": data.get("label"),
    "win_rate_window": data.get("win_rate_window"),
    "memory_window": data.get("memory_window"),
  }


def seat_field_present(raw_seats: list, field: str) -> bool:
  """True if any seat explicitly sets `field` (used for conflict detection)."""
  return any(field in s for s in raw_seats)


def resolve_seats(raw_seats: list, flag_defaults: dict) -> list:
  """Fill each seat's omitted fields from flag_defaults, then settings defaults."""
  fallback = {
    "model": settings.default_model,
    "reasoning_effort": settings.default_reasoning_effort,
    "memory": None,
    "learn": True,
  }
  seats = []
  for s in raw_seats:
    seats.append({f: s.get(f, flag_defaults.get(f, fallback[f])) for f in SEAT_FIELDS})
  return seats


def homogeneous_seats(model, players, memory, learn, reasoning_effort) -> list:
  """Build N identical seats for the no-config quick path."""
  return [
    {
      "model": model or settings.default_model,
      "reasoning_effort": reasoning_effort if reasoning_effort is not None else settings.default_reasoning_effort,
      "memory": memory,
      "learn": learn,
    }
    for _ in range(players)
  ]
