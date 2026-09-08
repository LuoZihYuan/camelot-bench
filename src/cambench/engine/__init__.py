"""Deterministic 5-10 player Avalon engine (framework-free)."""

from .game import AvalonGame, play_game
from .labels import label_to_seat, labels, seat_label
from .player import Decision, Player, RandomBot, Seat, Utterance, Vote
from .records import Event, EventType, GameRecord, LiveStream, format_event
from .roles import (
  Alignment,
  GameConfig,
  Knowledge,
  Role,
  alignment_of,
  barebones_config,
  recommended_config,
)

__all__ = [
  "AvalonGame",
  "play_game",
  "Player",
  "RandomBot",
  "Decision",
  "Vote",
  "Seat",
  "Utterance",
  "GameRecord",
  "Event",
  "EventType",
  "LiveStream",
  "format_event",
  "Role",
  "Alignment",
  "Knowledge",
  "GameConfig",
  "recommended_config",
  "barebones_config",
  "alignment_of",
  "seat_label",
  "label_to_seat",
  "labels",
]
