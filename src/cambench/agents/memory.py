"""Player-owned Combine memory: curated notes (rewritten each game) + a sliding window of recent games."""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_WINDOW = 3  # recent games kept in the sliding window
DEFAULT_NOTES_CHARS = 4000  # soft cap on curated-notes length (safety net)


@dataclass
class Memory:
  """Combine memory: rewritable curated notes + a FIFO window of recent game logs."""

  curated: str = ""
  window: list = field(default_factory=list)
  window_size: int = DEFAULT_WINDOW
  notes_chars: int = DEFAULT_NOTES_CHARS

  def update(self, game_log: str, new_curated: str) -> None:
    """After a learning game: append its log to the window (FIFO), rewrite notes."""
    self.window.append(game_log)
    if len(self.window) > self.window_size:
      self.window = self.window[-self.window_size :]
    self.curated = new_curated.strip()[: self.notes_chars]

  def recent_games(self) -> list:
    """The sliding window, oldest to newest."""
    return list(self.window)

  def to_dict(self) -> dict:
    return {
      "curated": self.curated,
      "window": list(self.window),
      "window_size": self.window_size,
      "notes_chars": self.notes_chars,
    }

  @classmethod
  def from_dict(cls, d: dict) -> "Memory":
    return cls(
      curated=d.get("curated", ""),
      window=list(d.get("window", [])),
      window_size=d.get("window_size", DEFAULT_WINDOW),
      notes_chars=d.get("notes_chars", DEFAULT_NOTES_CHARS),
    )
