"""Player-owned Combine memory: curated notes (rewritten each game) + a sliding window of recent games."""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_WINDOW = 3  # recent games kept in the sliding window
DEFAULT_NOTES_CHARS = 4000  # soft cap on curated-notes length (safety net)


@dataclass
class Memory:
  """Combine memory: rewritable curated notes + a FIFO window of recent game logs + own results."""

  curated: str = ""
  window: list = field(default_factory=list)
  results: list = field(default_factory=list)  # this agent's own per-game outcomes
  window_size: int = DEFAULT_WINDOW
  notes_chars: int = DEFAULT_NOTES_CHARS

  def update_after_revise(self, game_log: str, new_curated: str) -> None:
    """Store window + revised notes; the result was recorded earlier (before revision)."""
    self.window.append(game_log)
    if len(self.window) > self.window_size:
      self.window = self.window[-self.window_size :]
    self.curated = new_curated.strip()[: self.notes_chars]

  def record_result(self, result: dict) -> None:
    """Record a game outcome even when memory isn't otherwise updated (e.g. frozen)."""
    self.results.append(result)

  def wins(self) -> list:
    """This agent's per-game win/lose booleans, oldest to newest."""
    return [bool(r.get("won")) for r in self.results]

  def recent_games(self) -> list:
    """The sliding window, oldest to newest."""
    return list(self.window)

  def to_dict(self) -> dict:
    return {
      "curated": self.curated,
      "window": list(self.window),
      "results": list(self.results),
      "window_size": self.window_size,
      "notes_chars": self.notes_chars,
    }

  @classmethod
  def from_dict(cls, d: dict) -> "Memory":
    return cls(
      curated=d.get("curated", ""),
      window=list(d.get("window", [])),
      results=list(d.get("results", [])),
      window_size=d.get("window_size", DEFAULT_WINDOW),
      notes_chars=d.get("notes_chars", DEFAULT_NOTES_CHARS),
    )
