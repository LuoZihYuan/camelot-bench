"""GameRecord: the god-view event log, with per-seat perspective() and JSON serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from .roles import Alignment, Role, alignment_of


class EventType(str, Enum):
  START = "start"
  PROPOSE = "propose"
  SPEECH = "speak"
  VOTE = "vote"
  QUEST = "quest"
  ASSASSINATE = "assassinate"
  END = "end"
  DEBRIEF = "debrief"
  CAP = "cap"


# "every seat can see this event"
ALL = "all"


@dataclass
class Event:
  type: EventType
  payload: dict
  # ALL, or a set of seat ids
  visible_to: object = ALL
  # per-seat private overlay, merged only into that seat's perspective
  private: dict = field(default_factory=dict)

  def visible(self, seat: int) -> bool:
    return self.visible_to == ALL or seat in self.visible_to

  def render_for(self, seat: int) -> dict:
    """Render this event as the given seat should see it."""
    view = {"type": self.type.value, **self.payload}
    if seat in self.private:
      view = {**view, **self.private[seat]}
    return view


@dataclass
class GameRecord:
  game_id: str
  num_players: int
  # god view; never handed to agents
  assignment: dict[int, Role]
  # per-seat setup knowledge
  initial_knowledge: dict[int, frozenset[int]]
  events: list[Event] = field(default_factory=list)
  outcome: dict = field(default_factory=dict)

  # --- writing ---

  def add(self, event: Event) -> None:
    self.events.append(event)

  # --- god view (analysis only) ---

  def roles(self) -> dict[int, str]:
    return {s: r.value for s, r in self.assignment.items()}

  def evil_seats(self) -> set[int]:
    return {s for s, r in self.assignment.items() if alignment_of(r) is Alignment.EVIL}

  def merlin_seat(self) -> int:
    return next(s for s, r in self.assignment.items() if r is Role.MERLIN)

  # --- one seat's perspective (feeds review) ---

  def perspective(self, seat: int) -> dict:
    """One seat's filtered whole-game timeline as structured data."""
    from .labels import labels, seat_label

    k = self.initial_knowledge[seat]
    ik = {"known_evil": labels(sorted(k.known_evil))}
    if k.merlin_candidates:
      ik["merlin_candidates"] = labels(sorted(k.merlin_candidates))
    timeline = [ev.render_for(seat) for ev in self.events if ev.visible(seat)]
    return {
      "seat": seat_label(seat),
      "role": self.assignment[seat].value,
      "initial_knowledge": ik,
      "timeline": timeline,
    }

  def render(self, seat: int, include_beliefs: bool = True) -> str:
    """Seat's filtered history so far as formatted text (the log an LLM reads)."""
    from .roles import REASON_TEXT, ROLE_DISPLAY, ROLE_ORDER
    from .labels import seat_label

    def section(label):
      return f" {label} ".center(30, "\u2500")  # ─ game sections

    def phase(label):
      return f" {label} ".center(30, "\u00b7")  # · phases

    def reflect(label):
      return f" {label} ".center(30, "\u2550")  # ═ reflection

    def role_name(v):
      return ROLE_DISPLAY.get(v, v.title())

    def roles_in_play_text(vals):
      counts = {v: vals.count(v) for v in set(vals)}
      parts = []
      for r in ROLE_ORDER:
        if r in counts:
          n = counts[r]
          parts.append(role_name(r) + (f" x{n}" if n > 1 else ""))
      return ", ".join(parts)

    lines = []
    me = seat_label(seat)
    team_sizes, fails_needed = [], []
    cur_quest = None
    attempt = 0

    for ev in self.events:
      if not ev.visible(seat):
        continue
      e = ev.render_for(seat)
      t = e["type"]

      if t == "start":
        team_sizes = e["quest_team_sizes"]
        fails_needed = e["fails_required"]
        lines.append(section("START"))
        lines.append(f"[GAME] {e['num_players']} players. Roles in play: {roles_in_play_text(e['roles_in_play'])}.")
        if "your_role" in e:
          extra = ""
          if e.get("known_evil"):
            extra += f" Known evil: {', '.join(e['known_evil'])}."
          if e.get("merlin_candidates"):
            extra += f" Merlin is one of: {', '.join(e['merlin_candidates'])}."
          lines.append(
            f"[GAME] (privately) You are {role_name(e['your_role'])} — "
            f"{e['your_alignment'].title()}, on seat {e['your_seat']}."
            f"{extra}"
          )

      elif t == "propose":
        q = e["quest_index"]
        if q != cur_quest:
          cur_quest = q
          attempt = 0
          lines.append(section(f"QUEST {q + 1}"))
          lines.append(f"[GAME] Team size: {team_sizes[q]}. Fails needed: {fails_needed[q]}.")
        attempt += 1
        lines.append(phase(f"PROPOSE {attempt}"))
        lines.append(f"[{e['leader']}] team {', '.join(e['team'])}.")
        if "reasoning" in e:
          lines.append(f"[{me}] (privately) {e['reasoning']}")
        lines.append(phase(f"DISCUSS {attempt}"))

      elif t == "speak":
        if e.get("statement"):
          line = f"[{e['seat']}] {e['statement']}"
          if e.get("next"):
            line += f" {e['next']}, you're next."
          lines.append(line)
        if e.get("reasoning"):
          lines.append(f"[{e['seat']}] (privately) {e['reasoning']}")

      elif t == "cap":
        lines.append(f"[GAME] {e['seat']} has used all {e['turns']} turns.")

      elif t == "vote":
        lines.append(phase(f"VOTE {e['attempt'] + 1}"))
        res = "Approved" if e["approved"] else "Rejected"
        tally = ", ".join(f"{s}: {'yes' if v else 'no'}" for s, v in e["votes"].items())
        lines.append(f"[GAME] {res}. {tally}.")
        if "reasoning" in e:
          lines.append(f"[{me}] (privately) {e['reasoning']}")
        if "belief" in e and include_beliefs:
          guesses = ", ".join(f"{s}: {role_name(r)}" for s, r in e["belief"].items())
          lines.append(f"[{me}] (privately) I think: {guesses}.")

      elif t == "quest":
        lines.append(phase("MISSION"))
        res = "Success" if e["success"] else "Failed"
        nf = e["num_fails"]
        lines.append(f"[GAME] {res}. {nf} fail{'s' if nf != 1 else ''}.")
        if "reasoning" in e:
          lines.append(f"[{me}] (privately) {e['reasoning']}")

      elif t == "assassinate":
        lines.append(section("ASSASSIN"))
        lines.append(f"[GAME] Assassin: {e['assassin']}")
        lines.append(f"[{e['assassin']}] target {e['target']}.")
        if "reasoning" in e:
          lines.append(f"[{me}] (privately) {e['reasoning']}")
        lines.append(f"[GAME] Assassination {'succeeded' if e['hit_merlin'] else 'failed'}.")

      elif t == "end":
        lines.append(section("END"))
        reason = REASON_TEXT.get(e["reason"], e["reason"])
        lines.append(f"[GAME] {e['winner'].title()} wins. Reason: {reason}.")
        roles_txt = ", ".join(f"{s}: {role_name(r)}" for s, r in e["roles"].items())
        lines.append(f"[GAME] Roles: {roles_txt}.")

      elif t == "debrief":
        if "reasoning" in e:
          lines.append(reflect("REFLECTION"))
          lines.append(f"[{me}] (privately) {e['reasoning']}")

    return "\n".join(lines)

  # --- serialization ---

  def to_dict(self) -> dict:
    from .labels import labels, seat_label

    return {
      "game_id": self.game_id,
      "num_players": self.num_players,
      "assignment": {seat_label(s): r.value for s, r in self.assignment.items()},
      "initial_knowledge": {
        seat_label(s): {
          "known_evil": labels(sorted(k.known_evil)),
          **({"merlin_candidates": labels(sorted(k.merlin_candidates))} if k.merlin_candidates else {}),
        }
        for s, k in self.initial_knowledge.items()
      },
      "events": [
        {
          "type": ev.type.value,
          "payload": ev.payload,
          "visible_to": (ALL if ev.visible_to == ALL else labels(sorted(ev.visible_to))),
          "private": {seat_label(s): p for s, p in ev.private.items()},
        }
        for ev in self.events
      ],
      "outcome": self.outcome,
    }

  def to_json(self, indent: int | None = None) -> str:
    return json.dumps(self.to_dict(), indent=indent)
