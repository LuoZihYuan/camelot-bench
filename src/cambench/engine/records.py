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
  on_event: object = None  # optional callback(event) fired live as each event is added

  # --- writing ---

  def add(self, event: Event) -> None:
    self.events.append(event)
    if self.on_event is not None:
      self.on_event(event)

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
    state = {}
    lines = []
    for ev in self.events:
      if not ev.visible(seat):
        continue
      lines += format_event(ev, seat, state, include_beliefs)
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


# --- shared per-event formatting (used by render() batch and LiveStream live) ---


def _div(label, ch):
  return f" {label} ".center(30, ch)


def _role_name(v):
  from .roles import ROLE_DISPLAY

  return ROLE_DISPLAY.get(v, v.title())


def _roles_in_play(vals):
  from .roles import ROLE_ORDER

  counts = {v: vals.count(v) for v in set(vals)}
  return ", ".join(_role_name(r) + (f" x{counts[r]}" if counts[r] > 1 else "") for r in ROLE_ORDER if r in counts)


def _privates_god(ev, include_beliefs):
  """All seats' private overlays, labelled by owner (god view)."""
  from .labels import seat_label

  out = []
  for seat, ov in sorted(ev.private.items()):
    who = seat_label(seat)
    if ov.get("reasoning"):
      out.append(f"[{who}] (privately) {ov['reasoning']}")
    if include_beliefs and ov.get("belief"):
      g = ", ".join(f"{s}: {_role_name(r)}" for s, r in ov["belief"].items())
      out.append(f"[{who}] (privately) I think: {g}.")
  return out


def format_event(ev, viewer, state: dict, include_beliefs: bool = True) -> list:
  """Format one event as display lines.

  `viewer` is a seat id (filtered: only that seat's private overlay, shown as
  "You"/"me") or the string "god" (all seats' overlays, each labelled by owner).
  `state` is a mutable dict carrying quest/attempt/team-size tracking across calls.
  """
  from .labels import seat_label
  from .roles import REASON_TEXT

  god = viewer == "god"
  e = ev.payload if god else ev.render_for(viewer)
  t = ev.type.value
  me = None if god else seat_label(viewer)
  lines = []

  def privates():
    # in god view show everyone's; in seat view show only this seat's ("me")
    if god:
      return _privates_god(ev, include_beliefs)
    out = []
    if e.get("reasoning"):
      out.append(f"[{me}] (privately) {e['reasoning']}")
    if include_beliefs and e.get("belief"):
      g = ", ".join(f"{s}: {_role_name(r)}" for s, r in e["belief"].items())
      out.append(f"[{me}] (privately) I think: {g}.")
    return out

  if t == "start":
    state["team_sizes"] = e["quest_team_sizes"]
    state["fails"] = e["fails_required"]
    lines.append(_div("START", "\u2500"))
    lines.append(f"[GAME] {e['num_players']} players. Roles in play: {_roles_in_play(e['roles_in_play'])}.")
    if god:
      for seat, ov in sorted(ev.private.items()):
        extra = ""
        if ov.get("known_evil"):
          extra += f" Known evil: {', '.join(ov['known_evil'])}."
        if ov.get("merlin_candidates"):
          extra += f" Merlin is one of: {', '.join(ov['merlin_candidates'])}."
        lines.append(f"[GAME] (to {seat_label(seat)}) {_role_name(ov['your_role'])} — {ov['your_alignment'].title()}.{extra}")
    elif "your_role" in e:
      extra = ""
      if e.get("known_evil"):
        extra += f" Known evil: {', '.join(e['known_evil'])}."
      if e.get("merlin_candidates"):
        extra += f" Merlin is one of: {', '.join(e['merlin_candidates'])}."
      lines.append(
        f"[GAME] (privately) You are {_role_name(e['your_role'])} — "
        f"{e['your_alignment'].title()}, on seat {e['your_seat']}.{extra}"
      )

  elif t == "propose":
    q = e["quest_index"]
    if q != state.get("cur_quest"):
      state["cur_quest"] = q
      state["attempt"] = 0
      lines.append(_div(f"QUEST {q + 1}", "\u2500"))
      lines.append(f"[GAME] Team size: {state['team_sizes'][q]}. Fails needed: {state['fails'][q]}.")
    state["attempt"] = state.get("attempt", 0) + 1
    lines.append(_div(f"PROPOSE {state['attempt']}", "\u00b7"))
    lines.append(f"[{e['leader']}] team {', '.join(e['team'])}.")
    lines += privates()
    lines.append(_div(f"DISCUSS {state['attempt']}", "\u00b7"))

  elif t == "speak":
    if e.get("statement"):
      line = f"[{e['seat']}] {e['statement']}"
      if e.get("next"):
        line += f" {e['next']}, you're next."
      lines.append(line)
    lines += privates()

  elif t == "cap":
    lines.append(f"[GAME] {e['seat']} has used all {e['turns']} turns.")

  elif t == "vote":
    lines.append(_div(f"VOTE {e['attempt'] + 1}", "\u00b7"))
    res = "Approved" if e["approved"] else "Rejected"
    tally = ", ".join(f"{s}: {'yes' if v else 'no'}" for s, v in e["votes"].items())
    lines.append(f"[GAME] {res}. {tally}.")
    lines += privates()

  elif t == "quest":
    lines.append(_div("MISSION", "\u00b7"))
    res = "Success" if e["success"] else "Failed"
    nf = e["num_fails"]
    lines.append(f"[GAME] {res}. {nf} fail{'s' if nf != 1 else ''}.")
    lines += privates()

  elif t == "assassinate":
    lines.append(_div("ASSASSIN", "\u2500"))
    lines.append(f"[GAME] Assassin: {e['assassin']}")
    lines.append(f"[{e['assassin']}] target {e['target']}.")
    lines += privates()
    lines.append(f"[GAME] Assassination {'succeeded' if e['hit_merlin'] else 'failed'}.")

  elif t == "end":
    lines.append(_div("END", "\u2500"))
    reason = REASON_TEXT.get(e["reason"], e["reason"])
    lines.append(f"[GAME] {e['winner'].title()} wins. Reason: {reason}.")
    roles_txt = ", ".join(f"{s}: {_role_name(r)}" for s, r in e["roles"].items())
    lines.append(f"[GAME] Roles: {roles_txt}.")

  elif t == "debrief":
    priv = privates()
    if priv:
      if not state.get("reflection_shown"):
        lines.append(_div("REFLECTION", "\u2550"))
        state["reflection_shown"] = True
      lines += priv

  return lines


class LiveStream:
  """GameRecord.on_event callback: prints each event live in god view."""

  def __init__(self, printer=print, include_beliefs: bool = True):
    self._print = printer
    self._beliefs = include_beliefs
    self._state = {}

  def __call__(self, ev):
    lines = format_event(ev, "god", self._state, self._beliefs)
    if lines:
      self._print("\n".join(lines), flush=True)
