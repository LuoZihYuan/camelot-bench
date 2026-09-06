"""Prompt assembly: system prompt = game-fixed content, user prompt = per-call content."""

from __future__ import annotations

from ..labels import labels


# between-games wall in the recent-games window (chevrons, distinct from the
# in-game =/-/. banners); placed only BETWEEN games, never at the ends.
GAME_SEPARATOR = "<" * 30 + " NEXT GAME " + ">" * 30


RULES = """\
# Avalon

Avalon is a hidden-role game. Each player is secretly dealt a **role** (see
Roles). Every role is either **Good** or **Evil**, and some grant special
knowledge. Players are seated at labelled seats (A, B, C, ...). Good wins by
completing Quests; Evil wins by sabotaging them or by assassinating Merlin.

## Flow
The game runs up to 5 Quests. For each Quest:
  1. The leader proposes a team of a fixed size.
  2. Players discuss (see below), then everyone votes publicly to approve or
     reject the team. A strict majority approves.
  3. If rejected, leadership passes to the next seat and a new team is proposed.
     Five rejected proposals in a row on one Quest -> Evil wins immediately.
  4. If approved, the team goes on the Quest. Each team member secretly plays a
     Success or Fail card. Good players always play Success; Evil players may
     play either. The Quest fails if it receives the required number of Fail
     cards (usually one). Only the number of Fails is revealed, never who played
     them.

### Discussion
Before each vote, players speak in a chain: the leader (proposer) speaks first,
and each speaker names who speaks next (you cannot nominate yourself). You may
speak only a limited number of times per proposal; once your turns are used you
cannot speak again on that proposal.

## Winning
  - Good completes 3 Quests -> Good is winning, but the Assassin then names one
    player as Merlin. If correct, Evil wins instead; if wrong, Good wins.
  - Evil fails 3 Quests -> Evil wins.
  - Five consecutive rejected proposals on one Quest -> Evil wins.

## Roles
Only some are in play each game; you are told which, and which you are.

**Good**
  - Merlin: knows which players are Evil (except Mordred, if present). Must stay
    hidden -- if the Assassin identifies Merlin, Evil wins.
  - Percival: sees Merlin and Morgana but cannot tell which is which.
  - Loyal Servant: no special knowledge.

**Evil** (Evil players know each other, except Oberon)
  - Assassin: at the end, if Good has won, names a player as Merlin.
  - Morgana: appears as Merlin to Percival.
  - Mordred: hidden from Merlin.
  - Oberon: does not know the other Evil players, and they do not know him.
  - Minion: no special power.

# You

## Objective
You play many games. Your aim is to win as many as you can. Between games you
keep private notes and can see your win-rate trajectory; use them to learn and
adapt. How to play well is for you to work out.

## Visibility
Each of your outputs is tagged with one of these:
  - **public**: every player sees it.
  - **private**: only you ever see it.
  - **secret**: your individual choice is hidden from everyone; only an aggregate
    (e.g. the number of Fail cards) is revealed."""


def format_win_rate(series: list) -> str:
  """Rolling win-rate trajectory (oldest to newest); empty if no games yet."""
  if not series:
    return "## Win rate\nNo games completed yet."
  pts = ", ".join(f"{r:.0%}" for r in series)
  return f"## Win rate\nRolling over recent games, oldest to newest: {pts}"


def build_system_prompt(
  notes: str = "",
  recent_games: list | None = None,
  win_rate_series: list | None = None,
) -> str:
  """Everything fixed for the whole game, ordered general -> specific."""
  parts = [RULES, format_win_rate(win_rate_series or [])]
  if notes.strip():
    parts.append("## Notes\nCarried from your past games.\n\n" + notes.strip())
  if recent_games:
    joined = f"\n\n{GAME_SEPARATOR}\n\n".join(recent_games)
    parts.append("## Past games\nMost recent last.\n\n" + joined)
  return "\n\n".join(parts)


def build_user_prompt(log: str, ask: str) -> str:
  """What changes per call: the current game so far, then the decision ask."""
  return (
    "# Now\n"
    "You are in a live Avalon game. Below is the game so far, then the "
    "decision you must make.\n\n"
    f"## Current game\n{log}\n\n"
    f"## Your decision\n{ask}"
  )


# --- per-decision asks (parameters the schema doesn't carry) ---


def ask_propose(team_size: int) -> str:
  return f"You are the leader. Propose a team of {team_size}."


def ask_speak(can_nominate: list) -> str:
  elig = ", ".join(labels(can_nominate)) if can_nominate else "(none)"
  return f"Speak to the table, then name who speaks next. You may nominate: {elig}."


def ask_vote(proposed_team) -> str:
  team = ", ".join(labels(proposed_team))
  return f"Vote to approve or reject this proposed team: {team}."


def ask_quest() -> str:
  # only Evil are asked (Good must succeed and are not called)
  return "You are on the mission. Play Success or Fail."


def ask_assassinate() -> str:
  return "Good has completed 3 Quests. As the Assassin, name the player you believe is Merlin."


def ask_debrief() -> str:
  return "The game is over and all roles are revealed. Record notes to carry into your future games."
