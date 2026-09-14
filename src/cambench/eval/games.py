"""Game-level extraction: one raw row per game. Aggregation/plots live in the notebook."""

from __future__ import annotations

from ._io import read_records


def game_table(run_dir: str) -> list:
  """One dict per game (in order) with whole-game facts, for the notebook to slice.

  Fields:
      game           -- 1-based game index
      num_players    -- players in the game
      winner         -- "good" or "evil"
      reason         -- how it ended (three_quests, three_failed_quests,
                        merlin_assassinated, hammer_rejected, ...)
      assassinated   -- True if good was undone by the assassination
      proposals      -- team proposals put to a vote (a game-length measure)
      rejections     -- proposals that were voted down
      quests_played  -- quests that reached a card play
      successes      -- successful quests
      fails          -- failed quests
      total_fails    -- total Fail cards played across all quests
  """
  rows = []
  for idx, rec in read_records(run_dir):
    events = rec["events"]
    quests = [e for e in events if e.get("type") == "quest"]
    votes = [e for e in events if e.get("type") == "vote"]

    proposals = len(votes)
    rejections = sum(1 for e in votes if not e["payload"].get("approved"))

    rows.append(
      {
        "game": idx,
        "num_players": rec["num_players"],
        "winner": rec["winner"],
        "reason": rec["reason"],
        "assassinated": rec["reason"] == "merlin_assassinated",
        "proposals": proposals,
        "rejections": rejections,
        "quests_played": len(quests),
        "successes": rec.get("successes"),
        "fails": rec.get("fails"),
        "total_fails": sum(e["payload"].get("num_fails", 0) for e in quests),
      }
    )
  return rows


def hammer_blunders(run_dir: str) -> dict:
  """Per seat: rejections of the final proposal in a game lost to five straight
  rejections. A good player voting reject there throws the game outright, so it
  is an unambiguous strategic error.

  Returns:
      {"model@effort#seat": {"blunders": int, "hammer_games": int}}, where
      hammer_games is the five-reject games the seat played as good (the
      denominator) and blunders is those in which it rejected the final proposal.
  """
  from ._io import seat_keys, side_of

  keys_by_game = seat_keys(run_dir)
  out: dict = {}
  for idx, rec in read_records(run_dir):
    if rec["reason"] != "five_consecutive_rejects":
      continue
    truth = rec["assignment"]
    keys = keys_by_game.get(idx, {})
    votes = [e for e in rec["events"] if e.get("type") == "vote"]
    if not votes:
      continue
    final_votes = max(votes, key=lambda e: e["payload"].get("attempt", 0))["payload"].get("votes", {})
    for name, role in truth.items():
      if side_of(role) != "good":
        continue
      key = keys.get(name, name)
      cell = out.setdefault(key, {"blunders": 0, "hammer_games": 0})
      cell["hammer_games"] += 1
      if final_votes.get(name) is False:
        cell["blunders"] += 1
  return out
