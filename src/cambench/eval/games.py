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
      approvals      -- proposals that passed
      approval_rate  -- approvals / proposals
      max_rejects    -- most rejections seen on a single quest (contentiousness)
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
    approvals = sum(1 for e in votes if e["payload"].get("approved"))
    rejections = proposals - approvals
    rejects_by_quest: dict = {}
    for e in votes:
      if not e["payload"].get("approved"):
        qi = e["payload"].get("quest_index")
        rejects_by_quest[qi] = rejects_by_quest.get(qi, 0) + 1
    max_rejects = max(rejects_by_quest.values()) if rejects_by_quest else 0

    rows.append(
      {
        "game": idx,
        "num_players": rec["num_players"],
        "winner": rec["winner"],
        "reason": rec["reason"],
        "assassinated": rec["reason"] == "merlin_assassinated",
        "proposals": proposals,
        "rejections": rejections,
        "approvals": approvals,
        "approval_rate": approvals / proposals if proposals else 0.0,
        "max_rejects": max_rejects,
        "quests_played": len(quests),
        "successes": rec.get("successes"),
        "fails": rec.get("fails"),
        "total_fails": sum(e["payload"].get("num_fails", 0) for e in quests),
      }
    )
  return rows
