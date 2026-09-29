"""Guard Avalon mechanics that a platform integration must not change."""

from collections import Counter

import pytest

from cambench.engine import AvalonGame, RandomBot, recommended_config
from cambench.engine.labels import using_names
from cambench.engine.player import Decision, Utterance, Vote
from cambench.engine.roles import Role, compute_knowledge, is_evil


def test_special_role_knowledge():
  assignment = dict(enumerate([
    Role.MERLIN, Role.PERCIVAL, Role.MORGANA, Role.MORDRED, Role.OBERON,
    Role.ASSASSIN, Role.MINION, Role.LOYAL_SERVANT,
  ]))
  assert compute_knowledge(0, assignment).known_evil == {2, 4, 5, 6}
  assert compute_knowledge(1, assignment).merlin_candidates == {0, 2}
  assert compute_knowledge(4, assignment).known_evil == set()
  assert compute_knowledge(5, assignment).known_evil == {2, 3, 6}
  assert compute_knowledge(7, assignment).known_evil == set()


class ScriptedPlayer(RandomBot):
  def __init__(self, *, approve=True, sabotage=False, hit=False):
    super().__init__(0)
    self.approve = approve
    self.sabotage = sabotage
    self.hit = hit
    self.game = None

  def propose_team(self, log, ctx, team_size):
    evil = [seat for seat, role in self.game.assignment.items() if is_evil(role)]
    rest = [seat for seat in range(ctx.num_players) if seat not in evil]
    return Decision((evil + rest)[:team_size], f"SECRET-PROPOSAL-{ctx.seat}")

  def speak(self, log, ctx, can_nominate):
    return Utterance("test speech", f"SECRET-SPEECH-{ctx.seat}", can_nominate[0] if can_nominate else None)

  def vote(self, log, ctx, team):
    current = log.rsplit("PROPOSE", 1)[-1]
    assert "VOTE" not in current  # all voters see the same pre-reveal state
    return Vote(Decision(self.approve, f"SECRET-VOTE-{ctx.seat}"), {0: Role.MERLIN})

  def quest_card(self, log, ctx):
    assert ctx.is_evil()  # good players must never be asked
    current = log.rsplit("VOTE", 1)[-1]
    assert "MISSION" not in current
    return Decision(not self.sabotage, f"SECRET-CARD-{ctx.seat}")

  def assassinate(self, log, ctx):
    target = next(s for s, r in self.game.assignment.items() if (r == Role.MERLIN) == self.hit)
    return Decision(target, "SECRET-ASSASSINATION")


@pytest.mark.parametrize("approve,sabotage,hit,reason", [
  (False, False, False, "five_consecutive_rejects"),
  (True, True, False, "three_failed_quests"),
  (True, False, True, "merlin_assassinated"),
  (True, False, False, "quests_won_merlin_safe"),
])
def test_terminal_paths_and_private_overlays(approve, sabotage, hit, reason):
  players = [ScriptedPlayer(approve=approve, sabotage=sabotage, hit=hit) for _ in range(10)]
  with using_names(list("ABCDEFGHIJ")):
    game = AvalonGame(players, config=recommended_config(10), seed=73)
    for player in players:
      player.game = game
    record = game.play()
    assert record.outcome["reason"] == reason
    for seat in range(10):
      observation = record.render(seat)
      for other in range(10):
        if other != seat:
          assert all(f"SECRET-{action}-{other}" not in observation for action in ("PROPOSAL", "SPEECH", "VOTE", "CARD"))
    turns = Counter()
    for event in record.events:
      if event.type.value == "propose":
        turns.clear()
      elif event.type.value == "speak":
        turns[event.payload["seat"]] += 1
        assert turns[event.payload["seat"]] <= 3


@pytest.mark.parametrize("count", range(5, 11))
def test_fourth_quest_threshold(count):
  config = recommended_config(count)
  assert config.fails_required == ((1, 1, 1, 2, 1) if count >= 7 else (1, 1, 1, 1, 1))
  players = [ScriptedPlayer() for _ in range(count)]
  game = AvalonGame(players, config=config, seed=2)
  game.current_quest = 3
  evil = [s for s, role in game.assignment.items() if is_evil(role)]
  players[evil[0]].sabotage = True
  assert game._run_quest(tuple(range(count))) == (count >= 7)
