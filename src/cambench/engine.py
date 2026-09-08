"""The deterministic Avalon state machine: phases, rule enforcement, info hiding, logging."""

from __future__ import annotations

import random

from .records import ALL, Event, EventType, GameRecord
from .labels import labels, seat_label
from .roles import (
  MAX_CONSECUTIVE_REJECTS,
  TURNS_PER_SEAT,
  QUESTS_TO_LOSE,
  QUESTS_TO_WIN,
  GameConfig,
  Role,
  alignment_of,
  barebones_config,
  compute_knowledge,
  is_evil,
)
from .player import Decision, Seat, Utterance, Vote


class AvalonGame:
  def __init__(
    self,
    players: list,
    config: GameConfig | None = None,
    game_id: str = "game",
    seed: int | None = None,
    on_event: object = None,
  ):
    n = len(players)
    self.config = config or barebones_config(n)
    if self.config.num_players != n:
      raise ValueError(f"config is for {self.config.num_players} players but got {n}")
    self.num_players = n
    self.players = players
    self.game_id = game_id
    self._rng = random.Random(seed)

    self.team_sizes = self.config.team_sizes
    self.fails_required = self.config.fails_required

    # assign roles (god view, hidden from agents)
    roles = self.config.role_list()
    self._rng.shuffle(roles)
    self.assignment: dict = dict(enumerate(roles))
    self.knowledge = {s: compute_knowledge(s, self.assignment) for s in range(n)}

    self.record = GameRecord(
      game_id=game_id,
      num_players=n,
      assignment=dict(self.assignment),
      initial_knowledge=dict(self.knowledge),
      on_event=on_event,
    )

    # running public state
    self.current_quest = 0
    self.successes = 0
    self.fails = 0
    self.leader = self._rng.randrange(n)
    self.consecutive_rejects = 0
    self.proposed_team: tuple | None = None

    self._log_game_start()

  # View construction (information hiding)                              #

  # --- per-seat context + rendered log (information hiding) ---
  def _ctx(self, seat: int) -> Seat:
    role = self.assignment[seat]
    return Seat(
      seat=seat,
      num_players=self.num_players,
      role=role,
      alignment=alignment_of(role),
      knowledge=self.knowledge[seat],
    )

  def _log(self, seat: int) -> str:
    return self.record.render(seat)

  @staticmethod
  def _unwrap(d):
    """Accept a Decision/Vote (or bare value); return (value, reasoning)."""
    if isinstance(d, (Decision, Vote)):
      return d.value, d.reasoning
    return d, ""

  # --- logging helpers ---
  def _log_game_start(self) -> None:
    private = {}
    for s in range(self.num_players):
      k = self.knowledge[s]
      block = {
        "your_seat": seat_label(s),
        "your_role": self.assignment[s].value,
        "your_alignment": alignment_of(self.assignment[s]).value,
        "known_evil": labels(sorted(k.known_evil)),
      }
      if k.merlin_candidates:
        block["merlin_candidates"] = labels(sorted(k.merlin_candidates))
      private[s] = block
    self.record.add(
      Event(
        type=EventType.START,
        payload={
          "num_players": self.num_players,
          "seats": labels(range(self.num_players)),
          "quest_team_sizes": list(self.team_sizes),
          "fails_required": list(self.fails_required),
          "roles_in_play": sorted(r.value for r in self.assignment.values()),
          "starting_leader": seat_label(self.leader),
        },
        visible_to=ALL,
        private=private,
      )
    )

  # Action collection with validation / coercion                      #

  # --- action collection (validation / coercion) ---
  def _get_proposal(self) -> tuple:
    size = self.team_sizes[self.current_quest]
    raw, reasoning = self._unwrap(self.players[self.leader].propose_team(self._log(self.leader), self._ctx(self.leader), size))
    team = self._coerce_team(raw, size)
    self.proposed_team = team
    self.record.add(
      Event(
        type=EventType.PROPOSE,
        payload={
          "quest_index": self.current_quest,
          "leader": seat_label(self.leader),
          "team": labels(team),
        },
        private={self.leader: {"reasoning": reasoning}} if reasoning else {},
      )
    )
    return team

  def _coerce_team(self, raw, size: int) -> tuple:
    seen: list = []
    if isinstance(raw, (list, tuple)):
      for x in raw:
        if isinstance(x, int) and 0 <= x < self.num_players and x not in seen:
          seen.append(x)
    if len(seen) == size:
      return tuple(seen)
    for s in range(self.num_players):
      if len(seen) == size:
        break
      if s not in seen:
        seen.append(s)
    return tuple(seen[:size])

  def _run_discussion(self, team: tuple) -> None:
    cap = TURNS_PER_SEAT
    turns = {s: 0 for s in range(self.num_players)}

    # the proposer (current leader) always opens discussion
    current = self.leader

    while current is not None:
      # seats this speaker may nominate: under cap, not self
      can_nominate = [s for s in range(self.num_players) if s != current and turns[s] < cap]
      utt = self.players[current].speak(self._log(current), self._ctx(current), list(can_nominate))
      if not isinstance(utt, Utterance):
        utt = Utterance()
      turns[current] += 1

      # who did they nominate? (public directional signal, may be overridden)
      nominated = utt.next_speaker if isinstance(utt.next_speaker, int) and 0 <= utt.next_speaker < self.num_players else None
      self._log_speech(current, utt, nominated)

      # announce once, when this speech puts the speaker at the cap
      if turns[current] == cap:
        self.record.add(
          Event(
            type=EventType.CAP,
            payload={"seat": seat_label(current), "turns": cap},
          )
        )

      # route: nominee if eligible, else a random eligible seat, else end.
      # eligible = under cap and not the current speaker. Discussion can only
      # end when everyone else is capped -- which forces a frozen-out seat's
      # one closing turn.
      eligible = [s for s in range(self.num_players) if s != current and turns[s] < cap]
      if not eligible:
        current = None
      elif nominated in eligible:
        current = nominated
      else:
        current = self._rng.choice(eligible)

  def _log_speech(self, seat: int, utt: Utterance, nominated: int | None = None) -> None:
    payload = {
      "quest_index": self.current_quest,
      "seat": seat_label(seat),
      "statement": utt.statement,
    }
    if nominated is not None:
      payload["next"] = seat_label(nominated)
    self.record.add(
      Event(
        type=EventType.SPEECH,
        payload=payload,
        visible_to=ALL,
        private={seat: {"reasoning": utt.reasoning}} if utt.reasoning else {},
      )
    )

  def _collect_votes(self, team: tuple, attempt: int) -> bool:
    votes: dict = {}
    reasons: dict = {}
    for seat in range(self.num_players):
      result = self.players[seat].vote(self._log(seat), self._ctx(seat), team)
      v, r = self._unwrap(result)
      votes[seat] = bool(v)
      belief = getattr(result, "belief", None)
      entry = {}
      if r:
        entry["reasoning"] = r
      if belief:
        entry["belief"] = {seat_label(s): role.value for s, role in belief.items()}
      if entry:
        reasons[seat] = entry
    approvals = sum(1 for v in votes.values() if v)
    approved = approvals > self.num_players // 2

    self.record.add(
      Event(
        type=EventType.VOTE,
        payload={
          "quest_index": self.current_quest,
          "attempt": attempt,
          "leader": seat_label(self.leader),
          "team": labels(team),
          "votes": {seat_label(s): v for s, v in votes.items()},
          "approvals": approvals,
          "approved": approved,
        },
        private=reasons,
      )
    )
    return approved

  def _run_quest(self, team: tuple) -> bool:
    fails = 0
    qreasons: dict = {}
    for seat in team:
      if not is_evil(self.assignment[seat]):
        continue  # good must succeed (rulebook); not asked, not a choice
      c, r = self._unwrap(self.players[seat].quest_card(self._log(seat), self._ctx(seat)))
      if not c:
        fails += 1
      if r:
        qreasons[seat] = {"reasoning": r}
    required = self.fails_required[self.current_quest]
    success = fails < required

    self.record.add(
      Event(
        type=EventType.QUEST,
        payload={
          "quest_index": self.current_quest,
          "team": labels(team),
          "num_fails": fails,
          "fails_required": required,
          "success": success,
        },
        private=qreasons,
      )
    )
    return success

  # Main loop                                                          #

  # --- main loop ---
  def play(self) -> GameRecord:
    while self.successes < QUESTS_TO_WIN and self.fails < QUESTS_TO_LOSE:
      _, hammered = self._run_one_quest_proposals()
      if hammered:
        self._finish(winner="evil", reason="five_consecutive_rejects")
        return self.record

      if self._run_quest(self.proposed_team):
        self.successes += 1
      else:
        self.fails += 1

      self.leader = (self.leader + 1) % self.num_players
      self.current_quest += 1

    if self.fails >= QUESTS_TO_LOSE:
      self._finish(winner="evil", reason="three_failed_quests")
    else:
      self._run_assassination()
    return self.record

  def _run_one_quest_proposals(self) -> tuple:
    self.consecutive_rejects = 0
    attempt = 0
    while True:
      team = self._get_proposal()
      self._run_discussion(team)
      if self._collect_votes(team, attempt):
        self.consecutive_rejects = 0
        return True, False
      self.consecutive_rejects += 1
      self.leader = (self.leader + 1) % self.num_players
      attempt += 1
      if self.consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
        return False, True

  def _run_assassination(self) -> None:
    assassin_seat = next(s for s, r in self.assignment.items() if r is Role.ASSASSIN)
    raw_target, reasoning = self._unwrap(
      self.players[assassin_seat].assassinate(self._log(assassin_seat), self._ctx(assassin_seat))
    )
    target = raw_target if isinstance(raw_target, int) and 0 <= raw_target < self.num_players else 0
    hit_merlin = self.assignment[target] is Role.MERLIN

    self.record.add(
      Event(
        type=EventType.ASSASSINATE,
        payload={
          "assassin": seat_label(assassin_seat),
          "target": seat_label(target),
          "hit_merlin": hit_merlin,
        },
        private={assassin_seat: {"reasoning": reasoning}} if reasoning else {},
      )
    )
    if hit_merlin:
      self._finish(winner="evil", reason="merlin_assassinated")
    else:
      self._finish(winner="good", reason="quests_won_merlin_safe")

  def _finish(self, winner: str, reason: str) -> None:
    self.record.outcome = {
      "winner": winner,
      "reason": reason,
      "successes": self.successes,
      "fails": self.fails,
      "roles": {seat_label(s): r.value for s, r in self.assignment.items()},
    }
    self.record.add(
      Event(
        type=EventType.END,
        payload=dict(self.record.outcome),
        visible_to=ALL,
      )
    )

  def run_debrief(self) -> None:
    """After the game, ask each seat for a private post-reveal reflection."""
    for seat in range(self.num_players):
      note = self.players[seat].debrief(self._log(seat), self._ctx(seat))
      if note:
        self.record.add(
          Event(
            type=EventType.DEBRIEF,
            payload={"seat": seat_label(seat)},
            visible_to={seat},
            private={seat: {"reasoning": note}},
          )
        )


def play_game(
  players: list,
  config: GameConfig | None = None,
  game_id: str = "game",
  seed: int | None = None,
) -> GameRecord:
  return AvalonGame(players, config=config, game_id=game_id, seed=seed).play()
