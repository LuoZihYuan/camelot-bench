"""Seat-level metrics over a run's records: win-rate, belief-accuracy, hidden-rate."""

from __future__ import annotations

from ._io import (
  read_ledger as _read_ledger,
  read_records as _read_records,
  seat_key as _seat_key,
  seat_keys as _seat_keys,
  side_of as _side_of,
  groups as _groups,
  wilson as _wilson,
  rolling as _rolling,
)


# --- belief-accuracy ladder ---


def _snapshot_belief_accuracy(ev: dict, guesser: str, truth: dict, prior: set):
  """(exact_frac, align_frac) for one guesser's belief in one vote.

  Scored over every eligible target (all others minus prior-known); a target the
  guesser did not guess counts as a miss on both role and side. None if there are
  no eligible targets.
  """
  belief = ev.get("private", {}).get(guesser, {}).get("belief") or {}
  exact = align = n = 0
  for target, t_role in truth.items():
    if target == guesser or target in prior:
      continue
    guess = belief.get(target)  # None if the guesser omitted this target
    exact += int(guess == t_role)
    align += int(guess is not None and _side_of(guess) == _side_of(t_role))
    n += 1
  if n == 0:
    return None
  return (exact / n, align / n)


def _aggregate_game(snaps_by_subject: dict, seat_keys: dict, subject_role, by: str | None) -> dict:
  """Fold per-subject snapshot lists into one (role, side) game score each.

  snaps_by_subject: {subject_name: [(role_frac, side_frac), ...]}.
  subject_role(name) -> the subject's role (for `by` grouping).
  Returns {seat_key: {group: (role_frac, side_frac)}}.
  """
  buckets: dict = {}  # (seat_key, group) -> [(role, side), ...]
  for subject, snaps in snaps_by_subject.items():
    for g in _groups(subject_role(subject), by):
      buckets.setdefault((seat_keys[subject], g), []).extend(snaps)
  out: dict = {}
  for (key, g), s in buckets.items():
    out.setdefault(key, {})[g] = (sum(x[0] for x in s) / len(s), sum(x[1] for x in s) / len(s))
  return out


def _game_belief_accuracy(rec: dict, seat_keys: dict, by: str | None) -> dict:
  """One game's per-seat belief score: each guesser scored on its own guesses."""
  truth = rec["assignment"]
  known = rec["initial_knowledge"]
  snaps: dict = {}  # guesser -> [(role, side) per vote]
  for ev in rec["events"]:
    if ev.get("type") != "vote":
      continue
    for guesser, ov in ev.get("private", {}).items():
      if "belief" not in ov:
        continue
      prior = set(known.get(guesser, {}).get("known_evil", [])) | set(known.get(guesser, {}).get("merlin_candidates", []))
      s = _snapshot_belief_accuracy(ev, guesser, truth, prior)
      if s is not None:
        snaps.setdefault(guesser, []).append(s)
  return _aggregate_game(snaps, seat_keys, truth.get, by)


def _snapshot_hidden(ev: dict, target: str, truth: dict, known: dict):
  """(role_frac, side_frac) for how well `target` stayed HIDDEN in one vote --
  the fraction of eligible guessers who failed to identify it.

  Every eligible guesser counts (all others who were NOT told about `target`).
  A guesser who omitted `target` counts as failing to identify it (hidden on
  both). None if there are no eligible guessers.
  """
  t_role = truth.get(target)
  if t_role is None:
    return None
  role_miss = side_miss = n = 0
  for guesser, ov in ev.get("private", {}).items():
    if guesser == target or "belief" not in ov:
      continue
    prior = set(known.get(guesser, {}).get("known_evil", [])) | set(known.get(guesser, {}).get("merlin_candidates", []))
    if target in prior:  # this guesser was told about the target
      continue
    guess = (ov.get("belief") or {}).get(target)  # None if omitted -> stays hidden
    role_miss += int(guess != t_role)
    side_miss += int(guess is None or _side_of(guess) != _side_of(t_role))
    n += 1
  if n == 0:
    return None
  return (role_miss / n, side_miss / n)


def _game_hidden(rec: dict, seat_keys: dict, by: str | None) -> dict:
  """One game's per-seat hidden-rate: each target scored on how well it stayed hidden."""
  truth = rec["assignment"]
  known = rec["initial_knowledge"]
  snaps: dict = {}  # target -> [(role, side) per vote]
  for ev in rec["events"]:
    if ev.get("type") != "vote":
      continue
    for target in truth:  # every seat is a possible target
      s = _snapshot_hidden(ev, target, truth, known)
      if s is not None:
        snaps.setdefault(target, []).append(s)
  return _aggregate_game(snaps, seat_keys, truth.get, by)


# --- public metrics ---


def win_rate(run_dir: str, by: str | None = None) -> dict:
  """Win rate per seat, with Wilson 95% confidence intervals.

  Args:
      run_dir: a run directory (reads results.jsonl).
      by: split each seat's games -- None (overall), "side", or "role".

  Returns:
      {"model@effort#seat": {group: {win_rate, ci_low, ci_high, wins, n}}}.
  """
  tally: dict = {}  # seat_key -> group -> [wins, n]
  for row in _read_ledger(run_dir):
    winner = row["winner"]
    roles = row["roles"]
    for name, info in row["seats"].items():
      key = _seat_key(name, info)
      role = roles[name]
      won = int(_side_of(role) == winner)
      for g in _groups(role, by):
        cell = tally.setdefault(key, {}).setdefault(g, [0, 0])
        cell[0] += won
        cell[1] += 1
  out: dict = {}
  for key, groups in tally.items():
    out[key] = {}
    for g, (wins, n) in groups.items():
      lo, hi = _wilson(wins, n)
      out[key][g] = {"win_rate": wins / n if n else 0.0, "ci_low": lo, "ci_high": hi, "wins": wins, "n": n}
  return out


def belief_accuracy(run_dir: str, by: str | None = None) -> dict:
  """Belief accuracy per seat (macro: snapshot -> game -> run mean).

  Each guess is scored two ways -- exact (character) and align (good/evil).
  Guesses about
  players the guesser was told about at the start are excluded (deduction, not
  recall). Each game contributes once (mean of its snapshots), rewarding
  reading the table correctly early and weighting games equally.

  Args:
      run_dir: a run directory (reads records/ + results.jsonl).
      by: split each seat's games -- None (overall), "side", or "role".

  Returns:
      {"model@effort#seat": {group: {exact, align, n_games}}}.
  """
  seat_keys_by_game = _seat_keys(run_dir)
  acc: dict = {}  # seat_key -> group -> [(exact, align) per game]
  for idx, rec in _read_records(run_dir):
    game = _game_belief_accuracy(rec, seat_keys_by_game.get(idx, {}), by)
    for key, groups in game.items():
      for g, score in groups.items():
        acc.setdefault(key, {}).setdefault(g, []).append(score)
  out: dict = {}
  for key, groups in acc.items():
    out[key] = {}
    for g, games in groups.items():
      out[key][g] = {
        "exact": sum(x[0] for x in games) / len(games),
        "align": sum(x[1] for x in games) / len(games),
        "n_games": len(games),
      }
  return out


def win_rate_trajectory(run_dir: str, by: str | None = None, window: int = 10) -> dict:
  """Per-seat rolling win-rate over games (expanding-then-sliding).

  Returns:
      {"model@effort#seat": {group: [rolling win-rate per game, oldest to newest]}}.
  """
  seq: dict = {}  # seat_key -> group -> [win booleans in game order]
  for row in sorted(_read_ledger(run_dir), key=lambda r: r["game_id"]):
    winner = row["winner"]
    roles = row["roles"]
    for name, info in row["seats"].items():
      key = _seat_key(name, info)
      role = roles[name]
      won = int(_side_of(role) == winner)
      for g in _groups(role, by):
        seq.setdefault(key, {}).setdefault(g, []).append(won)
  return {key: {g: _rolling(wins, window) for g, wins in groups.items()} for key, groups in seq.items()}


def belief_accuracy_trajectory(run_dir: str, by: str | None = None, window: int = 10) -> dict:
  """Per-seat rolling belief accuracy over games (expanding-then-sliding).

  Returns:
      {"model@effort#seat": {group: {"exact": [series], "align": [series]}}}.
  """
  seat_keys_by_game = _seat_keys(run_dir)
  ex_seq: dict = {}
  al_seq: dict = {}
  for idx, rec in _read_records(run_dir):
    game = _game_belief_accuracy(rec, seat_keys_by_game.get(idx, {}), by)
    for key, groups in game.items():
      for g, (ex, al) in groups.items():
        ex_seq.setdefault(key, {}).setdefault(g, []).append(ex)
        al_seq.setdefault(key, {}).setdefault(g, []).append(al)
  out: dict = {}
  for key in ex_seq:
    out[key] = {}
    for g in ex_seq[key]:
      out[key][g] = {"exact": _rolling(ex_seq[key][g], window), "align": _rolling(al_seq[key][g], window)}
  return out


def hidden_rate(run_dir: str, by: str | None = None) -> dict:
  """How well hidden each seat is: how often OTHERS fail to identify it (macro:
  vote -> game -> run mean). Higher = better hidden. Mirror of belief_accuracy.

  Guesses by players who were told about the target are excluded (deduction,
  not recall). Each game contributes once (mean of its votes).

  Args:
      run_dir: a run directory (reads records/ + results.jsonl).
      by: split each seat's games -- None (overall), "side", or "role".

  Returns:
      {"model@effort#seat": {group: {exact, align, n_games}}}, where exact/align are
      how often others got this seat's exact role / alignment WRONG.
  """
  seat_keys_by_game = _seat_keys(run_dir)
  acc: dict = {}
  for idx, rec in _read_records(run_dir):
    game = _game_hidden(rec, seat_keys_by_game.get(idx, {}), by)
    for key, groups in game.items():
      for g, score in groups.items():
        acc.setdefault(key, {}).setdefault(g, []).append(score)
  out: dict = {}
  for key, groups in acc.items():
    out[key] = {}
    for g, games in groups.items():
      out[key][g] = {
        "exact": sum(x[0] for x in games) / len(games),
        "align": sum(x[1] for x in games) / len(games),
        "n_games": len(games),
      }
  return out


def hidden_rate_trajectory(run_dir: str, by: str | None = None, window: int = 10) -> dict:
  """Per-seat rolling hidden-rate over games (expanding-then-sliding).

  Returns:
      {"model@effort#seat": {group: {"exact": [series], "align": [series]}}}.
  """
  seat_keys_by_game = _seat_keys(run_dir)
  role_seq: dict = {}
  side_seq: dict = {}
  for idx, rec in _read_records(run_dir):
    game = _game_hidden(rec, seat_keys_by_game.get(idx, {}), by)
    for key, groups in game.items():
      for g, (r, s) in groups.items():
        role_seq.setdefault(key, {}).setdefault(g, []).append(r)
        side_seq.setdefault(key, {}).setdefault(g, []).append(s)
  out: dict = {}
  for key in role_seq:
    out[key] = {}
    for g in role_seq[key]:
      out[key][g] = {"exact": _rolling(role_seq[key][g], window), "align": _rolling(side_seq[key][g], window)}
  return out


def role_counts(run_dir: str) -> dict:
  """How many times each seat was dealt each role.

  Returns:
      {"model@effort#seat": {role: count}} -- a role-assignment tally per seat,
      useful for checking that roles were distributed fairly across seats.
  """
  out: dict = {}
  for row in _read_ledger(run_dir):
    roles = row["roles"]
    for name, info in row["seats"].items():
      key = _seat_key(name, info)
      role = roles[name]
      out.setdefault(key, {})[role] = out.setdefault(key, {}).get(role, 0) + 1
  return out


def assassination(run_dir: str) -> dict:
  """Per-seat assassination outcomes at the end-game kill.

  Only games that reached an assassination count. For each seat:
    as_assassin -- {hits, n}: times it correctly picked Merlin / times it was
                   the assassin.
    as_merlin   -- {survived, n}: times it avoided the kill / times it was
                   Merlin facing the assassin.

  Returns {"model@effort#seat": {"as_assassin": {...}, "as_merlin": {...}}}.
  Sparse at low game counts -- each seat is assassin or Merlin only occasionally.
  """
  seat_keys_by_game = _seat_keys(run_dir)
  out: dict = {}
  for idx, rec in _read_records(run_dir):
    ev = next((e for e in rec["events"] if e.get("type") == "assassinate"), None)
    if ev is None:
      continue
    keys = seat_keys_by_game.get(idx, {})
    truth = rec["assignment"]
    hit = ev["payload"]["hit_merlin"]
    assassin = ev["payload"]["assassin"]
    merlin = next((n for n, r in truth.items() if r == "merlin"), None)
    if assassin in keys:
      cell = out.setdefault(keys[assassin], {}).setdefault("as_assassin", {"hits": 0, "n": 0})
      cell["hits"] += int(hit)
      cell["n"] += 1
    if merlin in keys:
      cell = out.setdefault(keys[merlin], {}).setdefault("as_merlin", {"survived": 0, "n": 0})
      cell["survived"] += int(not hit)
      cell["n"] += 1
  return out
