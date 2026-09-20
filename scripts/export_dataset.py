"""Export a run into a flat, documented release dataset (CSV + JSONL) via the eval readers."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from datetime import datetime, timezone

from cambench.eval._io import read_ledger, read_records, side_of


# --- small writers -------------------------------------------------------


def write_csv(path: pathlib.Path, rows: list, fields: list) -> None:
  with path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
      w.writerow({k: _cell(r.get(k)) for k in fields})


def write_jsonl(path: pathlib.Path, rows: list) -> None:
  with path.open("w", encoding="utf-8") as f:
    for r in rows:
      f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _cell(v):
  """CSV cell: serialize lists (e.g. team seats) as a JSON string, pass scalars."""
  if isinstance(v, (list, dict)):
    return json.dumps(v, ensure_ascii=False)
  return v


# --- per-table extraction ------------------------------------------------


def extract(run_dir: str):
  records = read_records(run_dir)  # [(game_index, parsed_record)]

  games, seats, proposals, votes, quests = [], [], [], [], []
  sabotage, guesses, speeches, assassination, debriefs = [], [], [], [], []

  for idx, rec in records:
    truth = rec["assignment"]  # seat -> role
    known = rec.get("initial_knowledge", {})  # seat -> {known_evil, merlin_candidates}
    merlin_seat = next((s for s, r in truth.items() if r == "merlin"), None)

    # games
    games.append(
      {
        "game": idx,
        "num_players": rec["num_players"],
        "winner_side": rec.get("winner"),
        "reason": rec.get("reason"),
        "num_successes": rec.get("successes"),
        "num_fails": rec.get("fails"),
      }
    )

    # seats (per-game role assignment)
    for seat, role in truth.items():
      k = known.get(seat, {})
      seats.append(
        {
          "game": idx,
          "seat": seat,
          "role": role,
          "side": side_of(role),
          "won": _won(role, rec.get("winner")),
          "known_evil_seats": k.get("known_evil", []),
          "merlin_candidate_seats": k.get("merlin_candidates", []),
        }
      )

    # walk events; track attempt within each quest (attempt lives only on votes)
    attempt_by_quest: dict = {}
    turn_by_key: dict = {}
    for ev in rec["events"]:
      t = ev.get("type")
      p = ev.get("payload", {})
      priv = ev.get("private", {})
      q = p.get("quest_index")

      if t == "propose":
        a = attempt_by_quest.get(q, 0)
        proposals.append(
          {
            "game": idx,
            "quest": q,
            "attempt": a,
            "proposer_seat": p.get("leader"),
            "team_seats": p.get("team", []),
            "approved": None,
            "num_approvals": None,  # filled by the matching vote
            "reasoning": priv.get(p.get("leader"), {}).get("reasoning"),
          }
        )

      elif t == "speak":
        key = (q, attempt_by_quest.get(q, 0))
        turn = turn_by_key.get(key, 0)
        turn_by_key[key] = turn + 1
        seat = p.get("seat")
        speeches.append(
          {
            "game": idx,
            "quest": q,
            "attempt": attempt_by_quest.get(q, 0),
            "turn": turn,
            "speaker_seat": seat,
            "next_seat": p.get("next"),
            "statement": p.get("statement"),
            "reasoning": priv.get(seat, {}).get("reasoning"),
          }
        )

      elif t == "vote":
        a = p.get("attempt", attempt_by_quest.get(q, 0))
        # back-fill the just-added proposal for this (quest, attempt)
        for pr in reversed(proposals):
          if pr["game"] == idx and pr["quest"] == q and pr["attempt"] == a:
            pr["approved"] = p.get("approved")
            pr["num_approvals"] = p.get("approvals")
            break
        for seat, v in p.get("votes", {}).items():
          votes.append(
            {
              "game": idx,
              "quest": q,
              "attempt": a,
              "seat": seat,
              "approved": v,
              "reasoning": priv.get(seat, {}).get("reasoning"),
            }
          )
        # guesses: each voter's belief about every eligible target
        for guesser, ov in priv.items():
          belief = ov.get("belief")
          if belief is None:
            continue
          prior = set(known.get(guesser, {}).get("known_evil", [])) | set(known.get(guesser, {}).get("merlin_candidates", []))
          for target, t_role in truth.items():
            if target == guesser:
              continue  # a player does not guess its own role
            guesses.append(
              _guess_row(
                idx,
                q,
                a,
                guesser,
                target,
                belief.get(target),
                t_role,
                excluded=(target in prior),
                reasoning=ov.get("reasoning"),
              )
            )
        attempt_by_quest[q] = a + 1

      elif t == "quest":
        quests.append(
          {
            "game": idx,
            "quest": q,
            "succeeded": p.get("success"),
            "num_fails": p.get("num_fails"),
          }
        )
        for seat, ov in priv.items():
          sabotage.append(
            {
              "game": idx,
              "quest": q,
              "seat": seat,
              "sabotaged": ov.get("sabotaged"),  # None for pre-fix runs
              "reasoning": ov.get("reasoning"),
            }
          )

      elif t == "assassinate":
        assassin = p.get("assassin")
        assassination.append(
          {
            "game": idx,
            "assassin_seat": assassin,
            "target_seat": p.get("target"),
            "merlin_seat": merlin_seat,
            "assassinated": p.get("hit_merlin"),
            "reasoning": priv.get(assassin, {}).get("reasoning"),
          }
        )

      elif t == "debrief":
        seat = p.get("seat")
        debriefs.append(
          {
            "game": idx,
            "seat": seat,
            "reflection": priv.get(seat, {}).get("reasoning"),
          }
        )

  memory = _memory_rows(run_dir)
  return {
    "games": games,
    "seats": seats,
    "proposals": proposals,
    "votes": votes,
    "quests": quests,
    "sabotage": sabotage,
    "guesses": guesses,
    "speeches": speeches,
    "assassination": assassination,
    "debriefs": debriefs,
    "memory": memory,
  }


def _guess_row(game, quest, attempt, guesser, target, guessed, true_role, excluded, reasoning):
  role_matched = None if excluded else (guessed == true_role)
  side_matched = None if excluded else (guessed is not None and side_of(guessed) == side_of(true_role))
  return {
    "game": game,
    "quest": quest,
    "attempt": attempt,
    "guesser_seat": guesser,
    "target_seat": target,
    "guessed_role": guessed,
    "true_role": true_role,
    "role_matched": role_matched,
    "side_matched": side_matched,
    "excluded": excluded,
    "reasoning": reasoning,
  }


def _won(role, winner_side):
  if winner_side is None:
    return None
  return side_of(role) == winner_side


def _memory_rows(run_dir: str) -> list:
  """Unfold each seat's curated_history into one row per game."""
  mem_dir = pathlib.Path(run_dir) / "memory"
  rows = []
  if not mem_dir.exists():
    return rows
  for f in sorted(mem_dir.glob("*.json")):
    d = json.loads(f.read_text())
    seat = f.stem
    for i, note in enumerate(d.get("curated_history", []), start=1):
      rows.append({"game": i, "seat": seat, "notes": note})
  return rows


# --- metadata + data dictionary -----------------------------------------


def build_metadata(run_dir: str, tables: dict) -> dict:
  ledger = read_ledger(run_dir)
  roster = {}
  for row in ledger:
    for name, info in row.get("seats", {}).items():
      roster.setdefault(
        name,
        {
          "seat": name,
          "model": info.get("model"),
          "effort": info.get("reasoning_effort"),
          "learn": info.get("learn"),
          "memory_source": info.get("memory"),
        },
      )
  return {
    "run_id": pathlib.Path(run_dir).name,
    "num_games": len(tables["games"]),
    "num_players_per_game": tables["games"][0]["num_players"] if tables["games"] else None,
    "seed_base": _seed_base(run_dir),
    "roster": list(roster.values()),
    "generated_at": datetime.now(timezone.utc).isoformat(),
  }


def _seed_base(run_dir: str):
  """seed_base from the run manifest (results.jsonl line 1). Per-game seed = seed_base + game."""
  path = pathlib.Path(run_dir) / "results.jsonl"
  if not path.exists():
    return None
  with path.open(encoding="utf-8") as f:
    first = f.readline()
  try:
    return json.loads(first).get("seed_base")
  except (ValueError, AttributeError):
    return None


TABLE_FORMATS = {
  "games": "csv",
  "seats": "csv",
  "quests": "csv",
  "proposals": "jsonl",
  "votes": "jsonl",
  "sabotage": "jsonl",
  "guesses": "jsonl",
  "speeches": "jsonl",
  "assassination": "jsonl",
  "debriefs": "jsonl",
  "memory": "jsonl",
}

CSV_FIELDS = {
  "games": ["game", "num_players", "winner_side", "reason", "num_successes", "num_fails"],
  "seats": ["game", "seat", "role", "side", "won", "known_evil_seats", "merlin_candidate_seats"],
  "quests": ["game", "quest", "succeeded", "num_fails"],
}


CITATION = pathlib.Path(__file__).resolve().parent.parent / "CITATION.cff"


def _read_citation() -> dict:
  """Author, repo, title, and DOIs from CITATION.cff (the canonical citation source)."""
  import yaml

  try:
    cff = yaml.safe_load(CITATION.read_text())
  except (OSError, yaml.YAMLError):
    return {}
  authors = []
  for a in cff.get("authors", []):
    fam, giv = a.get("family-names", ""), a.get("given-names", "")
    authors.append(f"{fam}, {giv}".strip(", ") if (fam or giv) else a.get("name", ""))
  dataset_ref = next((r for r in cff.get("references", []) if r.get("type") == "dataset"), {})
  return {
    "authors": authors,
    "repo_url": cff.get("repository-code", ""),
    "title": cff.get("title", "camelot-bench"),
    "code_doi": _doi_of(cff),
    "dataset_doi": _doi_of(dataset_ref),
  }


def _doi_of(entry: dict) -> str:
  """DOI from a CFF entry: the `doi` field, or a `type: doi` identifier."""
  if entry.get("doi"):
    return entry["doi"]
  for ident in entry.get("identifiers", []):
    if ident.get("type") == "doi" and ident.get("value"):
      return ident["value"]
  return ""


TEMPLATE = pathlib.Path(__file__).resolve().parent.parent / "docs" / "templates" / "README.md"
LICENSE = pathlib.Path(__file__).resolve().parent.parent / "docs" / "templates" / "LICENSE"


def write_dataset(run_dir: str, out_dir: str) -> None:
  out = pathlib.Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  tables = extract(run_dir)
  meta = build_metadata(run_dir, tables)

  for name, rows in tables.items():
    fmt = TABLE_FORMATS[name]
    if fmt == "csv":
      write_csv(out / f"{name}.csv", rows, CSV_FIELDS[name])
    else:
      write_jsonl(out / f"{name}.jsonl", rows)

  (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
  (out / "README.md").write_text(_render_readme(tables, meta))
  (out / "LICENSE").write_text(LICENSE.read_text())
  print(f"wrote dataset to {out}/  ({sum(len(r) for r in tables.values())} rows across {len(tables)} tables)")


def _render_readme(tables: dict, meta: dict) -> str:
  cite = _read_citation()
  file_list = "\n".join(
    f"- `{name}.{'csv' if TABLE_FORMATS[name] == 'csv' else 'jsonl'}` ({len(rows)} rows)" for name, rows in tables.items()
  )
  authors = " and ".join(cite.get("authors") or ["Luo, Zih-Yuan"])
  title = cite.get("title", "camelot-bench") + " dataset"
  dataset_doi = cite.get("dataset_doi") or "<dataset DOI>"
  code_doi = cite.get("code_doi") or "<code DOI>"
  bibtex = (
    "@misc{camelot_bench_dataset,\n"
    f"  author       = {{{authors}}},\n"
    f"  title        = {{{title}}},\n"
    f"  year         = {{{datetime.now(timezone.utc).year}}},\n"
    "  publisher    = {Zenodo},\n"
    f"  doi          = {{{dataset_doi}}},\n"
    "}"
  )
  values = {
    "repo_url": cite.get("repo_url", ""),
    "run_id": meta["run_id"],
    "num_games": meta["num_games"],
    "num_players": meta.get("num_players_per_game"),
    "code_doi": code_doi,
    "dataset_doi": dataset_doi,
    "bibtex": bibtex,
    "file_list": file_list,
  }
  text = TEMPLATE.read_text()
  for key, val in values.items():
    text = text.replace("{{" + key + "}}", str(val))
  return text


def main():
  ap = argparse.ArgumentParser(description="Export a run into a release dataset.")
  ap.add_argument("--run", default=None, help="run dir (default: latest under data/runs)")
  ap.add_argument("--out", default="dataset", help="output dir (default: dataset/)")
  args = ap.parse_args()

  run = args.run
  if run is None:
    runs = sorted(pathlib.Path("data/runs").iterdir())
    if not runs:
      raise SystemExit("no runs found under data/runs")
    run = str(runs[-1])
  write_dataset(run, args.out)


if __name__ == "__main__":
  main()
