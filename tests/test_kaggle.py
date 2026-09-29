import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
import kaggle_benchmarks as kbench

from cambench.agents.decision_graph import DecisionAborted
from cambench.agents.llm_player import LLMPlayer
from cambench.agents.schemas import DebriefOutput
from cambench.engine import AvalonGame, recommended_config
from cambench.engine.labels import make_names, seat_label, using_names
from cambench.integrations.kaggle import CallBudget, KaggleClient, SeatSpec, evaluate_episode
from cambench.integrations.kaggle.checkpoint import Checkpoints, exclusive_run
from conftest import policy


def test_adapter_isolated_context_and_budget(proxy_factory):
  llm, calls = proxy_factory()
  original_retries = llm.client.max_retries
  client = KaggleClient(llm, CallBudget(2))
  with kbench.chats.new("outer") as root:
    client.call("secret seat A", "first private observation", DebriefOutput)
    client.call("secret seat B", "second private observation", DebriefOutput)
    with pytest.raises(DecisionAborted, match="budget"):
      client.call("no", "no", DebriefOutput)
  assert len(calls) == 2
  assert "secret seat A" not in json.dumps(calls[1])
  assert "first private observation" not in json.dumps(calls[1])
  assert "secret seat B" in json.dumps(calls[1])
  assert len(root.history) == 2  # audit subchats retained
  assert llm.client.max_retries == original_retries
  assert client.llm.client.max_retries == 0


def test_direct_paid_endpoint_rejected(proxy_factory):
  llm, calls = proxy_factory()
  llm.client = llm.client.with_options(base_url="https://api.openai.com/v1")
  with pytest.raises(ValueError, match="Refusing inference"):
    KaggleClient(llm, CallBudget())
  assert not calls


def test_quota_failure_aborts_without_random_fallback(proxy_factory, tmp_path):
  llm, calls = proxy_factory(handler=lambda _: httpx.Response(429, json={"error": {"message": "quota exhausted"}}))
  with pytest.raises(DecisionAborted, match="inference failed"):
    evaluate_episode(llm, run_dir=tmp_path / "episode")
  assert len(calls) == 1
  assert not list((tmp_path / "episode/checkpoints").glob("game_*.json"))
  assert not (tmp_path / "episode/metrics.json").exists()
  assert json.loads((tmp_path / "episode/usage.json").read_text())["calls"] == 1


@pytest.mark.parametrize("count", range(5, 11))
def test_complete_games_match_engine_and_keep_secrets(proxy_factory, tmp_path, count):
  llm, calls = proxy_factory()
  seats = [SeatSpec()] * count
  result = evaluate_episode(llm, seats=seats, seed=19, run_dir=tmp_path / "episode")
  rows = [json.loads(line) for line in (result.run_dir / "records/game_0001.jsonl").read_text().splitlines()]
  assert rows[0]["reason"] in ("merlin_assassinated", "quests_won_merlin_safe")
  assert result.win_rate in (0.0, 1.0)
  names = make_names(count, 19)
  # Every SDK request is a freshly rendered seat observation, never another
  # seat's reasoning, current hidden card, or private belief output.
  for request in calls:
    user = next(m["content"] for m in request["messages"] if m["role"] == "user")
    me = user.split("Your name is ")[1].split()[0].rstrip(".")
    assert all(f"private:{other}" not in user for other in names if other != me)
    if "Vote to approve" in user:
      # The latest proposal must not yet have a published vote tally.
      current = user.rsplit("PROPOSE", 1)[-1]
      assert "VOTE" not in current
  class LocalClient:
    def call(self, system, user, schema):
      return schema.model_validate(policy(user, schema.model_fields))
  with using_names(names):
    players = [LLMPlayer(LocalClient(), seed=i) for i in range(count)]
    game = AvalonGame(players, config=recommended_config(count), seed=20, game_id="game_0001")
    game.play()
    game.run_debrief()
    assert game.record.to_jsonl() == (result.run_dir / "records/game_0001.jsonl").read_text()


def test_learning_freeze_and_resume(proxy_factory, tmp_path, monkeypatch):
  llm, calls = proxy_factory()
  seats = [SeatSpec(learn=True), SeatSpec(memory={"curated": "frozen-notes"})] + [SeatSpec()] * 3
  uninterrupted = evaluate_episode(llm, seats=seats, games=3, seed=5, max_calls=2000, run_dir=tmp_path / "whole")
  original = Checkpoints.commit
  def interrupt(self, game):
    original(self, game)
    if len(self.games) == 1:
      raise RuntimeError("simulated interruption after atomic commit")
  monkeypatch.setattr(Checkpoints, "commit", interrupt)
  with pytest.raises(RuntimeError, match="simulated interruption"):
    evaluate_episode(llm, seats=seats, games=3, seed=5, max_calls=2000, run_dir=tmp_path / "resumed")
  monkeypatch.setattr(Checkpoints, "commit", original)
  resumed = evaluate_episode(llm, seats=seats, games=3, seed=5, max_calls=2000, run_dir=tmp_path / "resumed", resume=True)
  assert resumed.report == uninterrupted.report
  assert (resumed.run_dir / "results.jsonl").read_text() == (uninterrupted.run_dir / "results.jsonl").read_text()
  for file in (uninterrupted.run_dir / "records").glob("*"):
    assert file.read_text() == (resumed.run_dir / "records" / file.name).read_text()
  names = make_names(5, 5)
  learned = json.loads((resumed.run_dir / "memory" / f"{names[0]}.json").read_text())
  frozen = json.loads((resumed.run_dir / "memory" / f"{names[1]}.json").read_text())
  assert len(learned["results"]) == len(learned["window"]) == 3
  assert learned["curated"] == f"learned:{names[0]}"
  assert frozen["curated"] == "frozen-notes" and not frozen["results"] and not frozen["window"]
  # Resuming a completed episode consumes no calls.
  before = len(calls)
  evaluate_episode(llm, seats=seats, games=3, seed=5, max_calls=2000, run_dir=resumed.run_dir, resume=True)
  assert len(calls) == before
  with pytest.raises(ValueError, match="differs"):
    evaluate_episode(llm, seats=seats, games=3, seed=6, run_dir=resumed.run_dir, resume=True)


def test_budget_durable_across_incomplete_game(proxy_factory, tmp_path):
  llm, calls = proxy_factory()
  for resume in (False, True):
    with pytest.raises(DecisionAborted, match="budget"):
      evaluate_episode(llm, run_dir=tmp_path / "episode", max_calls=1, resume=resume)
  assert len(calls) == 1
  result = evaluate_episode(llm, run_dir=tmp_path / "episode", max_calls=600, resume=True)
  assert result.report["calls_attempted"] == len(calls)


def test_candidate_attribution_and_mixed_models(proxy_factory, tmp_path, monkeypatch):
  candidate, _ = proxy_factory("candidate")
  opponent, other_calls = proxy_factory("opponent")
  monkeypatch.setattr("kaggle_benchmarks.kaggle.models.load_model", lambda name: opponent)
  @kbench.task(name="camelot-test-attribution")
  def task(llm) -> float:
    return evaluate_episode(llm, seats=[SeatSpec()] + [SeatSpec(model="opponent")] * 4,
                            run_dir=tmp_path / "episode").win_rate
  run = task.run(llm=candidate)
  assert run.evaluated_subject is candidate
  assert isinstance(run.result, float)
  assert other_calls


def test_parallel_episodes_restore_names(proxy_factory, tmp_path):
  def play(seed):
    llm, _ = proxy_factory()
    # SDK contexts must be entered in each worker (as evaluate() does).
    with kbench.chats.new(f"worker-{seed}"):
      result = evaluate_episode(llm, seed=seed, run_dir=tmp_path / str(seed))
    assert seat_label(0) == "A"
    return json.loads((result.run_dir / "manifest.json").read_text())["names"]
  with ThreadPoolExecutor(2) as pool:
    names = list(pool.map(play, [31, 32]))
  assert names == [make_names(5, 31), make_names(5, 32)]
  assert seat_label(0) == "A"


def test_same_directory_cannot_run_concurrently(tmp_path):
  with exclusive_run(tmp_path):
    with pytest.raises(RuntimeError, match="Another evaluation"):
      with exclusive_run(tmp_path):
        pass
