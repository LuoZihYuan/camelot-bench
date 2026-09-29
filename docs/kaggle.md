# Kaggle Benchmarks

The optional integration runs the existing Avalon engine through **Kaggle's
Model Proxy**. It does not call the direct OpenAI, Anthropic, or Google clients,
and it refuses model endpoints outside HTTPS Kaggle domains. Personal API keys
are not used as fallback. Kaggle controls model access and free inference quota.

## Install and authenticate

Python **3.13+** is required, just like the existing project. Verify the hosted
notebook's interpreter before deployment; SDK support for Python 3.11 does not
establish that a particular Kaggle runtime can install this project.

```sh
uv sync --extra kaggle --group test
uv run pytest -q
uvx kaggle benchmarks init -y
uvx kaggle benchmarks tasks models
```

`init` writes a Model Proxy token to `.env`; never commit it. Use Kaggle account
settings to resolve phone/identity verification or expired credentials. Model
slugs available with a local token can differ from hosted model availability.
Newer CLI versions expose `kaggle benchmarks quota`; if your installed CLI does
not have that command, inspect the account's AI quota in Kaggle's UI. Example
dollar allowances in CLI documentation are not your actual entitlement.

## First local pilot

Run from the repository root after authentication:

```python
import kaggle_benchmarks as kbench
from cambench.integrations.kaggle import evaluate_episode

@kbench.task(name="my-camelot-pilot")
def pilot(llm) -> float:
    result = evaluate_episode(llm, max_calls=120, run_dir="data/kaggle/pilot")
    print(result.run_dir)  # metrics.json, records/, memory/, and checkpoints/
    return result.win_rate

run = pilot.run(llm=kbench.llm)
```

The default roster is **one evaluated candidate and four seeded RandomBots**.
This is an inexpensive integration check, not a social-deduction leaderboard:
RandomBots do not report beliefs, so concealment metrics may have no samples.
The baseline plays the complete game; discussion is not shortened.

The adapter uses fresh SDK chats and the engine's existing filtered observation
for every decision. These chats stay in the audit trace, but are never fed to
other players. Engine events determine when votes, quest results, and terminal
roles become public. Do not replace decisions with shared ChatRoom replies:
doing so can expose private reasoning and earlier sealed votes.

## Mixed rosters, learning, and warm starts

```python
import json
from cambench.integrations.kaggle import SeatSpec, evaluate_episode

seats = [SeatSpec(learn=True)] + [
    SeatSpec(model=opponent_slug, learn=False) for opponent_slug in fixed_opponents
]
result = evaluate_episode(
    kbench.llm, seats=seats, candidate_seat=0, games=3, seed=42,
    max_calls=1800, run_dir="data/kaggle/learning-42",
)
```

Set `fixed_opponents` to 4–9 Kaggle model slugs that are available to your account.
One shared call budget covers every seat, debrief, and memory update. Optional
special roles use the existing `recommended_config`; pass a `GameConfig` for a
different valid role set.

`SeatSpec(model=None)` always uses the evaluated `llm`. Multiple such seats allow
self-play, but the returned win rate still scores only `candidate_seat`. Do not
treat average all-seat self-play wins as model strength. `learn=False` preserves
the existing atomic freeze semantics: no notes, window, or result-history updates;
debriefs are still recorded. Warm-start a seat with
`SeatSpec(memory=json.loads(saved_memory_file.read_text()), learn=True)`.
Memory is copied per seat, so sharing an input dictionary does not share state.

The trajectory preserves the existing learning order: play, reveal, private
debrief, own result, rolling win rate, revised notes, recent-game window. Games
in one trajectory are sequential. Parallelize independent trajectories, using
separate run directories and SDK evaluation contexts. Seat labels are scoped to
the current context, including under threaded SDK evaluations.

## Budgets and errors

`max_calls` (default 600) caps SDK prompt attempts across the whole trajectory.
`max_tokens` (default 2048) is forwarded as the Model Proxy's output-token limit.
Requests have a 120-second client timeout and automatic OpenAI-client retries
are disabled. These are **request/output limits, not a guaranteed dollar cap**;
provider reasoning and billing semantics vary. Verify your remaining free quota
before starting and size the pilot conservatively.

Quota, authentication, transport, and SDK parsing errors abort the episode.
They do not become random moves or a completed score. Existing semantic-action
validation still retries/falls back as in the original agent. A call reservation
is persisted before dispatch, including failed attempts. There is no paid API
fallback. Keep SDK `max_attempts=1` for these stateful tasks: automatic retry of a
whole trajectory can repeat inference and create new output directories.

## Checkpoints and resume

```python
result = evaluate_episode(
    kbench.llm, seats=seats, candidate_seat=0, games=3, seed=42,
    max_calls=2400, run_dir="data/kaggle/learning-42", resume=True,
)
```

Resume must use the same source, SDK version, roster, memory inputs, seeds, game
count, and generation configuration. Only the call limit may be increased.
Each completed game atomically commits its record, result, memories, and player
fallback RNG states. An unfinished game is replayed from the last completed
boundary; its previous inference still counts against the budget, and a live
model need not return identical responses on replay. Resuming an already complete
episode makes no additional model calls.

`checkpoints/` is authoritative; the standard records, ledger, and memory files
are reconstructed from it. The run directory is locked against concurrent use
on Linux/macOS. Kaggle does not promise to retain local files across separate
hosted executions: save/download the **whole run directory**, including usage and
checkpoints, and restore it explicitly. SDK run caching alone does not restore
agent memory. Prior-game SDK traces are not regenerated on resume; the saved
engine records remain the complete trajectory evidence.

## Tasks and leaderboard scores

`benchmarks/kaggle/avalon.py` provides:

- `camelot-random-baseline`: a complete five-player game against random opponents.
- `camelot-fixed-roster`: an explicit fixed model roster and configurable learning
  trajectory, scored only for the evaluated candidate.

Both return an annotated float win rate. The fixed-roster task requires
`opponent_models`; configure and save those inputs in the notebook before asking
Kaggle to evaluate more models. Do not substitute the candidate for every opponent
when comparing candidate strength. Balance candidate seats and role exposure over
many seeds; one game is only a smoke test. Inspect `role_counts` rather than
assuming a small random sample is balanced. Freeze the opponent roster and label
learning/frozen variants separately. Fixed-roster results are a different protocol
from an all-seats-learning mixed-model league.

`metrics.json` preserves the existing win rates, role splits, belief accuracy,
concealment, assassination, role counts, and trajectories for every seat, plus
the selected candidate's metrics. Empty dictionaries mean no eligible samples,
not zero performance. Existing asymmetric confidence intervals are preserved in
artifacts rather than forced into Kaggle's symmetric CI field. Learning-trajectory
comparisons should use independent trajectory replicates for uncertainty estimates.

Group the published Tasks into a Camelot Benchmark in Kaggle's UI. A dictionary
of seat scores does not automatically create a leaderboard row for each model.
Use the original `llm` task argument for attribution; the adapter wraps it internally.

## Hosted deployment

After pushing the integration commit, generate an upload script pinned to its full
SHA (not a moving branch):

```sh
uv run python scripts/build_kaggle_task.py --source-ref FULL_40_CHARACTER_COMMIT_SHA
uvx kaggle benchmarks tasks push camelot-random-baseline -f data/kaggle/avalon_task.py --wait
uvx kaggle benchmarks tasks run camelot-random-baseline -m AVAILABLE_MODEL_SLUG --wait
```

The generated file installs `camelot-bench[kaggle]` from that commit before
defining Tasks and running the random-baseline pilot with `kbench.llm`. To deploy
the fixed-roster task, pass `--opponent-models SLUG1 SLUG2 SLUG3 SLUG4`
to the builder and use `camelot-fixed-roster` in the push/run commands. The
generated script saves that roster in the actual task invocation. It fails clearly on an older Python runtime. Local edits must be
committed and pushed before generating a deployable script. Notebook setup and
package downloads require the corresponding Kaggle networking capabilities.
Deployment should initially stay private. Public publication is a separate step.

The source is tested with **kaggle-benchmarks 0.6.1**, mocked HTTP responses through
the real SDK, and deterministic engine replay. A hosted pilot additionally needs
working account authentication, model access, quota, and a compatible interpreter.
Offline tests do not establish hosted acceptance or runtime capacity.

## Continuous integration

`docs/kaggle-ci.yml` is a ready-to-use GitHub Actions template for the offline
suite. Copy it to `.github/workflows/tests.yml` using a credential with workflow
permission to enable it. The integration does not require paid inference for tests.
