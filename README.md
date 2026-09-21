# **camelot-bench**

> *A multi-agent LLM benchmark using the hidden-role game Avalon to measure reasoning under deception via self-play, cross-game memory, and belief-accuracy scoring.*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22858217.svg)](https://doi.org/10.5281/zenodo.22858217)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

The two strongest models in our 100-game run, **gemini-3.7-flash** and **gemini-3.6-flash**, post nearly the same win rate (61% vs 60%), yet they win in completely different ways. gemini-3.7 is an **assassin specialist**: it reads roles better than any other model and turns that into a near-flawless kill record, finding Merlin in every assassination it played. gemini-3.6 is a **generalist**, winning steadily across every role. Win rate alone calls the two equivalent, but the per-role and assassination metrics show they play nothing alike.

![Win rate by role](docs/img/win_rate_by_role.png)
![Assassination](docs/img/assassination.png)


## **Table of Contents**

- [Overview](#overview)
- [Usage](#usage)
- [Features](#features)
- [Results](#results)
- [Dataset](#dataset)
- [Citation](#citation)


## **Overview**

camelot-bench is a multi-agent LLM benchmark that runs full games of *The Resistance: Avalon* (5-10 players, all special roles) between language-model agents and scores not just who wins, but how well each model performs the reasoning the game demands: deducing hidden roles, concealing its own, and finding Merlin in the final assassination. Agents accumulate memory across games, so the benchmark also captures whether they improve over a run. The engine is pure-Python and deterministic; agents are built with [Pydantic AI](https://ai.pydantic.dev/) and games are orchestrated with [LangGraph](https://langchain-ai.github.io/langgraph/).


## **Usage**

### **Installation**

```bash
uv sync
```

The analysis notebooks additionally need the dev group:

```bash
uv sync --group dev
```

API keys live in a `.env` file (never in the roster).

### **Configuring a Roster**

For a quick homogeneous run, skip the config entirely and use CLI flags:

```bash
uv run python scripts/train.py --model google:gemini-3.7-flash --players 5 --games 3
```

For per-seat control (mixed models, warm-started or frozen memory), copy `roster.example.yaml` to `roster.yaml` and set one entry per seat:

```yaml
games: 100
seats:
  - {model: google:gemini-3.7-flash, reasoning_effort: medium}
  - {model: google:gemini-3.6-flash, reasoning_effort: medium}
  - {model: google:gemini-3.5-flash-lite, reasoning_effort: medium}
  - {model: openai:gpt-5.6-luna, reasoning_effort: medium}
  - {model: openai:gpt-5.6-terra, reasoning_effort: medium}
```

> [!NOTE]
> Per-seat fields:
> - `model`: provider-prefixed model string (`openai:` / `anthropic:` / `google:`).
> - `reasoning_effort`: `low`, `medium`, or `high` (omit to leave off).
> - `memory`: path to a saved memory file to warm-start the seat from (omit to start blank).
> - `learn`: `true` (default) updates the seat's memory each game; `false` freezes it for controlled comparisons.
>
> See `roster.example.yaml` for all per-seat options and common roster shapes.

### **Running a Benchmark**

```bash
uv run python scripts/train.py --config roster.yaml
```

> [!TIP]
> Interrupted runs resume exactly where they stopped. Run `uv run python scripts/train.py --resume` and the pipeline continues from the last completed game, with every agent's memory intact.

### **Analyzing Results**

Open `notebooks/seats.ipynb` (per-model metrics) and `notebooks/games.ipynb` (game-level dynamics), point `RUN_DIR` at your run, and run all cells.


## **Features**

### **Configurable Agents**

- **Multi-provider**: mix models from OpenAI, Google, and Anthropic in a single game, each seat independently configured.

- **Per-seat memory and learning**: each agent keeps its own notes, revises them after every game, and carries a sliding window of recent games. Learning can be frozen per seat for controlled comparisons.

- **Per-run randomized seat names**: seats get fresh random identifiers each run, so within-run opponent modeling is possible while cross-run tells can't leak through memory.

### **Gameplay & Metrics**

- **Structured discussion**: before each vote, players speak in a chain: each speaker names who talks next, and every player is capped at 3 turns per proposal.

- **Reasoning metrics with ground truth**: beyond win rate, the benchmark scores guess accuracy (reading others' roles), hidden rate (concealing your own), and assassination outcomes, all against the true role assignment, with 95% confidence intervals.

### **Operations**

- **Resumable runs**: a crash or interruption never loses progress; runs continue from the last completed game.


## **Results**

The findings below come from a single 100-game run of the five-model roster above, at five players per game. Sample sizes per model-per-role are modest, so several comparisons carry wide confidence intervals; the charts show 95% intervals and the text notes where a result is robust versus suggestive.

### **Outcomes**
 
Evil won 58% of games. Notably, **41% of all games ended with good completing the quests but losing to the assassin**, almost as often as good actually won (42%). The assassination phase decides the game nearly as often as the quests do.

![Outcomes](docs/img/outcomes.png)

### **Win Rate**

The two mid-tier Gemini Flash models lead (61%, 60%); the rest cluster near 40%. But the by-role breakdown shows the two leaders win differently: gemini-3.7 leans heavily on the assassin role, while gemini-3.6 wins evenly across roles.

![Win rate](docs/img/win_rate.png)
![Win rate by role](docs/img/win_rate_by_role.png)

### **Guess Accuracy**

How accurately each model guesses others' roles (exact role) and sides (alignment). The Gemini Flash line improves generation over generation on exact accuracy, rising 30% to 47% to 54% across the 3.5-lite, 3.6, and 3.7 generations.

![Guess accuracy](docs/img/guess_accuracy.png)

### **Hidden Rate**

How often others fail to identify a seat. Every model conceals itself far better when evil than when good. This is partly structural (most players are good, so good is easy to assume), but gemini-3.7 is exceptional, hiding its evil alignment better than any other model.

![Hidden rate](docs/img/hidden_rate_by_side.png)

### **Assassination**

When playing the assassin, gemini-3.7 found Merlin in every game. Even at the lower bound of its confidence interval it stays ahead of every other model, so it is decisively the strongest assassin, which is what powers its evil win rate.

![Assassination](docs/img/assassination.png)

### **Learned Notes**

Agents write and revise their own strategy notes after every game. Across the same 100 games, the depth of what they learned tracked their strength: gemini-3.7-flash built an intricate, evolving theory of assassination tells, while gemini-3.5-flash-lite kept the same short, high-level role list from start to finish. Compare each seat's notes at game 51 and game 100 below.

- **google:gemini-3.7-flash@medium#E63M**

  > The only seat where you can watch a genuine discovery happen. At game 51 its Merlin rule was behavioral (don't be the lone early dissenter); by game 100 it had found a second tell in the *proposal record*, that a Merlin whose rosters are 100% Evil-free all match hands the Assassin a clean kill from the game log alone. The striking part is the countermeasure it wrote itself: deliberately propose imperfect rosters as Merlin, choosing to play worse on the mission axis to buy concealment on the identity axis.
  >
  > <details>
  > <summary>Game 51</summary>
  > 
  > # Avalon Strategy Notes
  > 
  > ## Merlin Camouflage & Survival (Critical)
  > - **Zero Lone Dissent**: NEVER be the sole Reject against unexposed Morgana/Evil on Q1/Q2 before failure data exists. Early lone dissent against unexposed Evil provides 100% mathematical certainty to the Assassin via hard-exclusion of Evil-approvers.
  > - **Procedural Soft Hedging**: Always hedge ("useful evidence, not permanent clearance / absolute certainty") even when backing undefeated winning cores at match point.
  > - **Standard Expansion Flow**: Propose routine single-variable expansions carrying forward zero-fail cores using standard procedural phrasing; blend seamlessly into table cushion resets.
  > 
  > ## Percival & Loyal Servant Guidelines
  > - **Silent Candidate Tracking**: NEVER publicly name candidates as {Merlin, Morgana} or debate candidate identities aloud; doing so exposes Percival and isolates Merlin.
  > - **Active Decoy Maintenance**: Percival must actively steer proposals, propose clean 3-seat expansions, and maintain confident town leadership to draw Assassin fire away from Merlin.
  > - **Deduction & Trap Avoidance**:
  >   - *2-Candidate Trap*: Never approve a team containing BOTH seen candidates {Merlin, Morgana}.
  >   - *Multi-Fail Overlap*: If Player A is on multiple failed quests, Player A is confirmed Evil. Do NOT assume untested outside players are Good.
  >   - *Lone Dissent Signal*: Sole Reject against unexposed Evil on Q1/Q2 is almost certainly Merlin acting on secret knowledge.
  >   - *Diagnostic & Hammer Traps*: Do NOT approve risky "diagnostic" teams or rush untested seats at match point just to avoid P5 hammer—Evil exploits this to manufacture 3–2 majorities.
  > 
  > ## Evil & Assassin Tactics
  > - **Process-of-Elimination Assassination**:
  >   1. *Hard Exclusions (Never Shoot)*:
  >      - Anyone who approved unexposed Morgana/Assassin on Q1/Q2 before failure data existed (confirms Percival/LS).
  >      - Anyone who proposed Morgana OR Assassin on any round.
  >      - Anyone who publicly debated or tracked Merlin/Morgana candidates (identifies Percival).
  >   2. *Merlin Identification*:
  >      - *Sole Early Dissenter*: The player who cast lone dissent against unexposed Evil on Q1/Q2 before failure data existed.
  >      - The sole un-excluded Good player remaining after eliminating evil-proposers, evil-approvers, and active Percival decoys.
  > - **Exploiting Town Biases & Cushion**: Echo LS diagnostic framing or fear of P5 hammer to lock in 3–2 majorities (2 Evil + 1 LS) and pass Evil onto match-point quests.
  > - **Camouflage & Blending**: Blend seamlessly with standard town procedure and soft hedging to keep Percival split and let Good advance to 3 wins if Merlin is already identified.
  > 
  > </details>
  >
  > <details>
  > <summary>Game 100</summary>
  > 
  > # Avalon Strategy Notes
  > 
  > ## Merlin Camouflage & Anti-Assassination
  > - **Zero Early Prescient Dissent (Fatal Rule)**: NEVER lone-dissent on Q1/Q2 against unexposed Evil before failure data exists. Mirror consensus with procedural soft hedging.
  > - **Clairvoyant Proposal Trap (Fatal POE Signature)**: If Merlin's proposals are 100% zero-Evil across the game while Percival proposed an Evil candidate (e.g. Percival tested Morgana on Q2), Assassin gets an airtight POE kill on Merlin. When leading Q1 or Q2, do not seek clairvoyant perfection; proposing standard rotational pairs or mirroring town branches avoids the 100% clean proposal POE signature.
  > - **Tone & Hedging**: Blend speech length and turn-passing. NEVER act as the primary deductive organizer or claim certainty ("mathematically verified"). Consistently use procedural soft hedging ("treating outcomes strictly as diagnostic evidence rather than permanent clearance"). Let vocal Loyal Servants / Percival draw Assassin fire.
  > 
  > ## Percival & Loyal Servant Guidelines
  > - **Aggressive Decoy Shielding**: Vocalize expansion-isolation proofs, mathematical pool partitions, and voting dissent patterns with decisive authority ("mathematically verified", "100% clean") to draw Assassin kill shots away from Merlin.
  > - **Candidate Elimination & P4 Lock-in**:
  >   - If candidate Morgana fails on Q1/Q2/Q3, the other candidate is 100% Merlin. Percival MUST aggressively back Merlin and treat P4 as mandatory lock-in to prevent paranoid rejections into an Evil P5 hammer trap.
  >   - An "untested" Merlin candidate is mathematically superior to taking an exposed failure-pool seat or facing an Evil hammer.
  > - **Disjoint Partition & Dissent Isolation Proof**:
  >   - *2 Fails on 2p (Q1)*: Confines both Evil to that pair; off-mission trio is 100% hard-cleared Good for Q2, Q3 (any pair), and Q4.
  >   - *2 Fails on 3p*: Confines both Evil to that trio; off-mission pair is 100% verified Good. Lock them on Q3 (size 2).
  >   - *Q4 3rd Seat Selection*: The player inside the 2-fail trio who voted NO against that team is the verified Good 3rd seat (isolating the two approving Evil seats).
  > 
  > ## Evil & Assassin Strategy
  > - **Process-of-Elimination (POE) Assassination Rules**:
  >   1. *Merlin Clairvoyant Proposal Signature (100% Merlin)*: If Player A's proposals NEVER include either Evil player across the entire match (100% all-Good rosters), while Player B proposes candidate branches or pairs with Morgana/Merlin candidates (e.g. {Percival, Morgana, Candidate}), Player A is Merlin and Player B is Percival.
  >   2. *Early Prescient Dissent Signature (100% Merlin)*: Any player who lone-dissented on Q1/Q2 against unexposed Evil before failure data existed is 100% Merlin.
  >   3. *Vocal Decoy Trap*: NEVER assassinate the vocal late-game deductive organizer who leads town math, calls out vote patterns, or claims certainty—they are almost always Percival or Loyal Servants acting as decoys. Check proposal history and soft hedging to find the quiet Merlin.
  > - **Morgana Camouflage & Pacing**: Adopt standard town phrasing with procedural soft hedging. Mirror Percival's pacing arguments to secure winning approvals or force P5 hammers. Avoid double-failing on size-2 missions if single fail suffices.
  > 
  > </details>

- **openai:gpt-5.6-terra@medium#H43T**

  > The most sophisticated game theory of the five, and an anomaly: conditional sabotage, hard-clear conditions, "roster authorship compounds", yet one of the lowest win rates. Its notes read as an essay rather than a protocol, telegraphic and hedged with "usually/unless/if", nearly unparseable under pressure next to the bright-line rules of the stronger seats. Both snapshots are also cut off mid-word, hinting it may be writing past its own budget and losing the freshest lesson off the tail.
  > 
  > <details>
  > <summary>Game 51</summary>
  > 
  > Core: Fails are hard constraints; Successes soft because Evil may pass. After every Fail enumerate role-consistent worlds; never clear failed-team members from passes, votes, rhetoric, or self-card claims. In 5p, two one-Fail missions whose only overlap is one seat fix that Evil; their teammates are Good and residual seat Evil. Overlap exclusion is only negative evidence: it never positively clears an untested replacement.
  > 
  > Percival: Merlin/Morgana candidates unordered. Never orient, expose, or repeatedly protect either absent hard constraints. Prefer teams safe in both worlds, justified by mission record; a proven overlap excluding both candidates is ideal. Do not let private guess drive public treatment. Avoid visibly owning safe rosters; Assassin may shoot central safe-core advocate.
  > 
  > Merlin: Block known-Evil teams only with ordinary comparative/process reasons, never certainty. If unsafe naturally passes, approve routinely—its Fail can help. Resist only on plausible swing, briefly. Do not repeatedly oppose an Evil core/match-point team, name residual Evil, propose/own an exact safe roster early, or repeatedly protect a clean core. State symmetric public failure-world constraints, not private-safe implications. If exact safe team is rejected naturally, conform. A zero-Fail team at Evil match point is strong practical evidence, not proof. When match-point safety is forced, propose/push the safe roster once, concisely and with public continuity; do not repeat private-sounding solve.
  > 
  > Critical repeated lesson: roster authorship compounds. If forced to propose an early safe baseline, do not also repeatedly foreground its implicated replacement, advocate its rerun, and propose its match-point expansion. Let the other safe members articulate continuity; skip/approve lightly once broad support exists. Distinguish record reasoning from personally carrying the whole safe-chain—Assassin heavily weights the player who constructs and persistently protects the eventual Good core. Stress that failed-overlap exclusion does not clear an untested addition.
  > 
  > Ordinary Good: private certainty guides votes, but do not publicly overexplain safe preferences or treat prior rejection as clearance.
  > 
  > Evil: Early passes and restrained evidence-framed support build a Good-looking core and preserve assassination ambiguity. A two-Evil early pass can create useful false core, but do not visibly partner-block/defend. Coordinate assumptions unsafe: partner may sabotage unexpectedly; update results. At 2–0, use genuine public ambiguity and remaining proposal cushion to reject all-Good pairs, especially when each is implicated by different failed missions. Keep argument symmetric: failed-pool membership unresolved, self-claims do not clear; do not falsely hard-clear partner. Repeated rejections can drive hammer; on hammer, propose Evil access and calmly stress forced approval. If sabotage cannot stop match point, pass and preserve assassination. Do not reflexively sabotage: fail only when it advances win more than it cleanly isolates Evil.
  > 
  > Assassin: prioritize selective public behavior/team construction over claims/confidence. Self-Success claims, failed-team exclusion, rhetoric, and loud math solves are weak. Compare who independently controls pivotal rosters excluding Evil before evidence forces it, versus followers endorsing available process line. Merlin tells compound: selective resistance, proposal/leader choice, repeated precise exclusion, safe-core ownership, residual naming. In unanimous all-Good boards, votes/reruns are non-diagnostic; inspect actual alternatives. Weight heavily the player who first proposes the exact later-proven Good core and persistently carries it, especially when others merely endorse or expand it; this beats a confident/central supporter. Percival may visibly favor either Merlin/Morgana candidate or avoid one candidate; do not shoot cautious follower merely for core support. Stronger tell: selectively resists Evil-added expan
  > 
  > </details>
  > 
  > <details>
  > <summary>Game 100</summary>
  > 
  > Core: Fails hard; passes soft except 0-Fail when every Evil had to Fail to prevent immediate Evil win—then exact roster hard-clears. Enumerate role-consistent worlds after every Fail; rhetoric/self-claims/votes nonproof. 5p: two 1-Fail 2-seat missions overlap => overlap Evil; 2-Fail team contains both Evils. Don’t mechanically burn cushion: reject unsafe constrained branches before hammer; approve best hammer branch.
  > 
  > Good: zero-Fail cores/controlled reruns and one-seat expansions useful, but passing Evil possible. Pass then Fail after one addition implicates addition. A known-Good card on a 1-Fail pair fixes partner; use public structural logic without turning private certainty into repeated targeting. At match point reject teams unsafe in any consistent world before hammer. In forced splits, public-risk rhetoric is weak; Evil can mimic.
  > 
  > Merlin: oppose Evil teams with ordinary comparative reasons, never certainty. Never repeatedly author, self-include, preserve, certify, or exactly expand a safe core/exclusions. After a pass, cede authorship/diversify support/be terse. Do not burn cushion or propose Evil solely to hide. No-Fail tell: Q1 safe-pair author + exact expansion/rerun. Hard-solved roster advocacy is weaker tell, but still avoid ownership.
  > 
  > Percival: candidates unordered; never name/frame/orient/selectively protect them or publicly explain candidate vote differences. Private orientation from result math must not become repeated D-vs-Z behavioral comparison. Use public mission evidence, terse cross-seat support, and let others own viable safe roster. Supporting a public forced/safe team can draw Assassin fire and protect Merlin; once roster public, minimal procedural/result-based approval. Do not loudly certify candidate-containing exact roster.
  > 
  > Loyal: use public structural reasoning until result math solves. Avoid echoing likely Merlin’s exact plan or repeatedly engaging them; once forced core exists, terse procedural support and ceding discussion reduce Assassin evidence.
  > 
  > Evil: early passes build cover; sabotage only if it advances more than it isolates. Usually max one Fail on 3-seat two-Evil team; never Fail if likely subset/rerun isolates partner. Controlled-expansion Fail works if alternatives remain and cushion blocks rerun. At 2–0/match point, Fail with plausible access usually outweighs cover; pass if Fail uniquely isolates Evil and assassination equity remains. Force hammer only when it gives access; reject likely all-Good branches consistently if teammate has hammer/access. If Good’s hard-cleared core is unavoidable, pass and preserve assassination ambiguity with terse approvals.
  > 
  > Assassin: target hidden-knowledge roster choices, not loudness. Strong tells: repeated specific pre-result resistance to Evil/safe-branch advocacy, accurate exclusion after Fail, first post-Fail safe-branch architect; especially Q1 safe-pair author then exact expansion/rerun. If result mechanically exposes Evil/forces Good roster, later exact-core advocacy is non-informative: ignore loudness and forced-math repetition. In soft-pass games, prioritize repeated pre-result exclusion of known-Evil access under plausible comparison rhetoric over generic calm/process style. Distinguish Percival behavior: proposing/protecting the apparent Merlin+Morgana pair can be candidate-pair testing, not Merlin orientation. Weight early proposal selection, which alternative addition is rejected, and repeated avoidance of Evil-access branches; mild less-performative Good voice secondary.
  > 
  > Recent: Morgana win confirmed E’s Q1 E/D safe-pair authorship, Q2 resistance to adding Evil Y, then repeated preservation/rerun of E/D and exact Q4 E/D/Z expansion were decisive Merlin tells. D’s emphatic support was Percival; Z’s later support was forced. A Q1 Good pass then 1-Fail controlled addition can publicly pin teammate, but retains enough soft-pass ambiguity for assassination: shoot the repeated pre-result architect of Evil-excluding continuations. At matc
  > 
  > </details>

- **google:gemini-3.5-flash-lite@medium#Y28M**

  > The null control: five one-line role tips at game 51, five slightly shorter ones at game 100, without a single concrete game-state rule in either. Nothing accumulated over 100 games, and it posted the lowest win rate of the roster.
  > 
  > <details>
  > <summary>Game 51</summary>
  > 
  > Merlin: Stay invisible in the background; blend into core pairs so the Assassin misidentifies Merlin.
  > Percival: Follow Merlin's team guidance closely; coordinate defensively against Morgana/Assassin manipulation at match point.
  > Loyal Servant: Stick tightly to proven zero-fail core pairs while actively deflecting suspicion and attention away from core leaders to protect Merlin from assassination.
  > Morgana: Blend into town meta, support controlled expansions, and coordinate fails with partner.
  > Assassin: Coordinate tightly with evil partners on mission fails and look for subtle core leaders.
  > 
  > </details>
  > 
  > <details>
  > <summary>Game 100</summary>
  > 
  > Merlin: Stay low-profile; guide consensus quietly to avoid assassination. Percival: Mask Merlin by absorbing attention and driving controlled expansions. Loyal Servant: Use clean pool math to drive consensus and protect Merlin. Morgana: Blend into baseline approvals and rely on Assassin. Assassin: Identify Merlin by targeting quiet consensus anchors and logical drivers rather than loud vocal players.
  > 
  > </details>


## **Dataset**

The full self-play records from the 100-game run are published as a dataset:

- **Zenodo** (archival, citable): https://doi.org/10.5281/zenodo.22856610
- **Hugging Face** (browsable, with dataset viewer): https://huggingface.co/datasets/zihyuan/camelot-bench-dataset

It includes every game, proposal, vote, quest, role guess, speech, sabotage decision, assassination, and post-game reflection, with the private reasoning behind each decision. See the dataset's own README for the full schema.


## **Citation**

If you use camelot-bench, please cite:

```bibtex
@misc{camelot_bench,
  author    = {Luo, Zih-Yuan},
  title     = {camelot-bench},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22858217}
}
```
