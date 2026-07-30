<div align="center">

# KDD Cup 2026 Data Agents Challenge — 4th Place Solution

### 🥇 Phase 1 final: **1st place** &nbsp;·&nbsp; 🏅 Phase 2 final: **4th place — Merit Award**

*Team KOBUSHI (team 1418) — official results at [dataagent.top](https://dataagent.top)*

</div>

---

## Results

| Phase | A-board | B-board (hidden) | Final | Rank |
| --- | --- | --- | --- | --- |
| Phase 1 final | 0.5965 | **0.6812** (highest of all teams) | 0.6685 | **1 / 700+ entered teams** (297 on the final board) |
| Phase 2 final | 0.5833 | 0.5889 | 0.5880 | **4 / 35 finalists — Merit Award** |

## What this is

The complete, unedited experiment history behind Team KOBUSHI's solution:
**139 commits and 181 experiment packages (`src/experiments/exp_001` … `exp_173`, several in multiple variants)**
from May to July 2026, built on top of the official
[starter kit](https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit).

The task: an autonomous data agent that answers natural-language analysis
questions over heterogeneous data packages — SQL databases, adversarial
narrative prose documents, videos, and per-source knowledge files — powered by
**Qwen3.5-35B-A3B** (the mandated model) and evaluated on constrained hardware
(16 CPU / 64 GB RAM / no GPU).

This repository is published **with its full commit history rather than a
cleaned snapshot, deliberately**: the ~170 approaches that *failed* — and the
decisions to reject them — are as much a part of the result as the handful
that survived. The history is the work.

## Solution: a phased ReAct agent

The heart of the solution is a **phased ReAct architecture** (`phased_agent.py`
inside each experiment package): a single Qwen3.5-35B-A3B agent whose ReAct
loop is segmented into four explicit phases —

<div align="center"><strong>PLAN → EXPLORE → ANSWER → VERIFY</strong></div>

Phase transitions are not left to the model's judgment. Moving on is an
explicit `complete_phase` action guarded by deterministic gates — a minimum
number of successful EXPLORE queries before ANSWER becomes reachable,
mandatory consumption of video evidence when the task ships a video,
answer-shape checks — and tool visibility and prompting are scoped per phase.
For a 35B-class local model, this is the difference between an agent that
wanders and one that reliably gathers evidence before committing to an answer.

Around that backbone, smaller measured-impact components:

- domain routing with edition-aware knowledge handling, validated for
  generalization against ~10k external questions;
- a deterministic "is prose needed?" gate plus prose→table extraction for
  adversarial narrative documents;
- fail-closed output guards (DISTINCT / LIMIT / answer-shape) that prefer a
  safe default over a clever guess — a design that survived the hidden final
  set where several higher-ranked A-board teams collapsed;
- a statistics-aware submission strategy (measuring same-code run-to-run
  variance on the leaderboard and correcting for best-of-N bias when
  choosing the final version).

## Experiment framework

Everything above was found by iteration, and the repository's two-layer
layout — shared `kobushi_core` infrastructure plus one isolated package per
experiment (`src/experiments/exp_NNN`) — exists purely to make that iteration
fast and honest: 181 controlled experiment packages, replicated benchmark
runs (`scripts/replicate_bench.sh`), and one unified scorer
(`kobushi_core.eval.evaluate_run`) so that no experiment could flatter
itself with its own metric.

## Quick reproduction

Requirements: [uv](https://docs.astral.sh/uv/), Docker (submission images
only), and an OpenAI-compatible endpoint serving **Qwen3.5-35B-A3B**.

```bash
uv sync
cp .env.example .env   # set AGENT_API_BASE / AGENT_API_KEY to your endpoint
# download the official demo data from dataagent.top into data/public/input/

# Run the final agents on the local benchmark, then score them:
bash scripts/launch_bench.sh exp_172_ehr_distinct    # Phase 2 final agent
bash scripts/launch_bench.sh exp_137_math_advisor    # Phase 1 final agent
uv run kobushi evaluate-run artifacts/runs/<run_dir>

# Build the exact submitted images (git tags mark the submitted states):
git checkout phase2-final && bash submission/build.sh kobushi v9   # exp_172_ehr_distinct
git checkout phase1-final && bash submission/build.sh kobushi v11  # exp_137_math_advisor
```

The submission image is what the evaluation host ran: it receives
`MODEL_API_URL` / `MODEL_API_KEY` / `MODEL_NAME` as env vars and mounts
`/input`, `/output`, and `/logs` (details in `submission/README.md`).

## Authorship

Every one of the 139 commits in this repository was authored by
**[Koki Shibata](https://github.com/kekshibata)**, who carried out the system
design, all experimentation, evaluation, and submission decisions for the
team, using Claude (Claude Code), Codex, and Cursor as coding agents
throughout. Team KOBUSHI entered as a three-person team; the author thanks
the teammates who provided the GPU infrastructure that served the model for
local benchmarking — 2× NVIDIA DGX Spark, plus a rented cloud server with an
NVIDIA RTX PRO 6000 in the final stretch of the competition — without which
this experiment throughput would not have been possible.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/kobushi_core/` | Shared agent infrastructure (model client, eval, tools) |
| `src/experiments/exp_NNN_*/` | One package per experiment iteration (181 total) |
| `LEADERBOARD.md` | Submission-by-submission leaderboard log with analysis |
| `EXPERIMENTS.md`, `PRIORITIES.md`, `BACKLOG.md` | Running lab notebook |
| `docs/` | Selected analyses (task analysis, harness designs, final analyses) |
| `scripts/` | One-off PoCs, benchmarks, and audits |

Competition data is **not** included (`data/` is untracked, per the challenge
terms); to run anything, obtain the official data from
[dataagent.top](https://dataagent.top) and place it under `data/public/`.
Internal hostnames and cloud account identifiers have been redacted from the
history; no other edits were made.

## License

The author's original work in this repository (everything outside the starter
kit scaffolding) is released under the [MIT License](LICENSE).
Files derived from the official starter kit remain subject to the challenge
organizers' terms; the original starter kit README is preserved in git
history.
