# KDD Cup 2026 DataAgent-Bench submission image.
#
# Build:
#   docker build -t <team_id>:v<N> -f submission/Dockerfile .
# Save:
#   docker save <team_id>:v<N> | gzip > <team_id>_v<N>.tar.gz
#
# The eval system runs the container with /input, /output, /logs mounted
# and MODEL_API_URL / MODEL_API_KEY / MODEL_NAME in the environment.

FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps for Python wheels (numpy/pandas/duckdb/pyarrow may need them).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (faster, deterministic resolver matching local dev).
RUN pip install --no-cache-dir uv==0.4.18

WORKDIR /app

# Copy dependency manifests first for cacheable layer.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Install with no dev extras. uv pip install -e . exposes the experiments
# package so `python -m submission.main` can import experiments.exp_NNN_*.
RUN uv pip install --system -e .

# v11: the v10 ICL few-shot stack (CPU torch + sentence-transformers + MiniLM +
# BULL/EHRSQL gold-SQL corpus) is REMOVED. Rules review: bundling same-source
# (homologous) benchmark gold SQL and retrieving it at answer time falls under the
# prohibited "use homologous data (same source data), identical databases" clause
# (dataagent.top/rules, prohibited behaviors). Embedding models per se are allowed,
# but the corpus is not — so the whole retrieval stack goes.

# v6 (exp_166_domain_rules) does not use ASR (no audio_asr module); the
# faster-whisper model download from earlier ASR-based versions is dropped.

COPY submission ./submission
# v10: data/external/domain_db (BIRD schema-description docs) DROPPED — the audit
# showed nothing in the exp_172 code path reads it (dead weight since the exp_166
# era), and carrying same-source-adjacent reference material we don't use is pure
# audit noise. If a future experiment needs it again, re-add the COPY consciously.

# Default knobs (the eval system can override at runtime).
#
# v6 (2026-05-14): exp_122_column_auditor + scorer-alignment fixes
#   Base = v5 (= 3-run variance mean 0.7950 ± 0.0304).
#   Changes from v5:
#     1. adaptive_vote `_norm_value` round granularity 3dp → 2dp to match
#        the official scorer's Decimal quantize (ROUND_HALF_UP, 2dp). Fixes
#        a vote-dedup miss where attempts differing only at the 3rd decimal
#        were kept as separate columns and inflated extras_ratio at eval
#        time (= cost ~0.005 score per affected task).
#     2. DuckDB execute_sql prose-file guard: reject any `read_csv` /
#        `read_json` / similar attempt to parse `.md`, `.txt`, or
#        `.markdown` files. These are free-form prose; SQL parsing burns
#        steps in a structurally impossible task. Error message redirects
#        the agent to `read_doc` + literal `SELECT-wrap`.
#   Expected LB delta: +0.005 ~ +0.015 (= clean wins on prose tasks +
#   no-op safety on SQL tasks). LB transfer 94% considered.
# Phase 2 v4 candidate (2026-06-13): exp_154_v1_audio_asr.
#   = v1 (exp_149_modality, highest A-board score 0.3664) base + online video
#   ASR, deliberately DROPPING the v2 additions (projection pruner + final SQL
#   guard) that correlated with the v2 A-board regression (0.301). Rationale:
#   both are demo-tuned LLM rewrites with unproven transfer; v1>v2 on LB.
#   Observed A-board v4 result: 0.3358. This is above v2 but below v1/v3, so
#   do not treat exp154 as the current best submission profile.
#   Fixed defaults: answer_shape + prose + task PDF preprocess +
#   video_keyframe_note + model source_router + anti_agg +
#   anti_agg_sql_guard + task-scoped prefix cache + online video ASR.
#   Preprocessing pipeline (pipelined keyframe-note + ASR) is inherited
#   unchanged from v3.
# Phase 2 v6 candidate (2026-06-23): exp_166_domain_rules.
#   = v1 (exp_149_modality, A-board best 0.3664) base + deterministic domain
#   router (fund/stock/macro via ccks source-id token; eicu/mimic via schema
#   fingerprint; else 'other') injecting per-domain schema/output notes +
#   context-driven modality note + value-fidelity preamble + math-advisor gated
#   to domain='other' only + source-router avoid-deadlock escape valve
#   (guard>=1 & 0-rows>=3 -> release) + L52 deletion. No ASR (forked from
#   exp_149, not exp_154): audio_asr.py is absent and submission.main disables
#   the ASR stage gracefully; video evidence comes from video_keyframe_note.
#   Local single-run mean ~0.69 (note: local is anti-correlated with A-board).
# Phase 2 v7 candidate (2026-06-24): exp_167_domain_pruner.
#   = v6 (exp_166) with: (1) the column auditor gated by the domain router — a
#   stronger finance/EHR projection auditor (drops date/period/id/region/name
#   metadata, time-series keeps only the metric) fires for non-'other' domains,
#   the conservative general auditor for 'other'; (2) the math-advisor gate
#   REVERTED (EXP167_MATH_OTHER_ONLY default OFF): gating the formula to
#   domain='other' regressed 4 finance tasks in testing — the formula acts as an
#   aggregation-structure anchor, so it now fires on every task again. domain +
#   modality notes, value-fidelity, source-router escape valve, L52 deletion all
#   inherited from v6.
#   v7 REBUILD (2026-06-24): adds (3) EXP167_EXPLORE_SHAPE=1 — the
#   NULL/BLANK + DISTINCT preservation note (previously only in ANSWER/VERIFY)
#   is now also kept in mind during EXPLORE, where the agent settles on its
#   result query. Framed around the RETURNED query only (diagnostic IS NOT NULL /
#   DISTINCT mid-exploration stay allowed). On an unstable 9-task A/B (OFF x3 vs
#   ON x3) this netted +0.125 with zero regressions (legit-DISTINCT guards held).
#   The flag defaults OFF in code (for clean local A/B); the image forces it ON.
#   v7 REBUILD #2 (2026-06-25, git 22bc193) — output-fidelity bottleneck fixes
#   (the trace analysis found "right rows, wrong values" = 73% of attackable loss):
#     (4) DISTINCT literal-only: SELECT/COUNT(DISTINCT) only when the question
#         literally says 去重/唯一/distinct/unique, never inferred from "how many X"
#         (external BULL: DISTINCT in 0.68% of 9801 SQLs). Fixes task_26 (0->1.0).
#     (5) datetime fidelity: never SUBSTR/strftime/CAST a date in the SELECT to
#         strip the time (date funcs in WHERE only). Fixes task_21.
#     (6) avoid guard is now warn-only (no hard block of broad SQL): it had killed
#         correct SQL (task_26) and deadlocked task_60. PoC: both recover, no regress.
#     (7) fund note disambiguates 基金管理人 (management co) vs 基金托管人 (trustee/bank).
#     (8) value fidelity preserves separators (、 etc.) in ANY value, not just names.
# Phase 2 v8 candidate (2026-06-28): exp_171_prose_router.
#   = v7 (exp_167) lineage + prose-to-table extraction (exp_170) NOW GATED by a
#   deterministic prose-needed router (design ③ "gate-then-select").
#   The gate (prose_extractor.prose_gate): one T=0 LLM call names the target
#   TABLE(s) from question+knowledge.md, then a deterministic check —
#     target in queryable views        -> prose NOT needed (skip extraction)
#     target == a doc stem (or substr) -> prose needed, that doc
#   needed = bool(matched_docs). When 不要, extraction is skipped entirely so NO
#   prose view is injected (= kills the always-extract decoy regression, e.g.
#   task_14). When 要, the existing preview-based select_prose_docs picks the doc.
#   Bench (EC2 full-60 x3, EXP171_EXPLORE_SHAPE+PROSE_EXTRACT+PROSE_GATE):
#     RUN 0.7931 / 0.7750 / 0.7819 -> mean 0.7833 (std 0.0074).
#     vs v7 0.7500 = +0.033; vs always-extract(no gate) 0.7472 = +0.036.
#     task_14 0.0 -> 0.75/1.0/1.0; real prose wins (task_3/8=1.0) preserved;
#     the 5 archive-name-collision false-positives did NOT regress (harmless).
#   Ships EXACTLY the benched flag set (the 3 non-default flags forced ON below);
#   EHR-LIMIT / math gates etc. left at their exp_171 defaults to match the bench.
# Phase 2 v9 candidate (2026-07-01): exp_172_ehr_distinct.
#   = v8 (exp_171 prose gate) + a DOMAIN-GATED DISTINCT fix for EHR. The finance-
#   empirical anti-DISTINCT guard ("DISTINCT is almost never correct; single-table
#   lists keep duplicate rows") INVERTS the correct behaviour on EHR set-valued
#   questions ("how is <drug> delivered / what routes/methods for X"), whose gold is
#   `SELECT DISTINCT route`. For domain in {eicu,mimic,ehr} (deterministic router)
#   we (a) swap the answer-shape prompt block for an EHR variant that allows DISTINCT
#   on set-valued questions, and (b) skip the hard SQL DISTINCT-reject guard in the
#   answer tool. Finance/other are byte-identical to v8 (flag+domain gated).
#   Evidence: external EHRSQL (MIMIC-IV) — 32.8% of EHR gold SQL use DISTINCT (vs
#   ~0.6% for finance); "how delivered / what methods" buckets are 90-100% DISTINCT.
#   Focused A/B (gpuhost, task_33/35/37 route questions): 0.0 -> 1.0 x3 (guard OFF
#   -> ON), agent writes SELECT DISTINCT route and it passes.
#   Caveat: the demo-60 EHR set has NO route/method questions (its DISTINCT cases —
#   COUNT(DISTINCT) with a literal "distinct", most-frequent GROUP BY — already pass),
#   so A-board transfer is uncertain; shipped as a safe, correct domain-scoped fix.
#   Ships the v8 flag set + EXP172_EHR_DISTINCT=1.
# Phase 2 v10 (2026-07-03): exp_172_ehr_distinct (= the v9 experiment) + defensive pack.
#   NOTE: an earlier v10 draft (exp_173 retrieval few-shot over a bundled BULL/EHRSQL
#   gold-SQL corpus) was built and uploaded to our own Drive but NEVER submitted
#   (no link emailed), then withdrawn after a rules review: bundling same-source gold
#   SQL falls under the prohibited "use homologous data (same source data), identical
#   databases" clause. This clean build replaced that unsubmitted draft in place.
#   Deltas vs v9 (all defensive, no modelling change):
#   (0) MULTI-VIDEO support: the keyframe pipeline previously took hits[0] of the
#       first matching extension — every additional video was silently invisible
#       (demo-60 never exposed this: all its video tasks ship exactly one). Now every
#       video gets its own keyframe extraction + note, concatenated under
#       "## video: <path>" headers; the note char cap scales per video. Single-video
#       tasks behave exactly as v9. domain_db (unused BIRD schema docs) dropped.
#   Harness change:
#     After the initial full pass, tasks whose prediction.csv is missing / empty /
#     header-only are re-run once with the remaining wall-clock budget
#     (SUBMISSION_RETRY_EMPTY=1 — the long-existing mechanism, never before enabled).
#     Two harness fixes make it safe to enable:
#     (1) SUBMISSION_BOARD=AUTO — board detected from mounted task count (~60 = A/2h,
#         >=100 = B/12h), so the wall budget matches the real kill deadline instead of
#         a hard-coded B/12h assumption that planned A-board retries past the 2h wall.
#     (2) wave-aware retry timeout — the retry batch runs in ceil(targets/workers)
#         sequential waves; the per-task timeout is budget/waves (was: full budget per
#         task, overrunning the wall by up to the wave count).
#     Completed predictions are never touched; a retry can only overwrite a stub.
#   Rationale: A-board loss analysis showed wall-clock/empty-prediction tasks are a
#   first-class loss source (v1/v2: workers×2h wall). This recovers them with zero
#   modelling change and zero compliance risk.
ENV EXPERIMENT_NAME=exp_172_ehr_distinct \
    EXP172_EXPLORE_SHAPE=1 \
    EXP172_PROSE_EXTRACT=1 \
    EXP172_PROSE_GATE=1 \
    EXP172_EHR_DISTINCT=1 \
    MAX_WORKERS=8 \
    MAX_STEPS=64 \
    TEMPERATURE=0.6 \
    TASK_TIMEOUT_SECONDS=6000 \
    SUBMISSION_BOARD=AUTO \
    SUBMISSION_PIPELINED_PREPROCESS=1 \
    SUBMISSION_PREPROCESS_WORKERS=3 \
    SUBMISSION_RETRY_EMPTY=1 \
    SUBMISSION_RETRY_WORKERS=2 \
    KDD_INPUT=/input \
    KDD_OUTPUT=/output \
    KDD_LOGS=/logs

ENTRYPOINT ["python", "-m", "submission.main"]
