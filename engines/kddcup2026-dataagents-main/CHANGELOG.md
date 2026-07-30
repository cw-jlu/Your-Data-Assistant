# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Breaking (internal API only)
- **ETL post-compression pipeline moved to a structured in-memory table.**
  `agents.etl._record` introduces `Record`/`EntityTable`: compressed KV text
  is parsed once (`parse_kv_text`) and the normalize/dedup/retry/verify
  stages mutate records instead of regex-patching text lines.
  `dedup_cross_field_copies`, `retry_missing_values` and
  `verify_field_values` now take an `EntityTable`; `normalize_entity_lines`,
  `deterministic_kv_to_csv`, `pre_merge_entity_lines`
  (`agents.etl._merge`) and `repair_csv_numeric_identities`
  (`agents.etl._schema`) are removed;
  `unify_csv_synonyms` is replaced by the pure `unify_table_synonyms`
  (header/rows in, header/rows out — CSV I/O lives in the extractor).
  The final CSV is materialized exactly once, via an atomic write.

### Added
- **On-demand document ETL tool.** The main agent now receives a `run_etl`
  tool and the explorer report includes `etl_sources`, so Markdown/text/PDF
  documents are converted only when exploration marks them as required.
  Eager pre-agent prose ETL is removed; `run_etl` exposes generated CSVs back
  into the task context via relative `csv_path` values.
- **`agent.enable_answer_verifier` config field.** When `false`, skips the
  LLM answer-verifier pass entirely. Default `true` (current behavior).
- **Answer verifier drops value-sanity check.** The former "Check 4 — VALUE
  sanity" (magnitude/percentage-range heuristics) is removed; it rejected
  valid edge-case answers (100%, 0%, large growth rates) more often than it
  caught real errors. Remaining checks renumbered 5→4, 6→5, 7→6.
- **`explore_video` sub-agent tool (explorer-only).** On tasks whose context
  contains a video, the explorer can launch a video sub-agent that sees the raw
  video in the leaf context only and returns a validated structured report;
  reports also pass through to the main agent verbatim as `video_findings`.
  A single explorer adapter serves all tasks: the tool is injected per task
  into the run's registry, which the ReAct loop pushes per call.
- **`agent.video_tool_choice` config field.** `"auto"` (default) coexists with
  thinking on every backend — DashScope compatible-mode rejects named
  `tool_choice` in thinking mode. `"forced"` is an opt-in optimization that
  pins the video sub-agent's tool choice to `report` in one turn, for
  deployments verified via `scripts/smoke_video_tool_choice.py`.
- **`agent.video_max_tokens` config field.** Video sub-agent calls can use a
  larger output cap than the main ReAct loop. When rate limiting is enabled and
  `video_max_tokens` is explicitly higher than `max_tokens`, the video adapter
  reserves that larger output budget while the prompt estimator still reserves
  the fixed video input token estimate.
- **Model adapters accept per-call `tools`.** `ModelAdapter.complete(messages,
  tools=...)` lets the ReAct loop push its own registry on every turn, so the
  request-body tool advertisement and the execution dispatch table are the
  same instance. `tools=None` (or the adapter's constructor registry) keeps
  the byte-stable cached payload; overrides render fresh.
- **Deterministic video-constraint answer check.** Before the LLM verifier
  runs, the terminal-policy gate applies two zero-token checks against the
  latest `explore_video` findings: (1) reject answers whose latest execute
  step filters by the batch's year alone when the video declared a more
  specific `batch_id` (e.g. `BATCH-2021-Q4` reduced to
  `WHERE endate LIKE '2021%'`); (2) reject single-row answers when the video
  declared multiple `snapshot_dates`/`reporting_dates` and the question is
  not an aggregate. Comments are stripped before matching so `# batch
  BATCH-2021-Q4` mentions in reasoning code do not mask the actual filter.
  Lives in `src/agents/verification/video_constraints.py`; shares the
  existing `rejections_used` budget so the agent still gets the standard
  retry quota.
- **`explore_video` attaches a deterministic filter recipe to its observation.**
  When `findings.extracted_data.rules` carries enough structure (a field+operator
  filter pair, a sort_field+top_n ranking pair, or any of group_by / batch_id /
  snapshot_dates), the tool now adds a `filter_recipe` plaintext block listing
  the canonical fields and a pandas template skeleton. Closes the
  rule-translation gap the explorer / main agent used to free-write every run
  (column-name choice, threshold boundary, forgotten top_n / group_by /
  batch_id). The recipe always carries a VERIFY-before-use header — downstream
  code must still confirm exact column names against the source schema via
  `inspect_files`. Tolerant of the heterogeneous key naming the video sub-agent
  emits across runs (`filter_field` / `threshold_value` / `ranking_limit` /
  `output_fields` / `aggregation_field` / Chinese-suffixed variants).

### Fixed
- **Explorer final-step retries now target `report`.** Budget-pressure prompts
  and final-step blocking use the active registry's terminal tool instead of
  hard-coding `answer`, so explorer sub-agents are told to submit `report` and
  stop after exhausted final-step retries. The `explore` observation now marks
  synthesized fallback findings with `explorer_fallback_used`.
- **ETL drops placeholder-only rows.** The placeholder vocabulary now treats
  `null` as an empty generated value, and ETL CSV cache validation rejects rows
  where no entity or measured value survived. This prevents all-null records
  from being reused in generated task CSVs.
- **Text-mode file IO defaults to UTF-8.** Multiple call sites used
  `Path.read_text()` / `path.open(...)` without an `encoding` argument, which
  falls back to the OS locale codec (cp936 on a zh_CN Windows install). Tasks
  with Chinese `task.json` questions raised `UnicodeDecodeError` before the
  agent even started; `prediction.csv` writes silently encoded Chinese column
  headers as GBK, breaking the grader. Affected files: `benchmark/dataset.py`,
  `config.py`, `llm/token_bucket.py`, `runs/artifacts.py`,
  `tools/inspect_files.py`, `tools/read_csv.py`, `tools/read_doc.py`,
  `tools/read_json.py`. All now pin UTF-8 explicitly.

### Behavior changes (intentional tightening)
- **Video report `extracted_data` has a fixed rules/sample shape.** The
  video `report` tool now normalizes missing extracted data to
  `{"rules": {}, "displayed_samples": {}}`. Flat legacy payloads
  (keys outside `rules`/`displayed_samples`) raise a validation error
  that the agent loop surfaces as a retry prompt, so downstream agents
  can reliably distinguish query instructions from on-screen preview
  values.
- **`run.blocklist` never reaches the judged container.** The submission
  entrypoint force-clears the blocklist baked into the image config: a
  skip-list entry kept from local debugging would otherwise silently
  zero the matching judged task.
- **Video is no longer inlined into main-agent/explorer conversations.** Raw
  frames reach only the video sub-agent; the explorer discovers videos via
  `inspect_files`.
- **Terminal tool calls mixed with non-terminal calls in one turn are rejected.**
  The terminal call's arguments were written before this turn's observations;
  the agent is told to re-issue it alone next turn. Pure-terminal turns keep the
  old short-circuit behavior.
- **Sub-agent `report` payloads are now validated at the registry boundary**
  (`ToolDefinition.input_model`), with a 64 KB serialized-size backstop for the
  video report.
- **ETL LLM calls cap output tokens and reserve a smaller rate-limit budget.**
  The ETL text adapter (`_make_text_adapter`) now sets a dedicated `max_tokens`
  ceiling and, when rate limiting is on, pre-deducts a smaller
  `reserve_output_tokens` than the global agent budget. ETL outputs (compressed
  entity lines, schema) are far smaller than the ReAct
  loop's; the global reserve (8192/16384) inflated per-call TPM pre-deductions,
  so several concurrent ETL workers throttled each other against the shared
  per-key token bucket for tokens they never used. Output above the smaller
  reserve is reconciled against actual usage inside `complete()`, so no quota is
  lost.
- **Video report `displayed_samples` accepts dict OR list.** TOP-N panels and
  ordered preview rows are naturally list-shaped (e.g. ranked fund tables);
  the previous strict ``dict[str, Any]`` schema rejected them and forced the
  video sub-agent into a retry round that produced degraded findings on ~30%
  of video tasks. Empirically reproducible: across 4 same-config runs on the
  30-task video subset, the relax delivers `report`-tool error counts of 1–3
  vs 3–4 strict, and a small but durable mean-score lift (4-run avg 0.758
  vs 2-run strict 0.721, ∆ +0.037 with `task_1=1.0` gold-error override).
  Downstream consumers (templater, video_constraints checker) already
  normalize both shapes.
- **`ETL_SCRATCH_ROOT` honours `$DABENCH_SCRATCH_ROOT`.** The path constant
  in `config.py` now defaults to `/tmp/dabench` but reads the environment
  variable when set, so non-POSIX dev environments can point at a writable
  alternative (e.g. `E:/tmp/dabench` on Windows) without patching the source.

### Changed
- **ReAct example and ETL defaults increase run throughput.** The example
  config now leaves `agent.seed` unset, runs six benchmark tasks in parallel,
  and lowers the per-task timeout to 1200 seconds. Prose ETL now permits
  eight file-level orchestrator workers so extraction can keep pace with
  higher task concurrency.
- **ETL grouping expands paragraph heads until a record signal appears.**
  Paragraph-grouping prompts now keep adding leading sentences beyond the
  second sentence until the prefix contains a standalone number; paragraphs
  with no standalone number are shown in full.
- **Answer submission guidance and verifier checks updated.** Inline
  answers now consistently allow up to 10 rows and 50 cells, with larger
  payloads directed to `from_csv`; error text points models to
  `DataFrame.to_csv`. Prompt and verifier guidance now reject unit suffixes
  only on numeric measure cells while preserving requested text, code, and
  unit-label columns verbatim.
- **`parse_answer_csv` rescues narrow headers caused by full-width commas.**
  When `pd.read_csv` reports a header/data width mismatch, the parser retries
  once after replacing unquoted full-width commas `，` in the header row only.
  Data cells and other full-width punctuation are not rewritten.
- **`execute_python` subprocess stdout/stderr hardcoded to UTF-8.**
  Previously inherited the parent process's encoding (GBK on Windows), making
  Chinese output unreadable to the model. Now the model can see what it wrote
  and self-correct encoding issues.
- **Prompt: CSV artifact rule bans hand-written CSV and full-width commas.**
  Adds `NEVER hand-write CSV with string concatenation — use DataFrame.to_csv`
  to the artifact handoff rule in `DATA_OUTPUT_RULES`.
- **`inspect_files` points video files at `explore_video`, not `execute_python`.**
  The `UNSUPPORTED_HINT_BY_EXT` table now maps every extension in
  `agents.runtime.media.VIDEO_EXTENSIONS` (`.mp4`/`.m4v`/`.avi`/`.mov`/`.mkv`/
  `.webm`/`.flv`/`.wmv`) to "Use explore_video to analyze video content."
  Previously these fell through to the default "use execute_python or another
  preview tool" reason — explorer would try pandas / csv on the video file,
  fail, and burn 1–3 ReAct steps before switching to `explore_video`.
  Explorer system prompt already directs models to `explore_video` for
  video-task workflows; this aligns the per-file tool observation so it
  stops contradicting the system prompt.
- **Explorer `value_samples` field description disallows non-tabular keys.**
  The schema (`{file.col: [v1, v2]}`) already required column-suffixed keys,
  but the laconic field description left room for the explorer to use a
  video file path (`video/briefing.mp4`) as a key, paired with a dict of
  scraped values instead of the required flat list of strings — every
  occurrence wasted a `report` turn on Pydantic rejection plus a retry.
  Description now states keys MUST be `file.column`, values MUST be flat
  lists of strings, and video findings reach the caller via the
  `explore_video` observation rather than this field. Observed on video
  task runs 20260614-002/003 (task_7 explorer report rejected on
  `value_samples.video/briefing.mp4`).
- **Submission image upgraded to Python 3.14.** Aligns the container with
  the dev interpreter. All locked dependencies resolve to cp314 manylinux
  wheels (verified via `uv pip install --dry-run --no-build`), so the
  no-compiler builder stage still holds.
- **The container entrypoint runs the full CLI task pipeline.** Each
  `agents.submission` task now goes through `run_single_task` — ETL
  extraction, virtual context, subprocess hard timeout,
  prediction.csv writing — identical to `dabench run-benchmark`.
  Previously the container hand-assembled the agent and silently skipped
  the preprocessing stages. The per-task timeout derives from the global
  wall budget (`GLOBAL_WALL_SECONDS` fair share per worker, floored at
  the configured `task_timeout_seconds`, capped at the budget).
- **Task and `python_exec` subprocesses always start via `spawn`.** The
  start method no longer follows the platform default (`fork` on Linux
  ≤3.13): forking a parent that runs ThreadPoolExecutor workers can
  deadlock the child on locks held at fork time. Local (macOS) and
  container (Linux) runs now share identical process semantics; measured
  overhead is ~25 ms per child, noise against task budgets.
- **Ungrouped aggregate answers are pinned to the SQL result shape.**
  Questions asking for aggregates without grouping (e.g., max and min of
  two quantities) now submit one row with one column per aggregate;
  metric-label columns and label+value transposes are forbidden, and a
  matching `knowledge.md` use-case SQL pins the shape. Previously the
  layout was a per-run coin flip — the same task produced wide, pivoted,
  and transposed layouts across three runs — and any non-wide layout
  scores 0 against the whole-column grader.
- **Grouping prompts render paragraph heads as leading sentences.** The
  paragraph-grouping prompt used to show each paragraph's first 300
  characters; it now shows the first sentence, extended to the second
  only when the first carries no standalone number while the paragraph
  does (rhetorical openers defer the record ID to sentence two). Halves
  grouping input tokens (37-54% of the previous cost on the benchmark
  corpus) with equal-or-better verified coverage; record-ID retention
  audited at 100%.
- **No joins for columns the base table already has.** A lookup join
  against a sparse dimension table can collapse the base row set
  (task_13: an inner merge with a 50-entity doc table cut 12,962 rows
  to 48 to fetch a fund-name column the source table already carried).
  The answer rules now require checking the base table's own columns
  before any merge/JOIN.
- **Name-splitting rule scoped by writing system.** The "names are
  always two columns" rule now applies to Latin-script names only; CJK
  person names (no whitespace boundary) are submitted as a single name
  column exactly as the source stores them. Previously an English
  question over Chinese names triggered a forced 谢/理斌-style split
  that zeroed the answer against a single-column 姓名 gold.
- **Answer verifier sees the producing code and rejects NULL-filtered
  listings.** The pre-submit verifier now receives the last execute
  steps that built the answer table and rejects listing answers whose
  final SQL/python drops NULL rows of a submitted column
  (`WHERE col IS NOT NULL` / `dropna()` / `notna()`), instructing a
  rebuild that keeps NULL rows. The agent rules also state explicitly
  that exploration-time `IS NOT NULL` sampling filters must not be
  copied into the final answer query.
- **%→ratio conversion decisions are grounded in sibling sources.** Whether
  a doc-extracted percentage field converts to a decimal fraction used to
  depend on question wording alone; the ETL now scans the task's sibling
  SQLite tables and CSVs for proportion-named columns, classifies each by
  value magnitude (a fraction of a whole cannot exceed 1), and aggregates
  per naming subfamily (pct/percentage vs ratio vs proportion vs 比例 —
  observed to carry different storage semantics within one database). A
  unanimous subfamily verdict deterministically installs or cancels the
  ×0.01 conversion for fields of that subfamily; mixed or absent evidence
  falls back to the existing heuristic.
- **Answer rules: conversions follow the semantic field; junk-row filters
  use identity columns only.** The unit-conversion rule now states that a
  factor bound to an empty canonical column applies to values answered
  from its synonym/substitute column, and the NULL-preservation rule now
  forbids `notna()` filters on submitted value columns (identity-column
  filters only) — the observed failure mode that silently deleted
  legitimate NULL rows while excluding a junk row.
- **Explorer reports stay data maps.** The explorer prompt now forbids
  resolving cross-source joins during exploration (extracting key lists
  from one source to look up another, paginating to match records,
  tallying toward the final answer): it reports the join relationship in
  `join_paths` plus both schemas, and the main agent performs the actual
  join via `execute_python`.
- **Superlative tie-breaking uses the minimal literal query.** For
  "last/first/latest/earliest" questions the agent now submits the first
  row of `ORDER BY <time column> DESC/ASC LIMIT 1` without adding
  secondary sort keys or treating record ids as time — gold references
  resolve ties by the table's natural storage order.
- **Default ReAct config renamed.** The tracked example config is now
  `configs/react.example.yaml`; local overrides should use
  `configs/react.local.yaml`. The old config naming has been
  removed from user-facing docs and submission defaults.

### Removed
- **Local retrieval index subsystem.** The `run.retrieval_index` config
  block and the `search_context` / `lookup_field` agent tools are gone.
  The index was disabled by default and never used in production runs; the
  explorer's direct file inspection covers the same ground. A config that
  still sets `run.retrieval_index` is now rejected as an unknown field, and
  per-task `summary.json` entries no longer carry a `retrieval_index` key.
- **ETL code-generation fallback.** When deterministic and direct-LLM
  extraction both fail — or the compressed text exceeds the
  direct-extraction size limit — prose ETL now yields no CSV for that
  file instead of prompting the model to write a Python extraction
  script and executing it in a subprocess. Drops an unsandboxed
  code-execution path and up to three extra LLM calls (one attempt plus
  two retries) per failing file.

### Fixed
- **`answer` artifact-exists error wins over the inline-threshold error.**
  When the model has already written `_answer/answer.csv` and then submits
  inline rows at or above the threshold, the rejection now points at the
  existing artifact path with the exact `answer({"from_csv": "<path>"})`
  to resubmit — instead of telling the model to "write the table to
  answer.csv from execute_python", which it had already done.
  Trace evidence: run 20260614-009/task_31 spent 27 `execute_python` calls
  computing the table, finally wrote `answer.csv`, then hit the old
  threshold error and ran the budget out without ever recognizing the fix.
  The threshold check moved from the `AnswerInput` Pydantic validator into
  `answer` (and the verifier-fallback builder) so that artifact detection
  runs first; standalone threshold guidance stays the same for the
  no-artifact case.
- **`from_csv` rejects malformed answer CSVs with rewrite guidance.**
  The answer parser now uses pandas CSV parsing so quoted commas and
  embedded newlines are handled consistently, while malformed files whose
  data rows do not match the header width are rejected with guidance to
  rewrite the artifact via `DataFrame.to_csv(..., index=False)` and proper
  text quoting. This keeps terminal answer submission from silently
  guessing extra columns for structurally invalid CSV files.
- **Grouping batches are balanced; no more weak tail batch.** Paragraph
  grouping used fixed 40-paragraph strides, so a document's remainder
  became a sub-handful final batch whose numbered-list format anchor was
  too weak — the model drifted to prose or rejected the batch wholesale,
  and its paragraphs silently fell to the unverified leftover path
  (task_53 ed_otherdepositorycorpbs: 606 paragraphs → a 6-paragraph tail
  batch lost the last five entities' totalliabilities, flipping the
  reported maximum). Batches now split into ceil(n/40) contiguous slices
  whose sizes differ by at most one, so every multi-batch slice keeps at
  least 20 lines.
- **Fabricated primary keys with no source anchor are dropped.** A digit
  pk that never appears as a standalone token in the source document is a
  hallucinated line (typically mixing values from several entities); the
  pre-merge now drops it before grouping. Besides removing the junk row
  itself, this restores round-2 anchor uniqueness so legitimate blank-PK
  fragments sharing the same anchor value merge back into their real
  record instead of surfacing as extra rows.
- **Identity-echo fragments no longer materialize as CSV rows.** Compression
  intentionally emits identity-only lines (security code + company name,
  no record label) as merge fodder; in single-entity documents those
  anchors are shared by every record, the anchor merge cannot place the
  fragment, and it used to surface as an extra junk row the gold table
  does not have. Pre-merge now drops unplaced blank-PK fragments whose
  every populated field is an anchor or a column-constant echo across the
  real records. Fragments carrying actual data are still kept (with a
  warning) rather than silently destroyed.
- **Approximate numerals recomputed from mined table invariants.** Prose
  documents state some numbers only coarsely ("持有约六百九十七万股");
  these used to land in the CSV as the literal coarse value (6970000) or
  vanish entirely, breaking exact-match grading. The ETL now mines
  additive identities (`a = b + c`) from each table's exact rows and uses
  them to recompute values that compression tagged approximate (`~`),
  untagged coarse values that uniquely violate a validated identity at a
  column-anomalous granularity, and single missing terms whose two
  siblings are present. Repairs apply only when every candidate identity
  agrees on one solution; rows missing 2+ terms (genuine source nulls)
  stay null.
- **Value-split synonym columns unified into governance names.** When a
  materialized prose CSV leaves a governance-defined column empty — or
  partially filled, with the remaining values routed under an
  extraction-discovered column of identical meaning (e.g.
  `transferor_shares_before` vs `sumbeforetran`) — the ETL now merges
  the values under the canonical name and re-points unit conversion
  metadata at it. The merge is gated deterministically: identity columns
  are protected, any row-level disagreement vetoes the pair, ambiguous
  multi-claims and unit-family mismatches are skipped, and an
  adjudication LLM call (semantic equivalence only) is spent only when a
  structurally mergeable pair exists.
- **Fabricated sequential primary keys repaired before entity merge.**
  When compression renumbers paragraphs (1, 2, 3, ...) instead of copying
  the stated record label, the pre-merge step now rewrites the primary
  key from a sibling identity column when two deterministic signatures
  hold: the column agrees with the PK on ≥3 lines and more often than it
  disagrees, and the disagreeing PKs form a dense integer run starting
  at 1. True dual-ID documents and legitimate ID disagreements never
  qualify; previously each fabricated fragment became its own bogus row.
- **Scoped superlative answer-shape checks.** Agent self-checks and the
  answer verifier now treat global singular superlatives as one-row
  answers while preserving grouped extrema such as "for each" or "per
  category" as one row per group. The native prompt also tells the agent
  to print and verify intermediate aggregation results before embedding
  them in outer queries.
- **Native answer tool-call JSON parse recovery.** Malformed native
  `answer` arguments now keep the partial model response in the recorded step
  (`raw_tool_calls`, reasoning, usage, and response payload) and feed back
  a targeted recovery hint to resubmit large answer tables via
  `answer({"from_csv": ...})` instead of losing the response as a generic
  model error.
- **`enable_thinking` now reaches the chat template on self-hosted vLLM.**
  Previously the flag rode `extra_body.enable_thinking` at the top level,
  which the vLLM OpenAI server silently drops as an unknown field — the
  Qwen3 chat template kept whatever its built-in default was
  (`enable_thinking=True` for Qwen3 family), so flipping the config to
  `false` was a no-op on self-hosted vLLM. The flag is now placed under
  `chat_template_kwargs.enable_thinking` for vLLM and remains top-level
  for DashScope (which uses the inverse convention). Selection is
  controlled by the new `agent.backend_kind` field below.

### Added
- **LLM-verified entity grouping in prose ETL.** Paragraph-to-entity
  grouping for prose→CSV conversion is now LLM-driven with deterministic
  verification (anchor presence, hijack guard, contested-index resolution,
  distinctness gate against time-series collapse); the builtin record-ID
  regex survives only as infrastructure-failure fallback and as the
  verification primitive. Digit fragments inside formatted numbers
  ("8,408,793" → "408") no longer spawn spurious entities, entity chunks
  carry `[RECORD_ID: N]` markers so compression copies authoritative IDs
  instead of renumbering, schema sampling is token-budgeted with centered
  per-section sampling, knowledge.md governance tables act as a schema
  floor, and unit conversion factors within known scale families are
  computed deterministically instead of trusting LLM arithmetic.
  Previously up to 3 of 50 entities per document were silently dropped
  from converted CSVs (task_59 regression). The grouping prompt
  disambiguates multi-ID paragraphs — group by the document-local record
  label that introduces the record, never by domain codes
  (identification/company/security codes), dates, or amounts stated as
  the record's data (task_13 mf_fcretscalerank: 37/152 paragraphs keyed
  by advisor code split 50 entities into 83 groups, invisible to anchor
  verification) — and all its example IDs are placeholders, never
  concrete digits a model could echo into verifiable group claims. The
  response parser strips enumeration prefixes echoing the input's
  paragraph numbering ("1) 9: 1"), which previously dropped whole
  batches to the leftover path as unparsed lines.
- **`agent.backend_kind` config field (`vllm` | `dashscope` | `null`).**
  Picks where `enable_thinking` is placed in the chat completions
  request: `"vllm"` → `extra_body.chat_template_kwargs.enable_thinking`
  (required by `vllm --reasoning-parser qwen3`); `"dashscope"` →
  top-level `extra_body.enable_thinking` (DashScope OpenAI-compatible
  convention). `null` (default) auto-infers from `agent.api_base` —
  hosts containing `aliyuncs.com` or `dashscope` → dashscope, otherwise
  vllm. Explicit values win over inference. Only affects requests when
  `agent.enable_thinking` is non-null.
- **Mode-aware Qwen3 sampling defaults.** `agent.{top_p, top_k, min_p,
  presence_penalty, seed}` are now configurable; when left unset, defaults
  follow the Qwen3 model card recommendation for the selected
  `enable_thinking` mode (thinking: `0.6 / 0.95 / 20 / 0 / 0`; non-thinking:
  `0.7 / 0.8 / 20 / 0 / 1.5`). `temperature` default changed from `0.0` to
  mode-aware resolution; explicit `temperature: 0.0` is preserved but emits a
  `UserWarning` under `enable_thinking: true` (Qwen3 explicitly discourages
  greedy decoding in thinking mode — performance degradation and endless
  repetitions). `top_p` / `presence_penalty` / `seed` ride the chat completions
  top-level fields; `top_k` / `min_p` ride `extra_body` and are merged with
  `enable_thinking` rather than overwriting it.
- **`ijson>=3.3.0` dependency** for streaming JSON schema extraction in
  `read_json` / `inspect_files` on files exceeding the full-load cap.
- **`run.blocklist` task skip list.** Tasks listed under `run.blocklist` in the
  YAML config are no longer executed: the runner short-circuits to a synthetic
  failure artifact (`succeeded=false`, `failure_reason="Task is in blocklist."`,
  no `prediction.csv`) before any rate-limit pin or subprocess spawn. Entries
  accept either integers (`- 5`) or `task_<n>` strings (`- task_19` / `- "5"`),
  normalized to canonical `task_<n>` form; duplicates and non-positive numbers
  are rejected. Applies to both `dabench run-task <id>` (single task) and
  `dabench run-benchmark` (batch); blocklisted tasks still appear in
  `summary.json` with the synthetic failure reason so dashboards can grep them.
- **PR-1 agent.** ReAct loop with text-fenced JSON action protocol; eight default
  tools (`inspect_files`, `read_csv`, `read_json`, `read_doc`, `inspect_sqlite_schema`,
  `execute_context_sql`, `execute_python`, `answer`); subprocess timeout isolation.
  `inspect_files` is the zero-decision first-turn entry point — it returns columns,
  dtypes, row counts, and sqlite/json schemas for every file under `context/` in one
  call, replacing the previous `list_context` directory walker.
- **PR-2 dual-key pool + cross-process rate limiter.** `agent.api_keys` enables
  per-task key pinning by `task_index % len(api_keys)`. `agent.rate_limit` wires a
  per-key `FileTokenBucket` (RPM + TPM, `filelock`-protected, atomic state writes,
  retry-after parsing, exponential backoff with jitter).
- **Native function calling.** `agent.tool_protocol = "native"` switches to OpenAI
  native `tool_calls`; legacy `text` protocol preserved as fallback. Qwen3 `<think>`
  blocks are stripped before parsing.
- **Refactor: agent layer split.** The ReAct loop was decomposed into
  `agents.agent` (loop driver), `agents/core/protocol.py` (text/native -> unified `ToolCall`),
  `agents/core/dispatcher.py` (tool execution + terminal short-circuit on
  success), and `agents/core/recorder.py` (typed event kinds with
  stable `StepRecord` serialization). `ToolDefinition` consolidates the previous
  `ToolSpec` + handler split; `ToolRegistry` is immutable post-construction.
  `application.py` introduces `build_application(config) -> AgentApp`; the runner stops
  importing concrete adapter classes directly.
- **Quality gates.** Pyright strict on `src/agents` (177 → 0 errors).
  `pyright`, `pytest-cov`, `pre-commit` added to dev dependencies.
  `pytest --cov-fail-under=70`. `.pre-commit-config.yaml` chains
  `ruff format` → `ruff check --fix` → `pyright src` (opt-in).
- **Documentation reset.** New concise English-only docs under `docs/`:
  `architecture.md`, `cli.md`, `agent-protocol.md`, `development.md`. Each
  capability has a canonical spec under `openspec/specs/<name>/spec.md`.
- **`run-benchmark --range S-E` task selector.** Filters by `task_<n>` number,
  e.g. `--range 5-10` or `--range task_5-task_10` (both endpoints inclusive,
  numeric / `task_<n>` forms mixable). Composes with `--limit`: range filters
  first, then `--limit` truncates. Empty selection or inverted range exits
  non-zero with a parameter error.
- **`answer({"from_csv": <abs_path>})` artifact handoff.** New optional field
  on `answer` for tables already on disk: the runner reads the CSV, infers
  per-column dtypes (int before float, then bool, str), coerces cells back to
  source types, and submits an `AnswerTable` exactly like the inline path.
  XOR with `columns`/`rows` (Pydantic `model_validator` rejects both forms in
  the same call). Fail-paths (missing file, > 5MB, ragged rows, non-UTF-8,
  relative path) raise `ValueError` so the dispatcher records a `tool_error`
  and the ReAct loop retries. `execute_python` exposes
  `os.environ['DABENCH_ANSWER_DIR']` (`<task_dir>/_answer/`, mkdir'd by the
  runner before each call) as the convention write location. Ack `content`
  carries `from_csv.{path, dtypes, head_preview[:5]}` for offline trace
  audit. Empirical motivation: in run `20260429T192409664288Z`, task_8
  step 8 spent 87s and 8073 completion tokens re-emitting a 140-row table
  the model had already printed in step 7; the artifact path replaces that
  with a ~30-token `{from_csv: ...}` call and a millisecond CSV re-parse.

### Changed
- **Agent core modules split by responsibility.** The old `agents.core`
  package was removed. Runtime state and recording now live in
  `agents.runtime`, response protocol normalization in `agents.llm.protocol`,
  tool-call dispatch in `agents.tools.dispatcher`, and answer self-checks in
  `agents.verification.answer`.
- **ReAct loop implementation moved to `agents.agent`.** The old
  `agents.core.loop` module was removed; runtime wiring, package-root exports,
  tests, and architecture docs import the agent loop from the top-level
  `agents.agent` module.
- **Python package root moved to `agents`.** Runtime code now lives under
  `src/agents/`, with the ReAct loop in `agents.agent` and model adapters /
  rate limiting in `agents.llm`. The previous package import path
  and module entrypoints were removed rather than kept as compatibility shims.
- **Dashboard data source is SQLite tracing only.** `dabench dashboard` now reads
  `tracing.db_path` directly and no longer imports legacy `trace.json` artifacts.
  Default benchmark runs keep tracing disabled; enable `tracing.enabled: true` in a
  config when dashboard inspection is needed.
- **Minimum Python bumped to 3.11.** `requires-python = ">=3.11"`,
  `[tool.ruff].target-version = "py311"`, `[tool.pyright].pythonVersion = "3.11"`,
  and `uv.lock` re-resolved against the new floor. Aligns the dev/CI surface
  with the submission image (`docker/Dockerfile` already runs
  `python:3.11-slim`); 3.10 was a phantom target with no real consumers and an
  EOL of 2026-10. Source uses `from __future__ import annotations` throughout
  and no 3.10-only runtime API, so no caller-visible behavior changes.
- **`read_json` / `inspect_files` size cap raised from 5MB to 100MB; >100MB
  files now return a streamed schema instead of `kind:"unknown" skipped:true`.**
  Empirical motivator: run `20260430-025` task_2 failed because the 5MB cap on
  `zip_code.json` (7.4MB) returned no structure info, forcing the agent to
  guess the top-level shape in `execute_python`. The wrong guess
  (`for key in data.keys()[:3]: print(data[key])` against a `{table, records}`
  object) printed all 40k records — 5.5MB of stdout that exceeded the 258K
  token model context for every subsequent step until `max_steps`. New
  behavior: ≤100MB takes the existing full-load + summarize path; >100MB uses
  `ijson` to stream top-level kind/keys/length/first_item_kind. The streamed
  result mounts `streamed=True` so the model knows `value_preview` is
  unavailable while keys/length remain accurate. Output-side bounds
  (`_summarize_value_for_preview`, `_array_head_with_caps`) already cap the
  observation regardless of input size, so the input cap was duplicate
  defense.
- **`run_id` auto-generation switched from microsecond timestamps to a daily
  counter.** New format: `<UTC YYYYMMDD>-<NNN>` (e.g. `20260430-001`); the
  counter scans existing dirs under `output_dir` and resumes at `max + 1`,
  zero-padded to 3 digits. Concurrent collisions are handled via
  `mkdir(exist_ok=False)` retry inside `create_run_output_dir`. Single-day cap
  is 999; exceeding it raises `RuntimeError`. Old microsecond-format directories
  (e.g. `20260424T123456789012Z`) are preserved untouched and ignored by the
  scan (no `-` separator → never matches the new prefix).
- **`run_single_task` now derives `effective_run_id` from `run_output_dir.name`
  instead of regenerating via `resolve_run_id(config.run.run_id)`.** Previously
  the CLI `run-task` path could end up with a `ratelimit/<id>/` directory whose
  id did not match the `runs/<id>/` it belonged to; rate-limit state is now
  pinned to the same id as the run output directory.
- **Configuration loader migrated to Pydantic v2.** `load_app_config` now validates
  the YAML through `AppConfigModel`, then materializes the existing frozen dataclasses.
  Public surface (`AppConfig` return type, `ValueError` messages) is byte-compatible.
- **`AgentConfig.tool_protocol` is now `Literal["native", "text"]`** end-to-end. The
  single `# type: ignore[arg-type]` in `runner.py` is removed.
- **Tool input validation now uses Pydantic v2 models.** Each default tool gets a
  per-tool `BaseModel` (`extra="forbid"`); missing-required, wrong-type, bound, and
  unknown-field errors surface as structured `ValueError("<tool>: <field> <reason>")`
  from `ToolRegistry.execute`. `to_openai_tools()` byte output is locked by
  `tests/fixtures/tool_schemas/*.json` and the in-module `_normalize_for_vllm`
  helper, so no wire-format change ships.

### Fixed
- **`tools/handlers.py` `_answer/` artifact write under read-only `/input`
  mount.** `handle_execute_python` rooted the per-task scratch dir at
  `<task_dir>/_answer/`, but §3.4 mounts `/input` read-only inside the judged
  container — every `execute_python` call hit `[Errno 30] Read-only file
  system` before the model code ran, and the agent looped on the same error
  until `max_steps`. Scratch root now anchors at
  `/tmp/dabench/<task_id>/_answer/` for both local CLI runs and the docker
  submission, so the path resolves to a writable location regardless of how
  `/input` is mounted. Also unblocks the `answer({"from_csv": ...})` artifact
  handoff under §3.4-mirroring rehearsals (`scripts/eval_local.sh`).
- **`submission.py` silent fallback on missing config.** `_build_config`
  previously fell back to dataclass defaults (`AppConfig()`) when
  `CONFIG_PATH` (or `DEFAULT_CONFIG = /app/configs/react.example.yaml`)
  was missing — locally tuned `max_steps`, sampling, prompts, and timeouts
  silently reverted, with no signal at runtime. The judge could mark a run
  "successful" while it actually used stock defaults. New behavior: missing
  config raises `SystemExit(78)` with the resolved path in stderr, matching
  the `EX_CONFIG` exit-code convention used elsewhere in the entrypoint.
- **`tools/python_exec.py` unbounded captured stream → context overflow.**
  `_read_captured_stream` previously read the entire stdout/stderr file
  verbatim; an agent's `print(huge_object)` could put a multi-MB observation
  into the conversation history, after which every subsequent ReAct step's
  prompt exceeded the model context limit and returned 400. New behavior:
  each stream is capped at 64KB; oversized output is truncated to head 32KB
  + tail 32KB joined by `\n... [TRUNCATED: N bytes elided] ...\n`. Per-stream
  cap (stdout / stderr independent) preserves debug-friendly head/tail while
  cutting the failure mode. Same task_2 motivator as the read_json change —
  these two fixes together root-cause the cascade and prevent its recurrence
  from any other tool path.
- **`react.py` empty-tool retry off-by-one.** Guard now uses `>=` so
  `max_empty_tool_call_retries=N` halts on the Nth empty turn, matching the
  documented contract.
- **`agents/models/rate_limit.py` non-atomic state writes.** `_save` writes to a tempfile
  in `state_path.parent` then `os.replace`, eliminating the torn-write window.
- **`tools/python_exec.py` unreliable cross-process queue check.** Replaced
  `Queue.empty()` with `process.join(timeout)` followed by `get_nowait()` /
  `queue.Empty`.
- **`runner.py` second-resolution `run_id` collisions.** Default format upgraded to
  `%Y%m%dT%H%M%S%fZ` (microsecond UTC).

### Removed
- **`README.zh.md`.** The project goes English-only.
- **Legacy workflow package directory** (only stale `__pycache__` remained).
- **`scripts/`** (only stale `__pycache__`).
- **`configs/delegate_experiment.yaml`** (referenced fields that no longer exist).
- **`docs/00`–`docs/11`** (legacy plan docs describing deleted modules).
- **`DABenchPublicDataset.iter_tasks(task_ids=, difficulty=, difficulties=)`** filter
  kwargs (zero callers).

### Behavior changes (intentional tightening)
- **System prompt: entity-vs-id and answer-cell truncation rules.** Two inline
  clauses added to `_DATA_OUTPUT_RULES` (shared by text and native protocols):
  (1) when the WH-target is a content entity ("the comment / answer /
  description / name / title / message"), the answer column is the entity's
  natural-language column (Text/Body/Name/Title), not the primary key — `Id` /
  `Code` is correct only when the question literally says "ID" or "identifier";
  (2) the verification snapshot must show FULL cell values for every column in
  the final list — slicing (`[:N]` / truncation) on a cell about to be
  submitted is forbidden, since the resulting ellipsis flips the model into
  substituting the row's primary key. Empirical motivation: task_31 failed 5/5
  runs (artifacts/runs/20260501-007..011) by submitting `Id=90813` instead of
  the gold `Text` content; the trace showed the agent both misreading "what is
  the comment" as "the comment's identifier" and printing `Text[:100] + "..."`
  for verification, demoting Text to a preview column.
- **Strict bool/int coercion.** YAML quoted booleans/integers (`"false"`, `"0"`,
  `"16384"`) are now rejected. Use unquoted YAML scalars (`false`, `0`, `16384`).
  The previous behavior silently flipped quoted bools (`bool("false") == True`).
- **`read_json` becomes inspect-style.** The `max_chars` parameter is removed
  (`extra='forbid'` rejects it). Output is now a structured `{kind, length |
  key_count, head | keys + value_preview, truncated}` shape with a 5MB
  filesystem guard, parallel to `read_csv`'s SAMPLE preview contract. Top-level
  `array` length and `object` key_count are preserved even under truncation.
  Within `value_preview`, nested arrays use the same `{kind, length, head,
  head_truncation, truncated}` shape as the top-level array — but with tighter
  caps (3 items / 4KB total) so wrapper shapes like `{table, records: [...]}`
  still surface a few records' schema, heterogeneity, and sort signal by
  example without forcing an extra `execute_python` round-trip. Object values
  inside `value_preview` remain depth-1 (`kind` + `key_count` only) to bound
  recursion. Callers needing full nested extraction, filtering, or deep
  traversal must still use `execute_python` with `json.load`. Empirical
  motivation: 0/37 calls in the reference build supplied a custom `max_chars`,
  while 73% of `read_json` invocations were already followed by same-file
  `execute_python` (100% of those were cross-file joins). The nested-array
  `head` refinement responds to a follow-up trace (run
  20260429T181710702941Z) where 12/13 read_json calls hit the
  `{table, records}` wrapper and the original depth-1 design hid every record
  field.
- **`read_doc` keyword mode upgraded from "first-match window" to "all matching
  paragraphs".** When `keyword` is provided, `read_doc` now returns every
  paragraph (blank-line-separated) containing the keyword across the whole file
  — capped at 20 matches / 2KB per paragraph / 8KB total. `start_line` and
  `line_count` are ignored in keyword mode. Return shape changes:
  `keyword_match` becomes `{keyword, found, match_count, matches, matches_truncated}`
  (replacing the prior `{keyword, found, line}` + co-located `preview` window);
  `preview`, `returned_range`, and `has_more` are `None` in keyword mode.
  Empirical motivation: in the reference build, `keyword` was used in 1/77
  `read_doc` calls (1.3%) while 88% of calls hit `has_more=True`; task_46 paged
  the 1087-line `superhero.md` 7 times sequentially and still failed with
  `max_steps`. The new contract collapses such serial pagers into a single
  call. Paging mode (no keyword) is unchanged.
- **System prompt now names the file-kind boundary for SQLite tools.**
  `inspect_sqlite_schema` and `execute_context_sql` open ONLY `.db` /
  `.sqlite` / `.sqlite3` files and fail with `file is not a database` on a
  CSV / JSON / Markdown / text path. The native-prompt SQLite decision-tree
  bullet now spells this out and routes CSV / JSON paths to
  `read_csv` / `execute_python` and `read_json` / `execute_python` instead.
  Empirical motivation: in run 20260429T215118057832Z, task_13 step 3 and
  task_29 step 20 invoked `execute_context_sql` with `csv/qualifying.csv`
  and `csv/postHistory.csv`; task_13 self-recovered via `execute_python`,
  task_29 hit a 429 immediately after and ended as `missing_prediction`.
  The pre-existing bullet said only "If you find a SQLite/database file,
  call `inspect_sqlite_schema` before `execute_context_sql`" without naming
  the rejected file kinds.
- **System prompt now requires full numeric precision for computed values.**
  Averages, ratios, sums, etc. must be submitted as Python prints them (e.g.
  `60.77956989247312`), not in a human-readable rounded form (e.g. `60.78`).
  Rounding is only permitted when the question explicitly asks for a specific
  precision. The rule sits in `_DATA_OUTPUT_RULES` so both text and native
  protocols receive it. Empirical motivation: in run
  20260429T192409664288Z, task_10's `execute_python` returned
  `60.77956989247312` for the average female-superhero weight; the model's
  next turn submitted `[[60.78]]` to `answer`, losing precision that the
  grader checks. The pre-existing "Preserve cell types from the source"
  bullet only covered int-vs-float typing for source IDs and did not
  constrain decimal precision on computed values.
- **System prompt now steers large answer tables to the artifact handoff.**
  Two bullets appended to `_DATA_OUTPUT_RULES` (so both text and native
  protocols inherit them): (1) for answers >= 10 rows OR >= 50 cells,
  prefer `answer({from_csv: <abs path under DABENCH_ANSWER_DIR>})` over
  inline `columns`/`rows`. The bullet also makes column projection a
  library-agnostic obligation: decide the column set FIRST from the
  question's WH-target alone, never enumerate the source row's full key
  set. Pandas (`df[['col1','col2']].to_csv(...)`, never `df.to_csv(...)`),
  `csv.DictWriter(fieldnames=[<wh-target>])`, and hand-built rows are
  all explicitly named so the rule does not leak past whichever API the
  model picked; (2) always print a verification snapshot (row count,
  sample rows, final column list) from `execute_python` before calling
  `answer`, and run two MANDATORY self-checks on it: (a) drop any
  column whose value is `''`/`None`/`NaN` across all sampled rows
  (passthrough source column, e.g. `k_symbol`/`bank`/`account` for a
  withdrawal-listing question); (b) reverse-count: re-read the
  question, count things it explicitly names you should output, and if
  the column count exceeds that, trim down to just the WH-target.
  Empirical motivation: in run `20260429T192409664288Z`, task_8
  step 8 spent 87s autoregressively re-emitting a 140-row table the
  model had already printed in step 7 (8073 completion tokens, ~80% of
  the task wall-clock); the artifact bullet collapsed that to ~30
  tokens and 24s in run `20260429T210729531308Z`. The same follow-up
  run, however, revealed the model using `csv.DictWriter` to enumerate
  all 10 source columns of `trans.csv` (3 of them all-empty for the
  filtered subset: `k_symbol`/`bank`/`account`) instead of the single
  WH-target column `trans_id` — the verification print displayed the
  empty cells in plain sight, and the model proceeded anyway. The
  library-agnostic projection clause and the two mandatory self-checks
  close that gap.
- **`execute_python` tool description now recommends `pandas` for tabular
  work.** The tool description (a single source of truth surfaced both via
  `to_openai_tools()` for native function calling and `describe_for_prompt()`
  for the text protocol) gains a paragraph: "For tabular work (filtering,
  aggregation, projection, cross-file joins), prefer `pandas` over the `csv`
  stdlib: `df[df['col'] == ...]` and `df[['target_col']]` make WH-target
  column selection idiomatic, types are preserved automatically, and
  `df.head()` / `df.dtypes` print the verification snapshot in one line.
  Available libs include pandas, numpy, pyarrow, polars, openpyxl. The
  `csv` module is fine only for trivial reads." Empirical motivation: in
  run `20260429T210729531308Z` task_8, the model defaulted to
  `csv.DictReader` + a hand-built dict that enumerated all 10 source
  columns; pandas' natural `df[[...]]` projection idiom would have steered
  it toward the single-column answer the grader expects. Pandas import
  cost (~320ms / `execute_python` call) is small relative to typical
  task wall-clock and is amortized across multiple calls. The artifact
  bullet in `_DATA_OUTPUT_RULES` was reordered to put pandas first as
  the idiomatic write so column projection becomes the path of least
  resistance.
- **System prompt now mandates pandas-only for the artifact-write idiom.**
  The `_DATA_OUTPUT_RULES` artifact bullet is rewritten as a single
  three-statement pandas form: `filtered = df[mask]` → `answer_df =
  filtered[['target_col_a', 'target_col_b']]` → `answer_df.to_csv(out_path,
  index=False)`. The middle assignment is the only place WH-target
  columns are named, and `answer_df` as a variable name signals "this
  IS the grader's view." `csv.DictWriter` and hand-built dict rows are
  now banned outright — both encourage enumerating every source column
  (the task_8 failure class from run `20260429T210729531308Z`, where 9
  unrequested columns leaked into `prediction.csv`). Pandas is already
  a top-level dependency and always available in `execute_python`, so
  the mandate has no runtime cost. Net effect: ~50 tokens saved per
  `_DATA_OUTPUT_RULES` instance (~26% of the artifact bullet) and one
  known failure path closed at the rule level rather than relying on
  three NEVER prohibitions to catch it.
- **`execute_python` tool description now matches the prompt's pandas
  mandate.** The previous "prefer `pandas` over the `csv` stdlib"
  comparative wording is replaced with a direct "use `pandas`" so the
  tool description, system prompt, and `_DATA_OUTPUT_RULES` all speak
  with the same voice. Ergonomic hints (auto type preservation, the
  `df.head()` / `df.dtypes` one-line verification snapshot) and the
  available-libs list are kept — they are unique to this surface and
  complement the prompt without duplicating it. The "csv module is
  fine only for trivial reads" concession is preserved: it scopes to
  input reads, not the artifact-write path that the prompt bans.
- **`answer` inline payload now hard-rejected above the artifact-handoff
  threshold.** `AnswerInput` validates that inline `rows` does not exceed
  10 rows OR 50 cells; over-threshold inline submissions raise
  `answer: inline rows payload exceeds artifact-handoff threshold (got
  <N> rows, <M> cells; threshold = 10 rows or 50 cells); write the
  table to os.path.join(os.environ['DABENCH_ANSWER_DIR'], 'answer.csv')
  from execute_python and resubmit with from_csv (no columns/rows)`.
  The dispatcher routes the validation error into a `ToolErrorEvent`,
  the ReAct loop continues, and the model re-emits with `from_csv`.
  Previously the same `10+ rows OR 50+ cells` threshold lived only as a
  `_DATA_OUTPUT_RULES` "prefer" guideline; the prompt rule is preserved
  but is now backed by a hard validator. Empirical motivation: in run
  `20260430-015` task_8, `execute_python` correctly wrote the 140-row
  cash-withdrawal CSV under `DABENCH_ANSWER_DIR`, but the next turn
  inlined the table as 143 rows in the `answer` tool_call — long
  structured generation drift produced 3 fabricated `trans_id`
  rows (816277 / 816287 / 816323) plus 57 rows whose `(date, amount)`
  pairs were rewritten to a synthetic `amount=15` monthly pattern that
  pattern-matched the bank-fee rows seen earlier in the same payload.
  The artifact handoff workflow already exists (`from_csv`); this
  change closes the only path that lets the model bypass it.

### Breaking (internal API only)
- `ScriptedModelAdapter` and `ScriptedNativeModelAdapter` moved from
  the previous model module to `tests.helpers.scripted_adapters`. No
  production caller; tests updated.
