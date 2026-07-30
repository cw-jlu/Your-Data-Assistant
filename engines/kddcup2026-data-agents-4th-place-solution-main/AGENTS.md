# Repository Agent Instructions

## Benchmark Status Checks

- Do not declare a tmux benchmark crashed only because `ps` or `pgrep` does not
  show the Python child process from Codex. Codex may be running inside a
  sandbox/PID namespace that cannot see tmux child processes reliably.
- Use this status checklist, in order:
  1. `tmux ls` / `tmux list-panes`: session and pane still exist?
  2. benchmark log `stat` + `tail`: log mtime or completed count still moving?
  3. run dir: task files, traces, or predictions still being created?
  4. `summary.json`: generated means finished.
  5. wrapper log: explicit `run done`, next-run start, or shell error?
- Report status with evidence-backed labels: `running confirmed`,
  `no new log yet`, `possibly stalled`, `finished`, or `crashed confirmed`.
  Never say `crashed confirmed` without either an explicit error/exit marker,
  a dead tmux pane with status, or a completed wrapper failure.

## Known Phase 2 Guard Conflict

- `task_60` in the exp153 smoke exposed a source-router/self-review conflict:
  when the answer is directly visible in video, `source_router` can set
  `route=VIDEO_GROUNDED` and `sql_role=avoid`, so `answer_from_sql` correctly
  asks for a literal `SELECT`/`VALUES` answer instead of broad SQL recomputation.
- The separate universal average rule can then reject that literal answer if the
  question contains wording such as "average daily", because the SQL has no
  `AVG()`. This is a rule interaction, not an ASR failure.
- Future fix should be a separate guard experiment: when `sql_role=avoid` and
  final SQL is literal/`VALUES` with no broad `FROM`, skip the AVG-required rule
  or make it advisory. Do not silently fold this fix into exp153, which should
  remain focused on measuring ASR delta.
