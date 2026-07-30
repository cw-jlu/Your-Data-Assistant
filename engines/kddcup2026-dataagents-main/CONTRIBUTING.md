# Contributing

Conventions for code contributions to this repository. They apply equally to human contributors and AI coding assistants (Claude Code, Cursor, etc.).

For developer-loop topics (toolchain, quality gates, project layout) see
[`docs/development.md`](docs/development.md). For the AI-assistant operational
digest see [`CLAUDE.md`](CLAUDE.md).

## Git commit messages

Follow **Conventional Commits + 50/72**, in **English**.

**Format**

```
<type>(<scope>): <subject>          # imperative, ≤50 chars (hard max 72), no period
                                    # ↑ blank line ↓ (only if a body exists)
<body wrapped at 72 chars>          # optional; only when "why" is non-obvious
```

**Rules**

- **Subject ≤50 chars** (hard max 72 including the `type(scope):` prefix). Imperative mood: `fix`, not `fixed` / `fixes`.
- **Body only when it adds value** — explain *why* or a non-obvious trade-off, never restate the diff. Skip for renames, formatting, trivial fixes.
- **One logical change per commit** — no `feat(a,b,c): X + Y + Z`. Split composite work.
- **English only** for subject and body. Code comments/docstrings may stay in Chinese; intrinsic Chinese proper nouns (file paths, teammate names like 华/吴) may appear inline.
- Valid `<type>`: `feat`, `fix`, `docs`, `refactor`, `style`, `test`, `chore`, `perf`, `ci`, `build`. Lowercase. `<scope>` optional.
- Avoid vague subjects: `update stuff`, `misc`, `WIP`, `fix bug`.

**Examples**

Good:

```
refactor(tools): rename _schema_normalize to schema_normalize
fix(runner): prevent zombie subprocess on timeout
feat(agents): add token-bucket rate limiter
```

Bad:

```
feat(agents,tools,ui): align ... + debug UI + inspect_files     # >72 chars, 3 things in 1
docs(agent): 给全量源码添加中文注释                               # not English
refactor: updated some files                                    # past tense + vague
```

## CHANGELOG maintenance

`CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Maintain it **synchronously with code changes**, not at release time.

**Trigger — write a CHANGELOG entry when:**

- A user-visible CLI flag, subcommand, or config field is added / removed / renamed.
- Public Python API in `src/agents/` changes (signature, default, return shape).
- Output artifact format changes (`traces.db`, `prediction.csv`, `summary.json` schema).
- Behavior changes that an existing user would notice (validation strictness, error messages, default values).
- Dependency / build / packaging changes that affect how users install or run the project.

**Skip — do not write a CHANGELOG entry for:**

- Pure internal refactors (file splits, renames of private symbols, dead-code removal) with no caller-visible diff.
- Tests, fixtures, CI config, lint config, type-checker config.
- Doc-only changes (`docs/`, `README.md`, `CLAUDE.md`).
- Bug fixes whose effect is "code now matches its documented contract" — unless the previous broken behavior was load-bearing for some users.

**How to write:**

- Add the entry under `## [Unreleased]` in the same commit (or PR) as the code change. Never let code land without its CHANGELOG entry — `[Unreleased]` is cheap, retroactive archaeology is not.
- Pick the right subsection: `Added` / `Changed` / `Deprecated` / `Removed` / `Fixed` / `Security`. Project-specific extras already in use: `Behavior changes (intentional tightening)`, `Breaking (internal API only)`.
- Format: `` - **<Subject>.** <One-to-three-sentence body explaining what changed and the user-visible impact.> ``. Bold-prefix the headline so the file scans top-down.
- Write *what users see*, not *what you did*. "`--range S-E` filters tasks by number" beats "added `_parse_task_range` to `cli.py`".
- English only, same rule as commit messages.

**Release time (maintainer):**

- Rename `## [Unreleased]` → `## [X.Y.Z] - YYYY-MM-DD` and create a fresh empty `## [Unreleased]` above it.
- Squash duplicates / fold related bullets only if it improves the narrative; don't rewrite history.

**Example entry (good):**

```
- **`run-benchmark --range S-E` task selector.** Filters by `task_<n>`
  number, e.g. `--range 5-10` or `task_5-task_10` (both endpoints inclusive).
  Composes with `--limit`: range filters first, then `--limit` truncates.
```

**Example entries (bad):**

```
- Updated CLI.                                          # what changed? for whom?
- Added _parse_task_range helper in cli.py.             # internal symbol — wrong audience
- 加了 --range 参数。                                    # not English
```
