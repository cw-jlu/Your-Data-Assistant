---
name: release-submission
description: Build a KDD Cup 2026 DABench submission Docker image, save as tar.gz with a manifest sidecar, and upload to Google Drive at gdrive:KDDCup2026/submission/. Auto-detects the next sequential version (team1418_vN.tar.gz) so naming stays compliant. Invokes scripts/release_submission.sh.
argument-hint: [auto|vN|--check]
allowed-tools: [Bash, Read]
---

# /release-submission

End-to-end flow for shipping a new KDD Cup submission.

## Arguments

- `$ARGUMENTS` is the version selector — one of:
  - `auto` (recommended) — auto-pick the next sequential `vN` based on what's
    already on gdrive
  - `v<N>` (e.g. `v3`) — explicit version. The script refuses to overwrite an
    existing version or skip numbers (= must be exactly current max + 1).
  - `--check` — dry run: just print the next version that would be used.
  - empty — default to `--check` for safety.

## Instructions

When this skill is invoked:

1. **Pre-flight** (Read tools):
   - Read `submission/Dockerfile` and verify `EXPERIMENT_NAME=...` is set to a
     real experiment under `src/experiments/`. If not, stop and tell the user
     to update the Dockerfile first.
   - Read the latest `artifacts/replications/<EXPERIMENT_NAME>/summary_*.json`
     and surface the local `n=N mean ± std` to the user — this is what they
     are about to ship.
   - Run `bash scripts/release_submission.sh --check` to confirm the next
     sequential version. Show this to the user.

2. **Confirmation**:
   - Summarise the plan in 4 bullets (exp_name, version, image size estimate,
     gdrive target path).
   - **Before any build**, ask the user to confirm if they invoked with
     anything other than `--check`. Build is 3-5 min and the upload is
     irreversible (tar.gz is permanent on gdrive once seen by organisers).
   - If the user provided no argument, treat that as `--check` and stop after
     the dry run.

3. **Execute**:
   - Run `bash scripts/release_submission.sh $ARGUMENTS` (or `auto` if user
     confirmed but didn't specify).
   - Stream the output so the user sees `docker build` progress.
   - On success, print:
     - The local archive path (`artifacts/submissions/team1418_v<N>.tar.gz`)
     - The local manifest path (`*.manifest.json`)
     - The gdrive URL via `rclone link gdrive:KDDCup2026/submission/team1418_v<N>.tar.gz`

4. **Post-flight**:
   - Update `LEADERBOARD.md` with a new row for `v<N>` — fill `date_submitted`,
     `exp`, `local 1-run` (max of the n=N runs), `local n=N mean`, `local n`.
     Leave `LB`, `rank`, `gap` blank — those go in once the user gets the
     official LB number back.
   - Update the dashboard:
     `uv run python scripts/update_docs.py`
   - Suggest a git commit + push: the LEADERBOARD.md change should be tracked.
     Do NOT commit `artifacts/submissions/` (gitignored).

## File-naming rules (enforced by the script)

- Filename: `${TEAM_ID}_v${N}.tar.gz` where `TEAM_ID=team1418` and `N` is a
  positive integer.
- Versions are strictly sequential: must equal `max(existing) + 1`.
- The script refuses to overwrite or skip — the only valid `arg` is `auto`,
  the next number, or `--check`.

## Failure modes to handle

- **rclone gdrive: not reachable** → tell user to run `rclone config` and
  re-authorise.
- **docker daemon not running** → tell user to start docker.
- **EXPERIMENT_NAME missing or unparseable** → tell user to fix the Dockerfile.
- **Disk full during `docker save`** → check `df -h artifacts/`.
- **rclone upload partial** → re-run the same command (script is idempotent
  for the upload step; tar.gz is identical so rclone copy does nothing extra).

## Notes

- Default `TEAM_ID=team1418` matches the v1 submission already on gdrive.
  Override only if the team registration changed: `TEAM_ID=xxx`.
- The manifest sidecar (`team1418_v<N>.manifest.json`) is the canonical record
  of what the tar contains: experiment, git sha, image size, local n=N score.
  Useful when the user wants to retro-correlate gdrive uploads with local
  experiment state weeks later.
- Build is reproducible (git sha is in the manifest) but not bit-identical
  (docker builds are non-deterministic). For exact reproduction use the
  archive itself.
