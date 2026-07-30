# Submission package

This directory contains the Docker image entrypoint and build scripts for the
KDD Cup 2026 DataAgent-Bench submission.

## What the eval system gives us
- `/input/task_<id>/{task.json, context/}` (read-only mount)
- `/output/task_<id>/prediction.csv` (we must write this; missing = 0 points)
- `/logs/` (free-form logs)
- env vars: `MODEL_API_URL`, `MODEL_API_KEY`, `MODEL_NAME`
- 16 vCPU / 64 GB / no GPU
- A-board: ≤2h wall clock, B-board: ≤12h total
- **No external internet** — only the model API endpoint is reachable.

## What `submission/main.py` does
1. Discover `task_*/` under `/input`.
2. Pre-write a stub `/output/<task>/prediction.csv` so partial crashes do not
   forfeit untouched tasks.
3. Import the experiment named by `EXPERIMENT_NAME` (default
   `exp_154_v1_audio_asr`) and run its agent on every task.
4. Write the agent's answer table to `/output/<task>/prediction.csv`.
5. Log a manifest to `/logs/submission_manifest.json`.

The runner uses the experiment's normal `_run_single_task_with_timeout` (spawn
subprocess + timeout). For the current Phase 2 v4 candidate, the intended
defaults are `EXPERIMENT_NAME=exp_154_v1_audio_asr`, `MAX_WORKERS=8`, and
`TASK_TIMEOUT_SECONDS=6000`. The default submission pass is one scan only:
empty predictions are not retried unless `SUBMISSION_RETRY_EMPTY=1` is set.

When `SUBMISSION_PIPELINED_PREPROCESS=1`, video tasks are preprocessed before
agent execution. The preprocessor keeps the original task order for agent
submission: video preprocessing runs ahead with `SUBMISSION_PREPROCESS_WORKERS`
workers, but the next task is enqueued only when that task is ready. By default
the image uses 3 preprocess workers and 3 ASR workers on the 16-vCPU eval host.

## Build & ship

```bash
# From repo root.
bash submission/build.sh kobushi v1
# -> creates kobushi:v1 image and kobushi_v1.tar.gz at the repo root.

# Sanity check size (must be ≤10 GB):
ls -lh kobushi_v1.tar.gz
```

Then upload `kobushi_v1.tar.gz` to Google Drive with "Anyone with the link can
view" sharing, and email the link to the organizers with subject:

```
[KDDCup2026 Data Agents] Submission - kobushi - v1
```

## Local smoke test

We can mimic the eval mount layout against the public dataset:

```bash
mkdir -p /tmp/kobushi_smoke/{input,output,logs}
# Copy a couple of public tasks in.
cp -r data/public/input/task_26  /tmp/kobushi_smoke/input/
cp -r data/public/input/task_22  /tmp/kobushi_smoke/input/

docker run --rm \
  -v /tmp/kobushi_smoke/input:/input:ro \
  -v /tmp/kobushi_smoke/output:/output \
  -v /tmp/kobushi_smoke/logs:/logs \
  -e MODEL_API_URL="$AGENT_API_BASE" \
  -e MODEL_API_KEY="$AGENT_API_KEY" \
  -e MODEL_NAME=qwen3.5-35b-a3b \
  -e EXPERIMENT_NAME=exp_154_v1_audio_asr \
  -e MAX_WORKERS=2 \
  kobushi:v1

ls /tmp/kobushi_smoke/output/   # should contain task_22/ and task_26/ dirs
cat /tmp/kobushi_smoke/output/task_26/prediction.csv
```

The local API uses Cloudflare Access; if the eval API does too, set the
`CF_ACCESS_CLIENT_ID` and `CF_ACCESS_CLIENT_SECRET` env vars on `docker run`.
The hidden eval will not need these (the eval network reaches the model
endpoint directly).

## Choosing the experiment

Set `EXPERIMENT_NAME` to point at any `src/experiments/exp_NNN_*` package.
The current Phase 2 v4 candidate is `exp_154_v1_audio_asr`: the v1/exp149
baseline plus online video ASR. Defaults are answer_shape, prose, task PDF
preprocess, video_keyframe_note, source_router, anti_agg, anti_agg_sql_guard,
audio_asr, and prefix cache. The v2 projection pruner and final SQL guard are
not enabled by default. Its observed A-board score was 0.3358, below v1
(`exp149`, 0.3664) and v3 (`exp153`, 0.3466).
