#!/usr/bin/env bash
# Local rehearsal of §3.4 evaluation. Mirrors the official `docker run`
# parameters (cpus, memory, mounts, env vars) so OOM / timing problems
# surface here instead of after upload.
#
# Pre-condition: docker/Dockerfile exists and the image's ENTRYPOINT
# iterates /input task_<id> dirs and writes /output/<task_id>/prediction.csv
# (see §3.6).
#
# Note: the official run also passes --network=eval_net (internal LAN
# with the model service). We omit it locally since that network doesn't
# exist on dev machines; the agent must use $MODEL_API_URL exclusively,
# so omitting the network changes nothing as long as the URL is reachable.

set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-dabench:dev}"
INPUT_DIR="${INPUT_DIR:-$PWD/data/public/input}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/artifacts/eval_output}"
LOGS_DIR="${LOGS_DIR:-$PWD/artifacts/eval_logs}"
MODEL_API_URL="${MODEL_API_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
MODEL_API_KEY="${MODEL_API_KEY:-${DASHSCOPE_API_KEY:-}}"
# §3.5 fixes the model name verbatim — kept as-is even though no public Qwen
# release uses this exact id, because the judge will only accept this string.
MODEL_NAME="${MODEL_NAME:-qwen3.5-35b-a3b}"
# §3.4 limits. On smaller hosts Docker silently falls back to host capacity,
# so leaving these at official values is safe and keeps rehearsal honest.
CPUS="${CPUS:-8}"
MEMORY="${MEMORY:-64g}"

[[ -d "$INPUT_DIR" ]]        || { echo "ERROR: input dir missing: $INPUT_DIR"; exit 2; }
[[ -n "$MODEL_API_KEY" ]]    || { echo "ERROR: MODEL_API_KEY empty"; exit 2; }
command -v docker >/dev/null || { echo "ERROR: docker not found"; exit 2; }
[[ -f docker/Dockerfile ]]   || { echo "ERROR: docker/Dockerfile missing — pure-version still needs one"; exit 2; }

# Reset output so the scorer doesn't see stale rows from earlier rehearsals.
rm -rf "$OUTPUT_DIR" "$LOGS_DIR"
mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"

echo "=== Building image $IMAGE_TAG ==="
docker build -t "$IMAGE_TAG" -f docker/Dockerfile .

# Image bytes ≠ archive bytes (gzip cuts ~30–60% on Python images), but a
# >8GB image is a strong tell that the .tar.gz will brush against the 10GB cap.
SIZE_MB=$(docker image inspect "$IMAGE_TAG" --format='{{.Size}}' | awk '{printf "%.0f", $1/1024/1024}')
echo "Image size: ${SIZE_MB} MB"
if (( SIZE_MB > 8000 )); then
    echo "WARNING: image > 8 GB; the tar.gz may exceed the §3.2 10 GB cap"
fi

echo "=== Running container (mirrors §3.4) ==="
docker run --rm \
    --cpus="$CPUS" \
    --memory="$MEMORY" \
    --memory-swap="$MEMORY" \
    -e MODEL_API_URL="$MODEL_API_URL" \
    -e MODEL_API_KEY="$MODEL_API_KEY" \
    -e MODEL_NAME="$MODEL_NAME" \
    -v "$INPUT_DIR:/input:ro" \
    -v "$OUTPUT_DIR:/output:rw" \
    -v "$LOGS_DIR:/logs:rw" \
    "$IMAGE_TAG"

echo "=== Run complete ==="

# Score only when gold is present (true on the public demo set, not on the
# hidden test set the judge uses).
GOLD_DIR="$(dirname "$INPUT_DIR")/output"
if [[ -d "$GOLD_DIR" ]]; then
    echo "=== Scoring ==="
    uv run python scripts/score_predictions.py \
        --pred "$OUTPUT_DIR" \
        --gold "$GOLD_DIR" \
        --lambda 0.1 \
        --output "$LOGS_DIR/scores.json" \
        --verbose
fi
