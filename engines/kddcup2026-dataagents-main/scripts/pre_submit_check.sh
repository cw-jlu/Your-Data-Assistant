#!/usr/bin/env bash
# Full pre-submission gate. Builds the image with the §3.1-mandated tag,
# rehearses §3.4, scores against the previous submission, and writes the
# .tar.gz under the exact filename the judge requires.
#
# Usage:
#   bash scripts/pre_submit_check.sh <team_id> <version_number>
#   bash scripts/pre_submit_check.sh team0042 3
#
# The team_id and version both feed into the image tag and archive name
# directly — naming the inputs at the top means we can never end up with
# main's old `dabench:vN` / `dabench_vN.tar.gz` (which §3.1 rejects).

set -euo pipefail

TEAM_ID="${1:?Usage: pre_submit_check.sh <team_id> <version_number>}"
VERSION="${2:?Usage: pre_submit_check.sh <team_id> <version_number>}"

# §3.1 — the judge enforces the naming pattern; bail before building so we
# don't burn ten minutes only to ship a nonconforming archive.
[[ "$TEAM_ID" =~ ^team[0-9]+$ ]]   || { echo "FAIL: team_id must look like team0042 (got: $TEAM_ID)"; exit 2; }
[[ "$VERSION" =~ ^[1-9][0-9]*$ ]] || { echo "FAIL: version must be a positive integer (got: $VERSION)"; exit 2; }

IMAGE_TAG="${TEAM_ID}:v${VERSION}"
ARCHIVE="artifacts/submissions/${TEAM_ID}_v${VERSION}.tar.gz"
PREV_VERSION=$((VERSION - 1))
PREV_SCORES="artifacts/submissions/v${PREV_VERSION}_scores.json"
NEW_SCORES="artifacts/submissions/v${VERSION}_scores.json"

mkdir -p artifacts/submissions

echo "=== Step 1: ruff check ==="
uv run ruff check src/

echo "=== Step 2: pytest ==="
if [[ -d tests ]]; then
    uv run pytest tests/ -q --tb=short
else
    echo "(no tests/ dir; skipping)"
fi

echo "=== Step 3: build $IMAGE_TAG ==="
[[ -f docker/Dockerfile ]] || { echo "FAIL: docker/Dockerfile missing"; exit 2; }
docker build -t "$IMAGE_TAG" -f docker/Dockerfile .

echo "=== Step 4: ENTRYPOINT/CMD check (§3.2) ==="
ENTRY=$(docker inspect "$IMAGE_TAG" --format='{{json .Config.Entrypoint}} {{json .Config.Cmd}}')
if [[ "$ENTRY" == "null null" ]]; then
    echo "FAIL: image has neither ENTRYPOINT nor CMD; §3.2 requires it runnable via 'docker run' alone"
    exit 1
fi
echo "ENTRYPOINT/CMD: $ENTRY"

echo "=== Step 5: image size warn ==="
IMG_SIZE_GB=$(docker image inspect "$IMAGE_TAG" --format='{{.Size}}' | awk '{printf "%.2f", $1/1024/1024/1024}')
echo "Image size: ${IMG_SIZE_GB} GB"
if awk -v s="$IMG_SIZE_GB" 'BEGIN{exit !(s > 9.0)}'; then
    echo "WARNING: image > 9 GB; the tar.gz size check below is the actual gate"
fi

echo "=== Step 6: rehearse on demo (mirrors §3.4) ==="
IMAGE_TAG="$IMAGE_TAG" bash scripts/eval_local.sh

echo "=== Step 7: score + regression check ==="
REGRESSION_ARG=""
if [[ -f "$PREV_SCORES" ]]; then
    REGRESSION_ARG="--fail-on-regression $PREV_SCORES"
fi
# shellcheck disable=SC2086
uv run python scripts/score_predictions.py \
    --pred artifacts/eval_output \
    --gold data/public/output \
    --lambda 0.1 \
    --output "$NEW_SCORES" \
    --verbose \
    $REGRESSION_ARG

echo "=== Step 8: save tar.gz (§3.1 naming) ==="
docker save "$IMAGE_TAG" | gzip > "$ARCHIVE"
TGZ_SIZE_GB=$(wc -c < "$ARCHIVE" | awk '{printf "%.2f", $1/1024/1024/1024}')
echo "Archive: $ARCHIVE (${TGZ_SIZE_GB} GB)"
if awk -v s="$TGZ_SIZE_GB" 'BEGIN{exit !(s > 10.0)}'; then
    echo "FAIL: tar.gz > 10 GB (§3.2 hard cap)"
    rm -f "$ARCHIVE"
    exit 1
fi

echo "=== All checks passed ==="
echo "Archive: $ARCHIVE"
echo "Next:"
echo "  1. Upload $ARCHIVE to Google Drive"
echo "  2. Set sharing: 'Anyone with the link' → 'Viewer' (§3.3 — wrong perms = invalid submission)"
echo "  3. bash scripts/submit.sh $TEAM_ID $VERSION '<gdrive_link>'"
