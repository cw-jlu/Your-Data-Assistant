#!/usr/bin/env bash
# End-to-end submission release: build → save → upload to Google Drive.
#
# Usage:
#   bash scripts/release_submission.sh <version>     # explicit, e.g. v2
#   bash scripts/release_submission.sh auto          # auto = next vN from gdrive
#   bash scripts/release_submission.sh --check       # dry run: show next vN, no build
#
# Strict file-naming convention enforced:
#   ${TEAM_ID}_v${N}.tar.gz   (sequential N starting at 1)
#
#   - Refuses to use a version that already exists on gdrive.
#   - Refuses to skip numbers (= must be exactly 1 higher than current max).
#
# What it does:
#   1. Reads EXPERIMENT_NAME from submission/Dockerfile (sanity check).
#   2. Builds the docker image as `${TEAM_ID}:${version}`.
#   3. Saves to artifacts/submissions/${TEAM_ID}_${version}.tar.gz.
#   4. Writes a sidecar manifest (exp_name, build time, image size, git sha,
#      local n=N score from artifacts/replications/).
#   5. Uploads BOTH to gdrive:KDDCup2026/submission/ via rclone.
#
# Requires:
#   - docker (engine running)
#   - rclone with `gdrive:` remote configured
#   - submission/Dockerfile pointing to a real EXPERIMENT_NAME
#
# Env overrides:
#   TEAM_ID=team1418         (image tag prefix; default matches v1 naming)
#   GDRIVE_PATH=KDDCup2026/phase2_submission   (target folder on gdrive:)
#   DOCKER_PLATFORM=linux/amd64  (official evaluation platform)
#   SKIP_BUILD=1             (re-upload existing tar.gz without rebuilding)
#   SKIP_UPLOAD=1            (build only, don't upload)

set -euo pipefail

arg=${1:?usage: release_submission.sh <vN|auto|--check>}
# Default matches the v1 submission already on gdrive (team1418_v1.tar.gz).
# Override with `TEAM_ID=xxx bash ...` if your team registration uses a
# different ID. The "team" prefix is part of the official ID.
TEAM_ID=${TEAM_ID:-team1418}
GDRIVE_PATH=${GDRIVE_PATH:-KDDCup2026/phase2_submission}
DOCKER_PLATFORM=${DOCKER_PLATFORM:-linux/amd64}

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"

# --- Determine next sequential version from gdrive --------------------------
# Look at gdrive:${GDRIVE_PATH}/${TEAM_ID}_v<N>.tar.gz, find max N, suggest N+1.
existing_files=$(rclone lsf "gdrive:${GDRIVE_PATH}/" 2>/dev/null || true)
existing_versions=$(printf '%s\n' "$existing_files" \
    | grep -E "^${TEAM_ID}_v[0-9]+\.tar\.gz$" \
    | sed -E "s/^${TEAM_ID}_v([0-9]+)\.tar\.gz$/\1/" \
    | sort -n || true)
max_n=$(echo "$existing_versions" | tail -1)
[ -z "$max_n" ] && max_n=0
next_n=$(( max_n + 1 ))

if [ "$arg" = "--check" ]; then
    echo "  team_id:           $TEAM_ID"
    echo "  gdrive path:       $GDRIVE_PATH"
    echo "  existing versions: $(echo $existing_versions | tr '\n' ' ')"
    echo "  → next sequential: v${next_n}"
    exit 0
fi

if [ "$arg" = "auto" ]; then
    version="v${next_n}"
    echo "  → auto-selected next sequential version: $version"
else
    version="$arg"
fi

# Validate version format
if [[ ! "$version" =~ ^v[0-9]+$ ]]; then
    echo "FATAL: version must match ^v[0-9]+$, got '$version' (e.g. v2)" >&2
    exit 2
fi

# Strict sequential check
n=${version#v}
if echo "$existing_versions" | grep -qx "$n"; then
    echo "FATAL: $version (= ${TEAM_ID}_v${n}.tar.gz) already exists on gdrive." >&2
    echo "       The submission system requires unique sequential versions —" >&2
    echo "       use 'auto' or specify v$((max_n+1))." >&2
    exit 2
fi
if [ "$n" -gt "$next_n" ]; then
    echo "FATAL: $version skips numbers (existing max=v${max_n}, expected next=v${next_n})." >&2
    echo "       Versions must be strictly sequential — use 'auto' to take v${next_n}." >&2
    exit 2
fi
if [ "$n" -lt "$next_n" ]; then
    echo "FATAL: $version is older than current max=v${max_n}." >&2
    echo "       Use 'auto' or v${next_n}." >&2
    exit 2
fi

# 1. Read EXPERIMENT_NAME from Dockerfile for the manifest
exp_name=$(grep -E '^ENV EXPERIMENT_NAME=' submission/Dockerfile | head -1 \
    | sed -E 's/.*EXPERIMENT_NAME=([^ \\]+).*/\1/' | tr -d ' ')
if [ -z "$exp_name" ]; then
    echo "FATAL: could not parse EXPERIMENT_NAME from submission/Dockerfile" >&2
    exit 2
fi
echo "  experiment: $exp_name"
echo "  version:    $version  (next sequential after v${max_n})"
echo "  team_id:    $TEAM_ID"

# 2/3. Build + save (skip if SKIP_BUILD=1 and tar exists)
mkdir -p artifacts/submissions
image_tag="${TEAM_ID}:${version}"
archive="artifacts/submissions/${TEAM_ID}_${version}.tar.gz"

if [ "${SKIP_BUILD:-0}" != "1" ] || [ ! -f "$archive" ]; then
    echo "==> docker build --platform $DOCKER_PLATFORM $image_tag (~3-5 min)"
    docker build --platform "$DOCKER_PLATFORM" -t "$image_tag" -f submission/Dockerfile .

    echo "==> docker save | gzip > $archive"
    docker save "$image_tag" | gzip > "$archive"
else
    echo "==> SKIP_BUILD=1 and $archive exists, reusing"
fi

# Image size for manifest
bytes=$(stat -c%s "$archive")
gb=$(awk "BEGIN{printf \"%.2f\", $bytes / 1024 / 1024 / 1024}")
mb=$(awk "BEGIN{printf \"%.0f\", $bytes / 1024 / 1024}")

# 4. Manifest (sidecar JSON next to tar.gz)
manifest="${archive%.tar.gz}.manifest.json"
git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
git_msg=$(git log -1 --pretty=%s 2>/dev/null || echo "?")
build_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Try to pick up local n=N score for this experiment
local_score="null"
local_n="null"
local_std="null"
summary=$(ls -t "artifacts/replications/${exp_name}/summary_"*.json 2>/dev/null | head -1 || true)
if [ -n "$summary" ] && [ -f "$summary" ]; then
    local_score=$(python3 -c "import json; print(json.load(open('$summary'))['lambda_0_5']['mean'])" 2>/dev/null || echo "null")
    local_std=$(python3 -c "import json; print(json.load(open('$summary'))['lambda_0_5']['std'])" 2>/dev/null || echo "null")
    local_n=$(python3 -c "import json; print(json.load(open('$summary'))['n'])" 2>/dev/null || echo "null")
fi

cat > "$manifest" <<EOF
{
  "team_id": "${TEAM_ID}",
  "version": "${version}",
  "experiment_name": "${exp_name}",
  "built_at": "${build_iso}",
  "image_tag": "${image_tag}",
  "docker_platform": "${DOCKER_PLATFORM}",
  "archive": "${TEAM_ID}_${version}.tar.gz",
  "size_bytes": ${bytes},
  "size_mb": ${mb},
  "size_gb": ${gb},
  "git_sha": "${git_sha}",
  "git_msg": $(printf '%s' "$git_msg" | python3 -c "import json,sys; print(json.dumps(sys.stdin.read().strip()))"),
  "local_score": ${local_score},
  "local_score_std": ${local_std},
  "local_score_n": ${local_n}
}
EOF
echo "==> manifest: $manifest"
cat "$manifest" | sed 's/^/    /'

# 5. Upload to gdrive
if [ "${SKIP_UPLOAD:-0}" = "1" ]; then
    echo "==> SKIP_UPLOAD=1, leaving local artifacts/submissions/ only"
    exit 0
fi

# Pre-flight: confirm rclone gdrive: works
echo "==> rclone health check on gdrive: ..."
rclone lsd gdrive: > /dev/null 2>&1 || {
    echo "FATAL: rclone gdrive: remote not reachable. Check 'rclone config' and re-auth." >&2
    exit 3
}

echo "==> uploading $archive to gdrive:${GDRIVE_PATH}/"
rclone copy "$archive" "gdrive:${GDRIVE_PATH}/" --progress
echo "==> uploading $manifest to gdrive:${GDRIVE_PATH}/"
rclone copy "$manifest" "gdrive:${GDRIVE_PATH}/" --progress

echo ""
echo "==> uploaded files in gdrive:${GDRIVE_PATH}/:"
rclone lsl "gdrive:${GDRIVE_PATH}/" | awk '{print "    "$0}'

echo ""
echo "DONE. Submission ${TEAM_ID}_${version} (exp=${exp_name}, size=${gb} GB)"
echo "  → ${archive}"
echo "  → ${manifest}"
echo "  → uploaded to gdrive:${GDRIVE_PATH}/"
echo ""
echo "Next: get a shareable link and email organizers per submission rules:"
echo "  rclone link gdrive:${GDRIVE_PATH}/${TEAM_ID}_${version}.tar.gz"
