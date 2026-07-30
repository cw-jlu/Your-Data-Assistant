#!/usr/bin/env bash
# Build, tag, save, and weigh a submission image.
# Usage: bash submission/build.sh <team_id> <version>
#   e.g. bash submission/build.sh kobushi v1
set -euo pipefail

team_id=${1:?team_id is required, e.g. kobushi}
version=${2:?version is required, e.g. v1}

repo_root=$(cd "$(dirname "$0")/.." && pwd)
image_tag="${team_id}:${version}"
archive_name="${team_id}_${version}.tar.gz"
platform=${DOCKER_PLATFORM:-linux/amd64}

cd "$repo_root"
echo ">> docker build --platform $platform $image_tag (this may take ~3-5 min)"
docker build --platform "$platform" -t "$image_tag" -f submission/Dockerfile .

echo ">> docker save $image_tag -> $archive_name"
docker save "$image_tag" | gzip > "$archive_name"

bytes=$(stat -c%s "$archive_name" 2>/dev/null || stat -f%z "$archive_name")
gb=$(awk "BEGIN{printf \"%.2f\", $bytes / 1024 / 1024 / 1024}")
echo ">> archive: $archive_name = ${gb} GB"
echo ">> upload to Google Drive (anyone with link), then email the link to the organizers."
echo ">> subject: [KDDCup2026 Data Agents] Submission - ${team_id} - ${version}"
