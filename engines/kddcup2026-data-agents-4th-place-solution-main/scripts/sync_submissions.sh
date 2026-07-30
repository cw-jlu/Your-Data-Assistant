#!/usr/bin/env bash
# Two-way sync the local artifacts/submissions/ folder with
# gdrive:KDDCup2026/submission/.
#
# Usage:
#   bash scripts/sync_submissions.sh push   # local -> gdrive (copy missing)
#   bash scripts/sync_submissions.sh pull   # gdrive -> local (copy missing)
#   bash scripts/sync_submissions.sh status # show what differs both ways
#
# Notes:
#   - Uses `rclone copy` (no delete on either side) so it's safe to run anytime.
#   - For destructive sync (= delete dst extras), use `rclone sync` manually.
#   - Local target: artifacts/submissions/ (gitignored)

set -uo pipefail

GDRIVE_PATH=${GDRIVE_PATH:-KDDCup2026/submission}
LOCAL_DIR=artifacts/submissions

direction=${1:-status}

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"
mkdir -p "$LOCAL_DIR"

case "$direction" in
push)
    echo "==> push: $LOCAL_DIR -> gdrive:${GDRIVE_PATH}/"
    rclone copy "$LOCAL_DIR/" "gdrive:${GDRIVE_PATH}/" --progress
    ;;
pull)
    echo "==> pull: gdrive:${GDRIVE_PATH}/ -> $LOCAL_DIR/"
    rclone copy "gdrive:${GDRIVE_PATH}/" "$LOCAL_DIR/" --progress
    ;;
status)
    echo "==> local files ($LOCAL_DIR/):"
    ls -lh "$LOCAL_DIR"/ 2>/dev/null | grep -v '^total' | awk '{print "    "$0}' | head -20
    echo ""
    echo "==> remote files (gdrive:${GDRIVE_PATH}/):"
    rclone lsl "gdrive:${GDRIVE_PATH}/" | awk '{print "    "$0}' | head -20
    echo ""
    echo "==> diff (--missing-on-dst then --missing-on-src):"
    echo "  local has but gdrive doesn't:"
    rclone check "$LOCAL_DIR/" "gdrive:${GDRIVE_PATH}/" --missing-on-dst /dev/stdout 2>/dev/null | sed 's/^/    /'
    echo "  gdrive has but local doesn't:"
    rclone check "$LOCAL_DIR/" "gdrive:${GDRIVE_PATH}/" --missing-on-src /dev/stdout 2>/dev/null | sed 's/^/    /'
    ;;
*)
    echo "usage: $0 {push|pull|status}"
    exit 2
    ;;
esac
