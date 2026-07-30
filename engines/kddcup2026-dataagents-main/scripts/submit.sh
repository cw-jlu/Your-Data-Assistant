#!/usr/bin/env bash
# Compose the §3.3 submission email body. Drafts only — the actual send is
# manual (paste subject + body into your mail client and send to
# kddcup@hkust-gz.edu.cn).
#
# Usage:
#   bash scripts/submit.sh <team_id> <version> '<gdrive_link>'

set -euo pipefail

TEAM_ID="${1:?team_id required (e.g., team0042)}"
VERSION="${2:?version required (e.g., 1)}"
GDRIVE_LINK="${3:?gdrive link required}"

[[ "$TEAM_ID" =~ ^team[0-9]+$ ]]               || { echo "FAIL: team_id must look like team0042 (got: $TEAM_ID)"; exit 2; }
[[ "$VERSION" =~ ^[1-9][0-9]*$ ]]              || { echo "FAIL: version must be a positive integer (got: $VERSION)"; exit 2; }
[[ "$GDRIVE_LINK" =~ ^https://drive\.google\.com/ ]] || { echo "FAIL: link must be a https://drive.google.com/... URL"; exit 2; }

ARCHIVE="artifacts/submissions/${TEAM_ID}_v${VERSION}.tar.gz"
[[ -f "$ARCHIVE" ]] || { echo "FAIL: $ARCHIVE not found — run pre_submit_check.sh first"; exit 2; }

# §3.3 subject template is verbatim — the judge parses it.
SUBJECT="[KDDCup2026 Data Agents] Submission - ${TEAM_ID} - v${VERSION}"
BODY_FILE="artifacts/submissions/${TEAM_ID}_v${VERSION}_email.txt"

# sha256sum is GNU coreutils; macOS only ships shasum. Pick whichever exists.
if command -v sha256sum >/dev/null 2>&1; then
    SHA256=$(sha256sum "$ARCHIVE" | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
    SHA256=$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')
else
    echo "FAIL: neither sha256sum nor shasum available" >&2
    exit 2
fi

CHANGELOG_BLOCK=""
if [[ -f CHANGELOG.md ]]; then
    CHANGELOG_BLOCK="$(head -n 30 CHANGELOG.md)"
fi

cat > "$BODY_FILE" <<EOF
Team ID: ${TEAM_ID}
Version: v${VERSION}
Submission date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Sharing link: ${GDRIVE_LINK}

Archive: ${TEAM_ID}_v${VERSION}.tar.gz ($(wc -c < "$ARCHIVE" | awk '{printf "%.2f MB", $1/1024/1024}'))
SHA256: ${SHA256}

Change summary:
${CHANGELOG_BLOCK}
EOF

echo "=== Email draft ready ==="
echo "To:      kddcup@hkust-gz.edu.cn"
echo "Subject: $SUBJECT"
echo "Body:    $BODY_FILE"
echo ""
echo "Pre-flight checklist (§3.3):"
echo "  [ ] Drive sharing set to 'Anyone with the link' → 'Viewer'"
echo "  [ ] Subject line copied verbatim (judge parses it)"
echo "  [ ] Drive filename matches: ${TEAM_ID}_v${VERSION}.tar.gz"
echo "  [ ] Drive file kept until evaluation completion notice (§3.3)"
echo ""
echo "Manual steps:"
echo "  1. Open mail client; paste subject + body"
echo "  2. Send; cc yourself for the audit trail"
