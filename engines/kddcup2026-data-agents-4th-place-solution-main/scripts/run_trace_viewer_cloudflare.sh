#!/usr/bin/env bash
# Run the local trace viewer and expose it through an existing Cloudflare Tunnel.
#
# Required for named/local tunnel:
#   TRACE_TUNNEL_NAME=kobushi-trace-viewer
#
# Or for dashboard-managed tunnel:
#   TRACE_TUNNEL_TOKEN=<token copied from Cloudflare Zero Trust>
#
# Optional:
#   TRACE_VIEWER_PORT=8765
#   TRACE_VIEWER_REQUIRE_CF_ACCESS=1
#   TRACE_VIEWER_AUTH_TOKEN=<shared temporary token>
#   TRACE_VIEWER_AUTH_TOKEN_TTL_HOURS=12
#   TRACE_ACCESS_EMAIL_DOMAIN=example.com
#   TRACE_ACCESS_EMAIL=user@example.com
#   TRACE_CLOUDFLARED_CONFIG=$HOME/.cloudflared/config.yml
#
# Quick tunnel, not Access-protected and not recommended for trace data:
#   TRACE_QUICK_TUNNEL=1 bash scripts/run_trace_viewer_cloudflare.sh

set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"

TRACE_VIEWER_HOST=${TRACE_VIEWER_HOST:-127.0.0.1}
TRACE_VIEWER_PORT=${TRACE_VIEWER_PORT:-8765}

if [ "${TRACE_QUICK_TUNNEL:-0}" = "1" ] && [ -z "${TRACE_VIEWER_AUTH_TOKEN:-}" ]; then
  TRACE_VIEWER_AUTH_TOKEN=$(python - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
)
  export TRACE_VIEWER_AUTH_TOKEN
fi
if [ "${TRACE_QUICK_TUNNEL:-0}" = "1" ] && [ -z "${TRACE_VIEWER_AUTH_TOKEN_TTL_HOURS:-}" ]; then
  TRACE_VIEWER_AUTH_TOKEN_TTL_HOURS=12
  export TRACE_VIEWER_AUTH_TOKEN_TTL_HOURS
fi

if ! command -v cloudflared >/dev/null 2>&1; then
  cat >&2 <<'EOF'
FATAL: cloudflared is not installed.

Ubuntu 22.04:
  sudo mkdir -p --mode=0755 /usr/share/keyrings
  curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg |
    sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
  echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared jammy main' |
    sudo tee /etc/apt/sources.list.d/cloudflared.list
  sudo apt-get update && sudo apt-get install cloudflared
EOF
  exit 127
fi

viewer_args=(scripts/trace_viewer.py --host "$TRACE_VIEWER_HOST" --port "$TRACE_VIEWER_PORT")
if [ -n "${TRACE_VIEWER_AUTH_TOKEN:-}" ]; then
  viewer_args+=(--auth-token "$TRACE_VIEWER_AUTH_TOKEN")
fi
if [ -n "${TRACE_VIEWER_AUTH_TOKEN_TTL_HOURS:-}" ]; then
  viewer_args+=(--auth-token-ttl-hours "$TRACE_VIEWER_AUTH_TOKEN_TTL_HOURS")
fi
if [ "${TRACE_VIEWER_REQUIRE_CF_ACCESS:-0}" = "1" ]; then
  viewer_args+=(--require-cf-access)
  if [ -n "${TRACE_ACCESS_EMAIL:-}" ]; then
    IFS=',' read -r -a _emails <<< "$TRACE_ACCESS_EMAIL"
    for email in "${_emails[@]}"; do
      viewer_args+=(--access-email "$email")
    done
  fi
  if [ -n "${TRACE_ACCESS_EMAIL_DOMAIN:-}" ]; then
    IFS=',' read -r -a _domains <<< "$TRACE_ACCESS_EMAIL_DOMAIN"
    for domain in "${_domains[@]}"; do
      viewer_args+=(--access-email-domain "$domain")
    done
  fi
fi

viewer_up() {
  python - "$TRACE_VIEWER_PORT" <<'PY'
import sys
from urllib.request import urlopen
try:
    with urlopen(f"http://127.0.0.1:{sys.argv[1]}/healthz", timeout=2) as r:
        raise SystemExit(0 if r.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

viewer_auth_matches() {
  if [ -z "${TRACE_VIEWER_AUTH_TOKEN:-}" ]; then
    return 0
  fi
  python - "$TRACE_VIEWER_PORT" "$TRACE_VIEWER_AUTH_TOKEN" <<'PY'
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

port, token = sys.argv[1], sys.argv[2]
base = f"http://127.0.0.1:{port}"
try:
    urlopen(base + "/api/runs", timeout=2)
    raise SystemExit(1)  # token should be required
except HTTPError as exc:
    if exc.code != 401:
        raise SystemExit(1)
except Exception:
    raise SystemExit(1)

try:
    req = Request(base + "/api/runs", headers={"X-Trace-Viewer-Token": token})
    with urlopen(req, timeout=2) as res:
        raise SystemExit(0 if res.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

viewer_pid=""
if viewer_up; then
  if ! viewer_auth_matches; then
    cat >&2 <<EOF
FATAL: a trace viewer is already running on http://${TRACE_VIEWER_HOST}:${TRACE_VIEWER_PORT},
but it does not enforce the requested TRACE_VIEWER_AUTH_TOKEN.
Stop the existing viewer first, then rerun this script.
EOF
    exit 3
  fi
  echo "Trace viewer already reachable on http://${TRACE_VIEWER_HOST}:${TRACE_VIEWER_PORT}"
else
  echo "Starting trace viewer on http://${TRACE_VIEWER_HOST}:${TRACE_VIEWER_PORT}"
  uv run python "${viewer_args[@]}" &
  viewer_pid=$!
  for _ in $(seq 1 30); do
    if viewer_up; then
      break
    fi
    sleep 1
  done
  if ! viewer_up; then
    echo "FATAL: trace viewer did not become ready" >&2
    if [ -n "$viewer_pid" ]; then kill "$viewer_pid" 2>/dev/null || true; fi
    exit 1
  fi
fi

cleanup() {
  if [ -n "$viewer_pid" ]; then
    kill "$viewer_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

origin="http://${TRACE_VIEWER_HOST}:${TRACE_VIEWER_PORT}"
if [ "${TRACE_QUICK_TUNNEL:-0}" = "1" ]; then
  echo "WARNING: starting a quick tunnel without Cloudflare Access policy."
  if [ -n "${TRACE_VIEWER_AUTH_TOKEN:-}" ]; then
    echo "Trace viewer token is enabled."
    echo "Trace viewer token TTL hours: ${TRACE_VIEWER_AUTH_TOKEN_TTL_HOURS:-0}"
    echo "Append this query string to the trycloudflare URL:"
    echo "  ?token=${TRACE_VIEWER_AUTH_TOKEN}"
  fi
  cloudflared tunnel --url "$origin"
  exit $?
fi

if [ -n "${TRACE_TUNNEL_TOKEN:-}" ]; then
  echo "Running dashboard-managed Cloudflare Tunnel from token"
  cloudflared tunnel run --token "$TRACE_TUNNEL_TOKEN"
  exit $?
fi

config=${TRACE_CLOUDFLARED_CONFIG:-$HOME/.cloudflared/config.yml}
if [ -n "${TRACE_TUNNEL_NAME:-}" ]; then
  if [ -f "$config" ]; then
    echo "Running named Cloudflare Tunnel '${TRACE_TUNNEL_NAME}' with config ${config}"
    cloudflared tunnel --config "$config" run "$TRACE_TUNNEL_NAME"
    exit $?
  fi
  echo "Running named Cloudflare Tunnel '${TRACE_TUNNEL_NAME}'"
  cloudflared tunnel run "$TRACE_TUNNEL_NAME"
  exit $?
fi

cat >&2 <<EOF
FATAL: no tunnel selector provided.

Set one of:
  TRACE_TUNNEL_NAME=kobushi-trace-viewer
  TRACE_TUNNEL_TOKEN=<dashboard token>
  TRACE_QUICK_TUNNEL=1  # temporary only; no Access policy
EOF
exit 2
