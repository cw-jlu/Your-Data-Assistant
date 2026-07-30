#!/usr/bin/env bash
# Run the Docker container against the local public dataset and score
# its predictions with the mock scorer.
#
# Usage:
#   bash scripts/local_eval.sh <version> [task_set_file]
#   e.g. bash scripts/local_eval.sh v1
#        bash scripts/local_eval.sh v1 data/public/holdout_ids.txt
#
# Requires:
#   - docker
#   - data/public/input/ populated with task_<id>/{task.json,context/}
#   - data/public/output/ populated with task_<id>/gold.csv
#   - a running vLLM/SGLang serving qwen3.5-35b-a3b on $MODEL_API_URL
#     (defaults to http://<host_lan_ip>:8000/v1)
#
# Notes:
#   - This script runs the evaluator in Docker but can call an inference server
#     on this host or another machine in the same network.
#   - Override MODEL_API_URL explicitly when using a remote inference server.

set -euo pipefail

VERSION="${1:-}"
TASK_SET="${2:-}"
if [[ "${VERSION}" == "" ]]; then
    echo "usage: bash scripts/local_eval.sh <version> [task_set_file]" >&2
    exit 1
fi
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

IMAGE="dabench:${VERSION}"
SANDBOX="${ROOT_DIR}/artifacts/sandbox/${VERSION}"
OUT_DIR="${SANDBOX}/output"
LOG_DIR="${SANDBOX}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

if [[ ! -d "${ROOT_DIR}/data/public/input" ]]; then
    echo "ERROR: data/public/input/ not found. Place the Phase 1 dataset there first." >&2
    exit 2
fi

# LAN IP autodetect (Linux only — macOS has no `ip` command). On macOS or
# any host where the LLM lives on a separate machine (DGX, port 8000), the
# user MUST export MODEL_API_URL explicitly. We refuse to silently fall
# back to 127.0.0.1 because that is almost never the right host and would
# burn a holdout run with cryptic connection errors.
if command -v ip >/dev/null 2>&1; then
    HOST_LAN_IP="$(ip route get 1.1.1.1 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i=="src") {print $(i+1); exit}}' || true)"
else
    HOST_LAN_IP=""
fi
if [[ -z "${MODEL_API_URL:-}" && -z "${HOST_LAN_IP}" ]]; then
    cat >&2 <<'__EOF__'
ERROR: MODEL_API_URL is not set and the LAN IP could not be auto-detected.
       Apple Silicon / macOS has no `ip` command; on those hosts the
       single-model policy requires you to point the eval container at the
       remote vLLM explicitly. Example:

         export MODEL_API_URL=http://<VLLM_HOST>:8000/v1
         bash scripts/local_eval.sh v<N> data/public/holdout_ids.txt

       Replace the host with whichever machine serves your qwen3.5-35b-a3b
       vLLM (DGX = <VLLM_HOST> by default for team1438).
__EOF__
    exit 5
fi
MODEL_API_URL="${MODEL_API_URL:-http://${HOST_LAN_IP}:8000/v1}"
MODEL_API_KEY="${MODEL_API_KEY:-EMPTY}"
MODEL_NAME="${MODEL_NAME:-qwen3.5-35b-a3b}"
# DABENCH_LAMBDAS = whitespace-separated λ list for sensitivity sweeps;
# falls back to single DABENCH_LAMBDA, then 0.10. Phase 3 column-ablation
# wants {0.05, 0.10, 0.20}; that's the sweep we use here by default.
DABENCH_LAMBDA="${DABENCH_LAMBDA:-0.10}"
DABENCH_LAMBDAS="${DABENCH_LAMBDAS:-${DABENCH_LAMBDA}}"

echo ">> running ${IMAGE} ..."
echo "   MODEL_API_URL=${MODEL_API_URL}"
echo "   MODEL_NAME=${MODEL_NAME}"
echo "   output → ${OUT_DIR}"

if ! curl -fsS "${MODEL_API_URL}/models" >/dev/null 2>&1; then
    echo "ERROR: model server not reachable at ${MODEL_API_URL}/models" >&2
    echo "       set MODEL_API_URL to a reachable inference server URL." >&2
    exit 4
fi

DOCKER_EXTRA_ARGS=()
if [[ "${TASK_SET}" != "" ]]; then
    if [[ ! -f "${TASK_SET}" ]]; then
        echo "ERROR: task set file ${TASK_SET} not found." >&2
        exit 3
    fi
    cp "${TASK_SET}" "${SANDBOX}/task_filter.txt"
    DOCKER_EXTRA_ARGS+=(-v "${SANDBOX}/task_filter.txt:/tmp/task_filter.txt:ro")
    DOCKER_EXTRA_ARGS+=(--entrypoint "uv")
    EVAL_CMD=("run" "dabench" "run-benchmark" "--config" "configs/eval.yaml" "--task-set" "/tmp/task_filter.txt")
else
    EVAL_CMD=()
fi

# --platform linux/amd64 + --cpus=16 + --memory=64g mirror the eval
# runtime envelope exactly (rules §compute §1). On Apple Silicon dev
# hosts the platform triggers QEMU emulation (slow but correct); on
# x86 dev hosts it is a no-op. The build script already gates the
# image arch, so this is belt-and-suspenders.
#
# Why we don't replicate `--network=eval_net`:
#   - eval_net is an organizer-defined isolation boundary we cannot
#     fabricate locally. The default docker bridge is sufficient for
#     dev runs because (a) UV_OFFLINE=1 baked into the image kills
#     the only known runtime PyPI lookup, and (b) our src/ has zero
#     external HTTP calls (verified by grep). Network behavior of the
#     image is therefore identical between dev and eval — only the
#     enforcement perimeter differs.
docker run --rm \
    --platform linux/amd64 \
    --cpus=16 --memory=64g \
    --add-host host.docker.internal:host-gateway \
    -e MODEL_API_URL="${MODEL_API_URL}" \
    -e MODEL_API_KEY="${MODEL_API_KEY}" \
    -e MODEL_NAME="${MODEL_NAME}" \
    -v "${ROOT_DIR}/data/public/input:/input:ro" \
    -v "${OUT_DIR}:/output" \
    -v "${LOG_DIR}:/logs" \
    "${DOCKER_EXTRA_ARGS[@]}" \
    "${IMAGE}" \
    "${EVAL_CMD[@]}"

echo ""
echo ">> scoring with mock_scorer (λ=${DABENCH_LAMBDAS}) ..."
TASK_FILTER_ARG=()
if [[ "${TASK_SET}" != "" ]]; then
    TASK_FILTER_ARG=(--task-filter "${TASK_SET}")
fi
SCORE_JSON="${SANDBOX}/score.json"
SCORE_MD="${SANDBOX}/score.md"
# shellcheck disable=SC2206
LAMBDA_ARGS=( ${DABENCH_LAMBDAS} )

uv run python -m data_agent_baseline.scoring.mock_scorer \
    --predictions "${OUT_DIR}" \
    --gold "${ROOT_DIR}/data/public/output" \
    --input "${ROOT_DIR}/data/public/input" \
    --lambda-values "${LAMBDA_ARGS[@]}" \
    --json \
    "${TASK_FILTER_ARG[@]}" \
    > "${SCORE_JSON}"

GIT_COMMIT="$(git -C "${ROOT_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"

uv run python "${ROOT_DIR}/scripts/render_score_report.py" \
    --json-input "${SCORE_JSON}" \
    --md-output "${SCORE_MD}" \
    --version "${VERSION}" \
    --commit "${GIT_COMMIT}" \
    --model-api-url "${MODEL_API_URL}" \
    --model-name "${MODEL_NAME}" \
    --predictions-root "${OUT_DIR}" \
    --print-text

echo ""
echo ">> reports:"
echo "   ${SCORE_JSON}"
echo "   ${SCORE_MD}"
