#!/usr/bin/env bash
# Serve Qwen via vLLM Docker Compose (ARM64/GB10-friendly path).
#
# Usage:
#   bash scripts/serve_qwen_docker.sh              # launch + probe
#   bash scripts/serve_qwen_docker.sh --no-probe   # launch only
#   bash scripts/serve_qwen_docker.sh --probe-only # probe running endpoint
#   bash scripts/serve_qwen_docker.sh --stop       # stop container
#   bash scripts/serve_qwen_docker.sh --status     # show container/endpoint status
#   bash scripts/serve_qwen_docker.sh --logs       # follow container logs
#   bash scripts/serve_qwen_docker.sh --build-image # build local vLLM image
#
# Important:
# - Runs vLLM inside Docker Compose (no project venv dependency for serving).
# - Default attention backend is TRITON_ATTN to avoid FlashAttention mismatches
#   seen on Blackwell-family GPUs.

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Competition-aligned naming:
# - MODEL_NAME: OpenAI API model id seen by clients (eval injects this)
# - MODEL_REPO: HuggingFace model repo path used by vLLM local serving
MODEL_NAME="${MODEL_NAME:-qwen3.5-35b-a3b}"
# NOTE: `Qwen/Qwen3.5-35B-A3B-Instruct` is not a valid HF repo id (404).
# Use the official public repo id as the default.
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.5-35B-A3B}"
MODEL_ALIAS="${MODEL_ALIAS:-${MODEL_NAME}}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
# DGX Spark tuned defaults:
# - bfloat16 for stable quality/perf on Blackwell
# - memory capped to 85% to avoid unified-memory OOM
# - bounded scheduler batch sizes to keep tail-latency predictable
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
DTYPE="${DTYPE:-bfloat16}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
QUANTIZATION="${QUANTIZATION:-}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-1200}"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# Fallback: use token saved by `huggingface-cli login` if env var is absent.
if [[ -z "${HF_TOKEN}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
fi
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/vllm_logs}"
# vLLM recipe guidance for Blackwell recommends cu130-nightly based images.
# We build and use a local image by default for reproducibility.
DOCKER_IMAGE="${DOCKER_IMAGE:-qwen35-vllm:cu130-nightly}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-${ROOT_DIR}/docker/vllm-qwen35/Dockerfile}"
BUILD_CONTEXT="${BUILD_CONTEXT:-${ROOT_DIR}}"
COMPOSE_FILE="${COMPOSE_FILE:-${ROOT_DIR}/docker/vllm-qwen35/docker-compose.yml}"
AUTO_BUILD_IMAGE="${AUTO_BUILD_IMAGE:-1}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen-vllm}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"
SEED="${SEED:-1024}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"

ENDPOINT="http://127.0.0.1:${PORT}/v1"
LAN_HOST_IP="${LAN_HOST_IP:-$(ip route get 1.1.1.1 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i=="src") {print $(i+1); exit}}' || true)}"
LAN_ENDPOINT="http://${LAN_HOST_IP:-127.0.0.1}:${PORT}/v1"
PROBE_REPORT="${ROOT_DIR}/docs/qwen_endpoint_capabilities.md"

mkdir -p "${HF_HOME}" "${LOG_DIR}"

log() { printf '\033[1;36m[serve_qwen_docker]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[serve_qwen_docker]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[serve_qwen_docker]\033[0m %s\n' "$*" >&2; exit 1; }

compose() {
  docker compose -f "${COMPOSE_FILE}" "$@"
}

compose_service_exists() {
  local cid
  cid="$(compose ps -q vllm 2>/dev/null || true)"
  [[ -n "${cid}" ]]
}

container_running() {
  local cid
  cid="$(compose ps -q vllm 2>/dev/null || true)"
  if [[ -z "${cid}" ]]; then
    return 1
  fi
  [[ "$(docker inspect -f '{{.State.Running}}' "${cid}" 2>/dev/null || true)" == "true" ]]
}

stop_server() {
  if ! compose_service_exists; then
    log "no compose service state found for ${CONTAINER_NAME}"
    return 0
  fi
  log "stopping compose service ${CONTAINER_NAME}"
  compose down --remove-orphans >/dev/null || true
}

show_status() {
  printf 'endpoint:      %s\n' "${ENDPOINT}"
  printf 'lan endpoint:  %s\n' "${LAN_ENDPOINT}"
  printf 'model alias:   %s  (repo: %s)\n' "${MODEL_ALIAS}" "${MODEL_REPO}"
  printf 'docker image:  %s\n' "${DOCKER_IMAGE}"
  printf 'container:     %s\n' "${CONTAINER_NAME}"
  printf 'tuning:        gpu_mem_util=%s max_len=%s max_num_seqs=%s max_num_batched_tokens=%s dtype=%s backend=%s\n' \
    "${GPU_MEM_UTIL}" "${MAX_MODEL_LEN}" "${MAX_NUM_SEQS}" "${MAX_NUM_BATCHED_TOKENS}" "${DTYPE}" "${ATTENTION_BACKEND}"
  if container_running; then
    printf 'state:         running\n'
    if curl -fsS "${ENDPOINT}/models" >/dev/null 2>&1; then
      printf 'health:        OK (/v1/models responds)\n'
    else
      printf 'health:        container up, endpoint not ready\n'
    fi
  elif compose_service_exists; then
    printf 'state:         stopped\n'
  else
    printf 'state:         not created\n'
  fi
  printf 'hf cache:      %s\n' "${HF_HOME}"
  if [[ -n "${HF_TOKEN}" ]]; then
    printf 'hf token:      set\n'
  else
    printf 'hf token:      not set\n'
  fi
  printf 'hf xet:        %s\n' "${HF_HUB_DISABLE_XET}"
  printf 'compose file:  %s\n' "${COMPOSE_FILE}"
  printf 'logs:          docker compose -f %s logs -f vllm\n' "${COMPOSE_FILE}"
}

show_logs() {
  if ! compose_service_exists; then
    fail "compose service ${CONTAINER_NAME} does not exist"
  fi
  log "following compose logs for ${CONTAINER_NAME}"
  compose logs -f vllm
}

image_exists() {
  docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1
}

build_image() {
  if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
    fail "dockerfile not found: ${DOCKERFILE_PATH}"
  fi
  if [[ ! -f "${COMPOSE_FILE}" ]]; then
    fail "compose file not found: ${COMPOSE_FILE}"
  fi
  log "building image ${DOCKER_IMAGE}"
  log "compose=${COMPOSE_FILE}"
  DOCKER_IMAGE="${DOCKER_IMAGE}" \
  DOCKERFILE_PATH="${DOCKERFILE_PATH}" \
  BUILD_CONTEXT="${BUILD_CONTEXT}" \
  HF_HOME="${HF_HOME}" \
  compose build vllm
}

launch_server() {
  if container_running; then
    log "container already running (${CONTAINER_NAME}) — skipping launch"
    return 0
  fi

  if [[ ! -f "${COMPOSE_FILE}" ]]; then
    fail "compose file not found: ${COMPOSE_FILE}"
  fi

  if compose_service_exists; then
    compose down --remove-orphans >/dev/null || true
  fi

  local quant_args=()
  if [[ -n "${QUANTIZATION}" ]]; then
    quant_args+=(--quantization "${QUANTIZATION}")
  fi
  local sched_args=()
  if [[ -n "${MAX_NUM_SEQS}" ]]; then
    sched_args+=(--max-num-seqs "${MAX_NUM_SEQS}")
  fi
  if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
    sched_args+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
  fi
  local model_mode_args=()
  if [[ "${LANGUAGE_MODEL_ONLY}" == "1" ]]; then
    model_mode_args+=(--language-model-only)
  fi

  local qwen_reasoning_args=()
  if [[ -n "${REASONING_PARSER}" ]]; then
    qwen_reasoning_args+=(--reasoning-parser "${REASONING_PARSER}")
  fi
  if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
    qwen_reasoning_args+=(--enable-auto-tool-choice)
  fi
  if [[ -n "${TOOL_CALL_PARSER}" ]]; then
    qwen_reasoning_args+=(--tool-call-parser "${TOOL_CALL_PARSER}")
  fi

  local optional_args=(
    "${model_mode_args[@]}"
    "${quant_args[@]}"
    "${sched_args[@]}"
    "${qwen_reasoning_args[@]}"
  )
  local vllm_optional_args=""
  if (( ${#optional_args[@]} > 0 )); then
    vllm_optional_args="$(printf '%q ' "${optional_args[@]}")"
    vllm_optional_args="${vllm_optional_args% }"
  fi

  if ! image_exists; then
    if [[ "${AUTO_BUILD_IMAGE}" == "1" ]]; then
      warn "image not found locally: ${DOCKER_IMAGE} (auto building)"
      build_image
    else
      fail "docker image not found: ${DOCKER_IMAGE} (run with --build-image or set AUTO_BUILD_IMAGE=1)"
    fi
  fi

  log "starting docker compose service: ${CONTAINER_NAME}"
  log "image=${DOCKER_IMAGE} model=${MODEL_REPO} alias=${MODEL_ALIAS} attention=${ATTENTION_BACKEND} gpu_mem_util=${GPU_MEM_UTIL} max_len=${MAX_MODEL_LEN} max_num_seqs=${MAX_NUM_SEQS}"

  DOCKER_IMAGE="${DOCKER_IMAGE}" \
  DOCKERFILE_PATH="${DOCKERFILE_PATH}" \
  BUILD_CONTEXT="${BUILD_CONTEXT}" \
  CONTAINER_NAME="${CONTAINER_NAME}" \
  MODEL_REPO="${MODEL_REPO}" \
  MODEL_ALIAS="${MODEL_ALIAS}" \
  HOST="${HOST}" \
  PORT="${PORT}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  SEED="${SEED}" \
  GPU_MEM_UTIL="${GPU_MEM_UTIL}" \
  TENSOR_PARALLEL="${TENSOR_PARALLEL}" \
  DTYPE="${DTYPE}" \
  ATTENTION_BACKEND="${ATTENTION_BACKEND}" \
  HF_HOME="${HF_HOME}" \
  HF_TOKEN="${HF_TOKEN}" \
  HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET}" \
  VLLM_OPTIONAL_ARGS="${vllm_optional_args}" \
  EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS}" \
  compose up -d vllm >/dev/null
}

wait_until_ready() {
  log "waiting for ${ENDPOINT}/models (up to ${WAIT_TIMEOUT_S}s)"
  local start; start=$(date +%s)
  while true; do
    if curl -fsS "${ENDPOINT}/models" >/dev/null 2>&1; then
      log "vllm is ready"
      return 0
    fi
    if ! container_running; then
      warn "container exited before ready. Last logs:"
      compose logs --tail 120 vllm >&2 || true
      fail "vllm compose service crashed"
    fi
    local elapsed; elapsed=$(( $(date +%s) - start ))
    if (( elapsed > WAIT_TIMEOUT_S )); then
      warn "timeout waiting for endpoint. Last logs:"
      compose logs --tail 120 vllm >&2 || true
      fail "ready-wait timeout exceeded ${WAIT_TIMEOUT_S}s"
    fi
    sleep 5
  done
}

run_probe() {
  if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 not found. install python3 to run probe_qwen.py"
  fi
  log "running probe against ${ENDPOINT}"
  python3 "${ROOT_DIR}/scripts/probe_qwen.py" \
    --base-url "${ENDPOINT}" \
    --api-key "local" \
    --model "${MODEL_ALIAS}" \
    --report "${PROBE_REPORT}"
  log "probe report: ${PROBE_REPORT}"
}

ACTION="start"
RUN_PROBE=1
case "${1:-}" in
  ""|start)        ACTION="start";;
  --no-probe)      ACTION="start"; RUN_PROBE=0;;
  --probe-only)    ACTION="probe";;
  --stop|stop)     ACTION="stop";;
  --status|status) ACTION="status";;
  --logs|logs)     ACTION="logs";;
  --build-image|build-image) ACTION="build";;
  -h|--help)
    sed -n '1,60p' "$0"
    exit 0
    ;;
  *) fail "unknown arg: $1 (try --help)";;
esac

case "${ACTION}" in
  start)
    launch_server
    wait_until_ready
    if (( RUN_PROBE == 1 )); then run_probe; fi
    show_status
    ;;
  probe)
    if ! curl -fsS "${ENDPOINT}/models" >/dev/null 2>&1; then
      fail "no live server on ${ENDPOINT} — run start first"
    fi
    run_probe
    ;;
  stop)   stop_server;;
  status) show_status;;
  logs)   show_logs;;
  build)  build_image;;
esac
