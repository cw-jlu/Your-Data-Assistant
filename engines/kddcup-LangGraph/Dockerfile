FROM python:3.11-slim

# 设置 uv 的默认下载源为清华大学镜像站
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple


# 已将默认镜像站修改为官方源 https://huggingface.co
ARG HF_ENDPOINT=https://huggingface.co
ARG VERIFY_QWEN_TOKENIZER_CACHE=1
ARG PRELOAD_FASTER_WHISPER_MODEL=base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_PROGRESS=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_ENDPOINT=${HF_ENDPOINT} \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DABENCH_SUBMISSION_CONFIG=configs/docker.yaml \
    DABENCH_LOG_ROOT=/logs \
    DABENCH_RETRY_SLEEP_SECONDS=60

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs/docker.yaml ./configs/docker.yaml
COPY assets/huggingface/ ./assets/huggingface/

RUN mkdir -p /input /output /logs
RUN uv sync --frozen --no-dev

RUN if [ -n "${PRELOAD_FASTER_WHISPER_MODEL}" ]; then \
      HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 .venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('${PRELOAD_FASTER_WHISPER_MODEL}', device='cpu', compute_type='int8'); print('faster-whisper model cached: ${PRELOAD_FASTER_WHISPER_MODEL}')" || \
      (echo 'Failed to cache faster-whisper model. Build with --build-arg PRELOAD_FASTER_WHISPER_MODEL= to skip preloading.' && exit 1); \
    else \
      echo "Skipping faster-whisper model preload"; \
    fi

RUN if [ "${VERIFY_QWEN_TOKENIZER_CACHE}" = "1" ]; then \
      .venv/bin/python -c "from data_agent_baseline.token_utils import count_tokens; print('Qwen tokenizer cache verified', count_tokens('hello world'))" || \
      (echo 'Missing Qwen tokenizer files. Make sure assets/huggingface/Qwen3.5-35B-A3B/ is present before building, or build with --build-arg VERIFY_QWEN_TOKENIZER_CACHE=0.' && exit 1); \
    else \
      echo "Skipping Qwen tokenizer cache verification"; \
    fi

ENTRYPOINT ["/bin/bash", "-lc", "set -o pipefail; mkdir -p /output \"${DABENCH_LOG_ROOT}\"; i=1; status=0; while [ \"$i\" -le 2 ]; do .venv/bin/dabench run-benchmark --skip-completed --config \"${DABENCH_SUBMISSION_CONFIG}\" 2>&1 | tee -a \"${DABENCH_LOG_ROOT}/runtime.log\"; status=$?; echo \"[entrypoint] run ${i}/2 exited status=${status}\" | tee -a \"${DABENCH_LOG_ROOT}/runtime.log\"; i=$((i+1)); if [ \"$i\" -le 2 ]; then echo \"[entrypoint] sleeping ${DABENCH_RETRY_SLEEP_SECONDS}s before next run...\" | tee -a \"${DABENCH_LOG_ROOT}/runtime.log\"; sleep \"${DABENCH_RETRY_SLEEP_SECONDS}\"; fi; done; exit \"$status\""]