# DataAgent-Bench eval container.
#
# Build with: bash scripts/build_submission.sh v<N>
#   (uses docker buildx with --platform linux/amd64 explicitly so dev
#    machines on Apple Silicon still produce a runtime-correct image)
#
# The eval runtime mounts:
#   /input  read-only    — task data
#   /output read-write   — prediction.csv per task
#   /logs   read-write   — runtime.log
# and injects MODEL_API_URL / MODEL_API_KEY / MODEL_NAME env vars.
#
# We pin a slim Python base + uv to keep the image small. Target ≤ 9 GB
# (1 GB headroom under the 10 GB cap).
#
# Platform pin — rules-compute mandates x86-64. Apple Silicon dev hosts
# emulate via QEMU; production runs natively. Hard-pinning here keeps
# Docker Desktop on M-series from accidentally publishing arm64 images.
FROM --platform=linux/amd64 python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python

# v8: BLAS thread cap. Organizer spec is 16 vCPU; with max_workers=8 in
# eval.yaml, each worker gets exactly 2 BLAS threads (8×2 = 16). Without
# these caps pandas/numpy auto-detect all 16 cores and one worker's
# read_csv / groupby can monopolize the box, starving the other 7. Caps
# apply transitively to subprocess.run() spawned children (execute_python).
ENV OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2

WORKDIR /app

# System deps. Keep minimal — every byte counts.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv.
RUN pip install --no-cache-dir uv==0.5.14

# Copy lock + project metadata first to maximize layer caching.
# README.md is referenced as `readme = "README.md"` in pyproject.toml,
# so hatchling needs it on the project install step.
COPY pyproject.toml uv.lock README.md ./

# Install dependencies only — the project itself is installed after src/ lands.
# Splitting the syncs preserves the deps cache layer when only project source
# changes between builds.
RUN uv sync --frozen --no-dev --no-install-project

# Project source.
COPY src ./src
COPY configs/eval.yaml ./configs/eval.yaml

# Now install the project (data-agent-baseline) on top of the cached deps.
RUN uv sync --frozen --no-dev

# Runtime offline guard. `uv run` re-validates `build-system.requires`
# (PEP-517, e.g. hatchling) on every invocation and attempts a TCP/DNS
# lookup against pypi.org for that purpose. Rules-runtime §4 blocks all
# outbound traffic except MODEL_API_URL, so an unsuccessful PyPI lookup
# would either fail the ENTRYPOINT (`uv run` exits) or — under stricter
# rule interpretations — count as a prohibited external-internet
# attempt. Pinning UV_OFFLINE=1 short-circuits the lookup entirely:
# every dependency is already installed in the previous layers, so
# offline resolution is always sufficient. Setting it AFTER the sync
# steps keeps build-time deps install fully online.
ENV UV_OFFLINE=1

# Smoke-test that the CLI loads — also implicitly verifies that
# UV_OFFLINE=1 does not break runtime invocation.
RUN uv run dabench --help > /dev/null

# Arch sanity (build-time): refuse to finalize the image if `uname -m`
# disagrees with the platform pin. This catches a misconfigured buildx
# builder before the tarball ever ships.
RUN test "$(uname -m)" = "x86_64" || ( \
        echo "ERROR: image arch is $(uname -m), expected x86_64." >&2; \
        echo "       Eval runtime is x86-64 only (rules §compute)." >&2; \
        exit 4 \
    )

# Runtime invocation. The eval driver injects env vars and mounts.
ENTRYPOINT ["uv", "run", "dabench", "run-benchmark", "--config", "configs/eval.yaml"]
