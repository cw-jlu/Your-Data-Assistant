# syntax=docker/dockerfile:1.6
#
# Submission image per §3.2 / §3.4. Two-stage build: builder runs `uv export`
# and installs Python deps into the system site-packages, final image only
# carries runtime artefacts (no compilers / no uv).
#
# Python version: keep both FROM lines and the site-packages COPY path on the
# same minor version, aligned with the dev interpreter (3.14) — rehearsal and
# judged run must share runtime semantics. multiprocessing is pinned to spawn
# in code (runs/subprocess.py, tools/python_exec.py), so safety does not hinge
# on this version, but stdlib behaviour should still not drift.
#
# Opt-in CN mirror (uv reads UV_INDEX_URL only — not pip.conf):
#   docker build \
#       --build-arg UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
#       -t <team_id>:v<N> -f docker/Dockerfile .

FROM --platform=linux/amd64 python:3.14-slim AS builder

ARG UV_INDEX_URL=""
ENV UV_INDEX_URL=${UV_INDEX_URL}
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# All listed deps (numpy/pandas/pyarrow/polars/pydantic/...) ship
# manylinux wheels for x86_64 on this Python version, so no compilers needed.
# If a future dep requires C extensions, re-add: build-essential gcc g++.
RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml uv.lock ./
# --no-emit-project: skip the `-e .` self-install line. The final stage uses
#   PYTHONPATH=/app instead of an installed package, so a self-install would
#   force us to also COPY README.md + src/ into the deps layer — wasteful and
#   it'd bust the deps cache on every code change.
# --no-hashes: trades integrity for build speed; acceptable since the .tar.gz
#   we ship carries an sha256 in the email body anyway (see scripts/submit.sh).
RUN uv export --frozen --no-dev --no-emit-project --no-hashes -o requirements.txt \
 && uv pip install --system --no-cache -r requirements.txt

FROM --platform=linux/amd64 python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Parsing deps are self-contained manylinux wheels (pymupdf bundles MuPDF;
# pyarrow/pandas ship their own native libs), so no OCR / rasterisation apt
# stack (tesseract, poppler-utils, libgl1) is needed on top of the slim base.

COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
# Configs first (rarely change), then code (changes most). .dockerignore
# excludes configs/*.local.yaml so dev secrets never leak into the image.
COPY configs ./configs
COPY docs/assets/mamba-agent.avif ./mamba-agent.avif
COPY src/agents ./agents
RUN <<'PYEOF' python
import json
from pathlib import Path

path = Path("agents/llm/tokenizer.json")
if not path.is_file():
    raise SystemExit(f"missing bundled tokenizer: {path}")
if path.stat().st_size < 1_000_000:
    raise SystemExit(
        f"{path} is too small; Git LFS content was probably not checked out"
    )
with path.open("rb") as f:
    json.load(f)
PYEOF

# Mount points (the judge mounts these via `docker run`, see §3.4):
#   /input  ro - all task_<id>/{task.json,context/}
#   /output rw - prediction.csv targets, one dir per task
#   /logs   rw - §3.7: minimal status only; never raw input rows / gold

ENTRYPOINT ["python", "-m", "agents.submission"]
