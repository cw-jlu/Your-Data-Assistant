FROM python:3.10-slim

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install uv

# Copy project configuration and required files for package build
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install dependencies into system environment (using Tsinghua mirror to avoid timeouts)
ENV UV_HTTP_TIMEOUT=300
RUN uv pip install --system -e . --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Copy additional files needed for runtime
COPY main.py ./
COPY configs/ ./configs/

# Evaluation entrypoint
# Competition Spec 3.7: Synchronously save stdout/stderr to /logs/runtime.log
# We use the shell form to allow for pipe and redirection
ENTRYPOINT python main.py 2>&1 | tee /logs/runtime.log
