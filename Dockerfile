# ---------------------------------------------------------------------
# Development & Automated Test Environment
# ---------------------------------------------------------------------
FROM rust:1.85-slim-bookworm AS development

# Install Python 3 (3.11 on bookworm), its headers and libpython for PyO3
# to link against, and required system build libraries
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv

WORKDIR /app

# Build the virtual environment from uv.lock with the system interpreter.
# The container runs cargo test only, so the oxidized_rl module itself is
# not built (--no-install-project).
ENV UV_PROJECT_ENVIRONMENT=/app/venv \
    UV_PYTHON_DOWNLOADS=never
ENV PATH="/app/venv/bin:$PATH"
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project --no-cache

# Tell PyO3 exactly which Python interpreter binary to bind against during compilation
ENV PYO3_PYTHON="/app/venv/bin/python"

# Copy workspace metadata (with the lockfile for reproducible builds) and code
COPY Cargo.toml Cargo.lock ./
COPY reinforcement_learning/ ./reinforcement_learning/
COPY reinforcement_learning_py/ ./reinforcement_learning_py/
