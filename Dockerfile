# ---------------------------------------------------------------------
# Development & Automated Test Environment
# ---------------------------------------------------------------------
FROM rust:1.85-slim-bookworm AS development

# Install Python 3, pip, and required system build libraries
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Setup isolated Python virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Pre-install core open-source machine learning and simulation frameworks
COPY reinforcement_learning/python/requirements.txt ./reinforcement_learning/python/
RUN pip install --no-cache-dir -r ./reinforcement_learning/python/requirements.txt

# Tell PyO3 exactly which Python interpreter binary to bind against during compilation
ENV PYO3_PYTHON="/app/venv/bin/python"

# Copy workspace metadata (with the lockfile for reproducible builds) and code
COPY Cargo.toml Cargo.lock ./
COPY reinforcement_learning/ ./reinforcement_learning/
