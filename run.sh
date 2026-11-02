#!/usr/bin/env bash
# /run.sh
set -euo pipefail

# Print helpful usage patterns if arguments are omitted
if [ $# -eq 0 ]; then
    echo "❌ Missing command argument."
    echo "Usage:"
    echo "  ./run.sh dev-test   - Run tests natively on your Mac host machine"
    echo "  ./run.sh build      - Force clean cache and build the Docker container images"
    echo "  ./run.sh test       - Execute full test suite completely inside Docker container"
    exit 1
fi

COMMAND=$1

case "$COMMAND" in
    dev-test)
        echo "🚀 Running tests natively on macOS host..."
        # Dynamically append the absolute path to your local virtual environment site-packages
        # This completely eliminates the need to run 'source .venv/bin/activate' first!
        export PYO3_PYTHON="$(pwd)/.venv/bin/python"
        
        # Verify the local environment exists before compiling
        if [ ! -d ".venv" ]; then
            echo "❌ Local .venv directory not found. Run 'uv sync' first."
            exit 1
        fi
        
        cargo test
        uv run pytest -q
        ;;
        
    build)
        echo "🧱 Wiping build cache and assembling Docker containers..."
        cargo clean
        docker-compose build --no-cache rl-test
        ;;
        
    test)
        echo "🐳 Launching test suite inside the isolated Docker container..."
        # Launch Docker Desktop if it isn't running
        if ! docker ps >/dev/null 2>&1; then
            echo "🔄 Docker daemon not detected. Launching Docker Desktop..."
            open -a Docker
            while ! docker ps >/dev/null 2>&1; do sleep 1; done
        fi
        docker-compose run --rm rl-test
        ;;
        
    *)
        echo "❌ Unknown command: $COMMAND"
        exit 1
        ;;
esac
