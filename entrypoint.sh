#!/bin/sh
# Entrypoint script for Silence Decoder container

set -e

echo "Starting Silence Pattern Decoder..."
echo "Container started at: $(date)"
echo "Python version: $(python --version)"

# Run the command passed to docker run
exec "$@"