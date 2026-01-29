#!/bin/bash
# =============================================================================
# AIPR Backend - Docker Entrypoint Script
# =============================================================================
# Handles startup tasks before launching Gunicorn

set -e

echo "========================================"
echo "AIPR Backend - Starting up..."
echo "========================================"

# Validate required environment variables
echo "[Entrypoint] Checking environment..."

if [ -z "$GOOGLE_API_KEY" ]; then
    echo "[WARNING] GOOGLE_API_KEY not set - some features may be unavailable"
fi

if [ -z "$AZURE_STORAGE_CONNECTION_STRING" ]; then
    echo "[WARNING] AZURE_STORAGE_CONNECTION_STRING not set - Azure Blob features disabled"
fi

if [ -z "$PINECONE_API_KEY" ]; then
    echo "[WARNING] PINECONE_API_KEY not set - Pinecone vector store unavailable"
fi

# Create session directory if needed
if [ ! -d "/app/flask_session" ]; then
    mkdir -p /app/flask_session
    echo "[Entrypoint] Created flask_session directory"
fi

# Create vector store directory if needed
if [ ! -d "/app/vector_store_data" ]; then
    mkdir -p /app/vector_store_data
    echo "[Entrypoint] Created vector_store_data directory"
fi

# Log configuration
echo "[Entrypoint] Configuration:"
echo "  - Gunicorn Workers: ${GUNICORN_WORKERS:-auto}"
echo "  - Gunicorn Threads: ${GUNICORN_THREADS:-4}"
echo "  - Gunicorn Timeout: ${GUNICORN_TIMEOUT:-3600}s"
echo "  - Bind Address: ${GUNICORN_BIND:-0.0.0.0:5000}"

echo "========================================"
echo "Launching application..."
echo "========================================"

# Execute the main command
exec "$@"
