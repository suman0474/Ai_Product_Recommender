# =============================================================================
# AIPR Backend - Dockerfile
# =============================================================================
# Multi-stage Python build for Flask API with Gunicorn

# Stage 1: Builder - Install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# =============================================================================
# Stage 2: Production - Gunicorn WSGI Server
# =============================================================================
FROM python:3.11-slim AS production

# Labels
LABEL maintainer="AIPR Team" \
    version="1.0.0" \
    description="AIPR Backend - Flask API with LangChain/LangGraph"

# Create non-root user for security
RUN groupadd -r aipr && useradd -r -g aipr aipr

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /home/aipr/.local

# Ensure scripts in .local are usable
ENV PATH=/home/aipr/.local/bin:$PATH

# Copy application code
COPY --chown=aipr:aipr . .

# Create necessary directories
RUN mkdir -p /app/flask_session /app/chroma_data /app/vector_store_data /app/static \
    && chown -R aipr:aipr /app

# Copy entrypoint script
COPY --chown=aipr:aipr docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=main.py \
    FLASK_ENV=production \
    GUNICORN_WORKERS=4 \
    GUNICORN_THREADS=4 \
    GUNICORN_TIMEOUT=3600 \
    GUNICORN_BIND=0.0.0.0:5000

# Switch to non-root user
USER aipr

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:5000/health || exit 1

# Entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]

# Default command - Gunicorn with production config
CMD ["gunicorn", "-c", "gunicorn.conf.py", "main:app"]
