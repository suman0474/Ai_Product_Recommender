"""
Gunicorn Production Configuration for AI Product Recommender API

This file configures Gunicorn WSGI server with optimal settings for production deployment.

Usage:
    gunicorn -c gunicorn.conf.py main:app

Or use the provided startup script:
    python start_production.py
    ./start_production.sh
"""

import multiprocessing
import os
import logging

# =============================================================================
# Server Socket Configuration
# =============================================================================

# Bind to all interfaces on port 5000 (override with GUNICORN_BIND env var)
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:5000")

# Backlog - maximum number of pending connections
backlog = 2048


# =============================================================================
# Worker Processes Configuration
# =============================================================================

# Number of worker processes
# Default: (2 x CPU cores) + 1
# Override with GUNICORN_WORKERS environment variable
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))

# Worker class - 'gthread' for better concurrency with async operations
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "gthread")

# Threads per worker (only for gthread worker class)
# Total concurrent requests = workers * threads
threads = int(os.getenv("GUNICORN_THREADS", 4))


# =============================================================================
# Worker Lifecycle & Timeout Configuration
# =============================================================================

# Workers silent for more than this many seconds are killed and restarted
# Set to 1 hour for long-running AI/LLM operations
timeout = int(os.getenv("GUNICORN_TIMEOUT", 3600))

# Graceful timeout for workers during reload/restart
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", 60))

# Maximum requests a worker will process before restarting
# Helps prevent memory leaks from long-running processes
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", 1000))

# Randomize max_requests to avoid all workers restarting simultaneously
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", 100))

# Timeout for keep-alive connections
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", 5))


# =============================================================================
# Security Configuration
# =============================================================================

# Limit request line size (prevents malicious large headers)
limit_request_line = 4096

# Limit request header field size
limit_request_field_size = 8190

# Limit number of request header fields
limit_request_fields = 100


# =============================================================================
# Logging Configuration
# =============================================================================

# Access log format with timing information
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Log to stdout/stderr (Systemd friendly)
accesslog = os.getenv("GUNICORN_ACCESS_LOG", "-")  # "-" means stdout
errorlog = os.getenv("GUNICORN_ERROR_LOG", "-")    # "-" means stderr

# Log level
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")

# Capture Flask app logs
capture_output = True

# Disable access log if needed (reduces log volume)
if os.getenv("GUNICORN_ACCESS_LOG_ENABLED", "true").lower() != "true":
    accesslog = None


# =============================================================================
# Process Naming
# =============================================================================

# Process name in ps/top output
proc_name = "aipr-gunicorn"


# =============================================================================
# Server Mechanics
# =============================================================================

# Daemonize (keep False for Systemd - it handles daemonization)
daemon = False

# PID file location
pidfile = os.getenv("GUNICORN_PID_FILE", "/tmp/gunicorn_aipr.pid")

# Temp directory for worker heartbeat (use RAM disk for performance)
worker_tmp_dir = os.getenv("GUNICORN_WORKER_TMP_DIR", None)  # Set to "/dev/shm" for RAM disk


# =============================================================================
# Development vs Production Settings
# =============================================================================

# Reload workers when code changes (DEVELOPMENT ONLY!)
reload = os.getenv("GUNICORN_RELOAD", "false").lower() == "true"

# Preload application code before worker processes are forked
# Saves memory but makes code reloading harder
preload_app = os.getenv("GUNICORN_PRELOAD", "true").lower() == "true"


# =============================================================================
# Server Hooks (Lifecycle Events)
# =============================================================================

def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("="*70)
    server.log.info("Starting AI Product Recommender API (Production Mode)")
    server.log.info(f"Workers: {workers} | Threads/Worker: {threads} | Timeout: {timeout}s")
    server.log.info(f"Worker Class: {worker_class} | Max Requests: {max_requests}")
    server.log.info(f"Bind: {bind} | Preload: {preload_app}")
    server.log.info("="*70)


def when_ready(server):
    """Called just after the server is started."""
    server.log.info("✓ Server is ready. Accepting connections.")


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info(f"Worker spawned (pid: {worker.pid})")


def worker_int(worker):
    """Called when worker receives SIGINT or SIGQUIT signal."""
    worker.log.info(f"Worker interrupted (pid: {worker.pid})")


def worker_abort(worker):
    """Called when a worker received the SIGABRT signal."""
    worker.log.warning(f"Worker aborted (pid: {worker.pid})")


def on_exit(server):
    """Called just before exiting Gunicorn."""
    server.log.info("Shutting down: Gunicorn master process")


# =============================================================================
# SSL Configuration (HTTPS) - Uncomment for production with SSL
# =============================================================================

# keyfile = os.getenv("GUNICORN_SSL_KEY", "/path/to/ssl/key.pem")
# certfile = os.getenv("GUNICORN_SSL_CERT", "/path/to/ssl/cert.pem")
# ca_certs = os.getenv("GUNICORN_SSL_CA", "/path/to/ssl/ca-bundle.crt")


# =============================================================================
# Environment Variables Reference
# =============================================================================
"""
Available Environment Variables:

GUNICORN_BIND                - Server bind address (default: 0.0.0.0:5000)
GUNICORN_WORKERS             - Number of worker processes (default: CPU*2+1)
GUNICORN_THREADS             - Threads per worker (default: 4)
GUNICORN_TIMEOUT             - Worker timeout in seconds (default: 3600)
GUNICORN_WORKER_CLASS        - Worker type: sync, gthread, gevent (default: gthread)
GUNICORN_LOG_LEVEL           - Log level: debug, info, warning, error (default: info)
GUNICORN_MAX_REQUESTS        - Max requests before worker restart (default: 1000)
GUNICORN_PRELOAD             - Preload app code (default: true)
GUNICORN_RELOAD              - Auto-reload on code changes - DEV ONLY (default: false)
GUNICORN_ACCESS_LOG_ENABLED  - Enable access logging (default: true)
GUNICORN_WORKER_TMP_DIR      - Worker temp directory (default: None, use "/dev/shm" for RAM disk)

Production Tuning Guide:

1. For AI/LLM workloads (current setup):
   - Workers: 2-4 (fewer workers due to high memory per worker)
   - Threads: 4-8 (good for I/O-bound LLM API calls)
   - Worker class: gthread
   - Timeout: 3600s (1 hour for long-running inference)
   - Memory: ~500MB-1GB per worker

2. For high-traffic API:
   - Workers: CPU cores * 2 + 1
   - Threads: 2-4
   - Worker class: gthread
   - Timeout: 300s
   - Use load balancer for horizontal scaling

3. Memory optimization:
   - Set max_requests=500-1000 to recycle workers
   - Monitor: ps aux | grep gunicorn
   - Adjust workers down if memory > 80%

4. Performance monitoring:
   - Watch worker restart frequency (shouldn't restart often)
   - Monitor request latency in access logs
   - Use APM tools (New Relic, Datadog) for deep insights
"""
