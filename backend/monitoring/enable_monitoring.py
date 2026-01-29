"""
Enable LLM Monitoring at Application Startup

Add this import to your main application file (e.g., api.py):
    from monitoring.enable_monitoring import enable_llm_monitoring

Then call it during startup:
    enable_llm_monitoring()

This will:
1. Initialize the Phoenix monitoring framework
2. Patch LLM fallback for automatic call tracking
3. Set up metrics endpoints

You can then access metrics via:
- llm_monitor.get_metrics() - Get all metrics
- llm_monitor.print_summary() - Print formatted summary
- llm_monitor.export_to_json("metrics.json") - Export to file
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def enable_llm_monitoring(enable_phoenix: bool = True) -> bool:
    """
    Enable LLM monitoring for the application.

    This function:
    1. Initializes the Phoenix monitoring framework (if enabled)
    2. Patches llm_fallback.py for automatic call tracking
    3. Returns True if successful, False otherwise

    Args:
        enable_phoenix: Whether to enable Phoenix framework integration
                       (set to False if Phoenix is not installed)

    Returns:
        True if monitoring was successfully enabled
    """
    try:
        from .phoenix_llm_monitor import llm_monitor, patch_llm_fallback_for_monitoring

        # Patch llm_fallback for automatic monitoring
        patched = patch_llm_fallback_for_monitoring()

        if patched:
            logger.info("=" * 60)
            logger.info("[MONITORING] LLM Monitoring ENABLED")
            logger.info("[MONITORING] All LLM calls will be tracked automatically")
            logger.info("[MONITORING] Access metrics via:")
            logger.info("[MONITORING]   - llm_monitor.get_metrics()")
            logger.info("[MONITORING]   - llm_monitor.print_summary()")
            logger.info("[MONITORING]   - GET /api/monitoring/metrics")
            logger.info("=" * 60)
            return True
        else:
            logger.warning("[MONITORING] Monitoring enabled but auto-patching failed")
            logger.warning("[MONITORING] Use @track_llm_call decorator for manual tracking")
            return True

    except Exception as e:
        logger.error(f"[MONITORING] Failed to enable monitoring: {e}")
        return False


def get_monitoring_api_routes():
    """
    Get FastAPI routes for monitoring endpoints.

    Add these routes to your FastAPI app:
        from monitoring.enable_monitoring import get_monitoring_api_routes
        app.include_router(get_monitoring_api_routes())

    Returns:
        FastAPI APIRouter with monitoring endpoints
    """
    try:
        from fastapi import APIRouter
        from fastapi.responses import JSONResponse
        from .phoenix_llm_monitor import llm_monitor

        router = APIRouter(prefix="/api/monitoring", tags=["Monitoring"])

        @router.get("/metrics")
        async def get_metrics():
            """Get comprehensive LLM monitoring metrics"""
            return JSONResponse(content=llm_monitor.get_metrics())

        @router.get("/metrics/{function_name}")
        async def get_function_metrics(function_name: str):
            """Get metrics for a specific function"""
            return JSONResponse(content=llm_monitor.get_function_report(function_name))

        @router.post("/metrics/reset")
        async def reset_metrics():
            """Reset all monitoring metrics"""
            llm_monitor.reset()
            return {"status": "success", "message": "Metrics reset"}

        @router.get("/metrics/export")
        async def export_metrics():
            """Export metrics to JSON"""
            llm_monitor.export_to_json("llm_metrics_export.json")
            return {"status": "success", "file": "llm_metrics_export.json"}

        return router

    except ImportError:
        logger.warning("[MONITORING] FastAPI not available, skipping API routes")
        return None


# Auto-enable monitoring when this module is imported
# (can be disabled by setting DISABLE_LLM_MONITORING=true)
import os
if os.getenv("DISABLE_LLM_MONITORING", "").lower() != "true":
    enable_llm_monitoring()
