"""
Monitoring Package for AIPR Backend

This package provides comprehensive monitoring capabilities for:
- LLM API calls (using Phoenix framework)
- Workflow execution times
- Error tracking
- Cost estimation
"""

from .phoenix_llm_monitor import (
    llm_monitor,
    track_llm_call,
    LLMMonitor,
    patch_llm_fallback_for_monitoring,
)

__all__ = [
    "llm_monitor",
    "track_llm_call",
    "LLMMonitor",
    "patch_llm_fallback_for_monitoring",
]
