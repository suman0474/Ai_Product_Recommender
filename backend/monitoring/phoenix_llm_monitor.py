"""
Phoenix LLM Monitoring - Track LLM Calls, Duration, and Performance

This module provides comprehensive monitoring for LLM API calls using the Phoenix framework.
It tracks:
- Number of LLM calls per function
- Execution time per call
- Token usage (input/output)
- Error rates and types
- Cost estimation

Usage:
    from monitoring.phoenix_llm_monitor import llm_monitor, track_llm_call

    # Option 1: Decorator
    @track_llm_call(function_name="identify_instruments")
    def identify_instruments(...):
        ...

    # Option 2: Context manager
    with llm_monitor.track("generate_answer"):
        result = llm.invoke(prompt)

    # Option 3: Manual tracking
    llm_monitor.start_call("my_function", model="gemini-2.5-flash")
    result = llm.invoke(prompt)
    llm_monitor.end_call("my_function", success=True, tokens_in=100, tokens_out=50)

    # Get metrics
    metrics = llm_monitor.get_metrics()
    print(metrics)
"""

import logging
import time
import threading
import json
import os
from typing import Optional, Dict, Any, List, Callable
from functools import wraps
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# =============================================================================
# PHOENIX FRAMEWORK INTEGRATION (Optional - falls back to built-in if not available)
# =============================================================================

PHOENIX_AVAILABLE = False
try:
    import phoenix as px
    from phoenix.trace import SpanKind
    PHOENIX_AVAILABLE = True
    logger.info("[PHOENIX] Phoenix framework available for advanced tracing")
except ImportError:
    logger.info("[PHOENIX] Phoenix not installed - using built-in monitoring only")


# =============================================================================
# DATA CLASSES FOR METRICS
# =============================================================================

@dataclass
class LLMCallRecord:
    """Record of a single LLM call"""
    function_name: str
    model: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_estimate: float = 0.0
    trace_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class FunctionMetrics:
    """Aggregated metrics for a function"""
    function_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost: float = 0.0
    errors: Dict[str, int] = field(default_factory=dict)
    calls_by_model: Dict[str, int] = field(default_factory=dict)
    last_call_time: Optional[str] = None


# =============================================================================
# COST ESTIMATION (2026 pricing)
# =============================================================================

MODEL_PRICING = {
    # Google Gemini pricing (per 1M tokens)
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-lite": {"input": 0.02, "output": 0.10},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},

    # OpenAI pricing (per 1M tokens)
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},

    # Claude pricing (per 1M tokens)
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},

    # Default fallback
    "default": {"input": 0.10, "output": 0.50},
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate cost for an LLM call based on model and token usage"""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
    cost = (tokens_in * pricing["input"] / 1_000_000) + (tokens_out * pricing["output"] / 1_000_000)
    return round(cost, 6)


# =============================================================================
# LLM MONITOR CLASS
# =============================================================================

class LLMMonitor:
    """
    Centralized LLM call monitoring with Phoenix integration.

    Features:
    - Track LLM calls per function
    - Measure execution time
    - Estimate costs
    - Export metrics to various formats
    - Phoenix framework integration for advanced tracing
    """

    def __init__(self, enable_phoenix: bool = True):
        self._lock = threading.RLock()
        self._calls: List[LLMCallRecord] = []
        self._active_calls: Dict[str, LLMCallRecord] = {}
        self._function_metrics: Dict[str, FunctionMetrics] = defaultdict(
            lambda: FunctionMetrics(function_name="unknown")
        )
        self._total_calls = 0
        self._session_start = datetime.now()
        self._enable_phoenix = enable_phoenix and PHOENIX_AVAILABLE

        # Initialize Phoenix if available
        if self._enable_phoenix:
            try:
                # Phoenix session for tracing
                self._phoenix_session = px.launch_app()
                logger.info(f"[PHOENIX] Phoenix UI available at: {self._phoenix_session.url}")
            except Exception as e:
                logger.warning(f"[PHOENIX] Failed to launch Phoenix app: {e}")
                self._enable_phoenix = False

    def start_call(
        self,
        function_name: str,
        model: str = "unknown",
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Start tracking an LLM call.

        Args:
            function_name: Name of the function making the call
            model: LLM model being used
            trace_id: Optional trace ID for distributed tracing
            session_id: Optional session ID for user session tracking

        Returns:
            Call ID for tracking
        """
        call_id = f"{function_name}_{time.time_ns()}"

        record = LLMCallRecord(
            function_name=function_name,
            model=model,
            start_time=time.time(),
            trace_id=trace_id,
            session_id=session_id
        )

        with self._lock:
            self._active_calls[call_id] = record
            self._total_calls += 1

        logger.debug(f"[LLM_MONITOR] Started call: {function_name} (model={model})")
        return call_id

    def end_call(
        self,
        call_id: str,
        success: bool = True,
        error: Optional[str] = None,
        tokens_in: int = 0,
        tokens_out: int = 0
    ) -> Optional[LLMCallRecord]:
        """
        End tracking an LLM call.

        Args:
            call_id: Call ID from start_call
            success: Whether the call succeeded
            error: Error message if failed
            tokens_in: Number of input tokens
            tokens_out: Number of output tokens

        Returns:
            Completed call record
        """
        with self._lock:
            record = self._active_calls.pop(call_id, None)
            if not record:
                logger.warning(f"[LLM_MONITOR] Unknown call_id: {call_id}")
                return None

            # Complete the record
            record.end_time = time.time()
            record.duration_ms = (record.end_time - record.start_time) * 1000
            record.success = success
            record.error = error
            record.tokens_in = tokens_in
            record.tokens_out = tokens_out
            record.cost_estimate = estimate_cost(record.model, tokens_in, tokens_out)

            # Store the record
            self._calls.append(record)

            # Update function metrics
            self._update_function_metrics(record)

        logger.debug(
            f"[LLM_MONITOR] Completed call: {record.function_name} "
            f"(duration={record.duration_ms:.0f}ms, success={success})"
        )

        return record

    def _update_function_metrics(self, record: LLMCallRecord):
        """Update aggregated metrics for a function"""
        func_name = record.function_name
        metrics = self._function_metrics[func_name]
        metrics.function_name = func_name

        # Update counts
        metrics.total_calls += 1
        if record.success:
            metrics.successful_calls += 1
        else:
            metrics.failed_calls += 1
            if record.error:
                error_type = record.error.split(":")[0] if ":" in record.error else record.error[:50]
                metrics.errors[error_type] = metrics.errors.get(error_type, 0) + 1

        # Update duration metrics
        if record.duration_ms:
            metrics.total_duration_ms += record.duration_ms
            metrics.avg_duration_ms = metrics.total_duration_ms / metrics.total_calls
            metrics.min_duration_ms = min(metrics.min_duration_ms, record.duration_ms)
            metrics.max_duration_ms = max(metrics.max_duration_ms, record.duration_ms)

        # Update token metrics
        metrics.total_tokens_in += record.tokens_in
        metrics.total_tokens_out += record.tokens_out
        metrics.total_cost += record.cost_estimate

        # Update model tracking
        metrics.calls_by_model[record.model] = metrics.calls_by_model.get(record.model, 0) + 1

        # Update last call time
        metrics.last_call_time = datetime.now().isoformat()

    @contextmanager
    def track(
        self,
        function_name: str,
        model: str = "unknown",
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Context manager for tracking LLM calls.

        Usage:
            with llm_monitor.track("my_function", model="gemini-2.5-flash") as tracker:
                result = llm.invoke(prompt)
                tracker.set_tokens(tokens_in=100, tokens_out=50)
        """
        call_id = self.start_call(function_name, model, trace_id, session_id)
        tracker = _CallTracker(call_id)

        try:
            yield tracker
            self.end_call(
                call_id,
                success=True,
                tokens_in=tracker.tokens_in,
                tokens_out=tracker.tokens_out
            )
        except Exception as e:
            self.end_call(
                call_id,
                success=False,
                error=str(e),
                tokens_in=tracker.tokens_in,
                tokens_out=tracker.tokens_out
            )
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        with self._lock:
            session_duration = (datetime.now() - self._session_start).total_seconds()

            # Calculate totals
            total_duration = sum(r.duration_ms or 0 for r in self._calls)
            total_tokens_in = sum(r.tokens_in for r in self._calls)
            total_tokens_out = sum(r.tokens_out for r in self._calls)
            total_cost = sum(r.cost_estimate for r in self._calls)
            successful = sum(1 for r in self._calls if r.success)
            failed = len(self._calls) - successful

            return {
                "summary": {
                    "total_calls": len(self._calls),
                    "successful_calls": successful,
                    "failed_calls": failed,
                    "success_rate": f"{(successful / len(self._calls) * 100):.1f}%" if self._calls else "N/A",
                    "total_duration_ms": round(total_duration, 2),
                    "avg_duration_ms": round(total_duration / len(self._calls), 2) if self._calls else 0,
                    "total_tokens_in": total_tokens_in,
                    "total_tokens_out": total_tokens_out,
                    "total_tokens": total_tokens_in + total_tokens_out,
                    "estimated_cost_usd": round(total_cost, 4),
                    "session_duration_seconds": round(session_duration, 2),
                    "calls_per_minute": round(len(self._calls) / (session_duration / 60), 2) if session_duration > 0 else 0,
                },
                "by_function": {
                    name: {
                        "total_calls": m.total_calls,
                        "successful_calls": m.successful_calls,
                        "failed_calls": m.failed_calls,
                        "avg_duration_ms": round(m.avg_duration_ms, 2),
                        "min_duration_ms": round(m.min_duration_ms, 2) if m.min_duration_ms != float('inf') else 0,
                        "max_duration_ms": round(m.max_duration_ms, 2),
                        "total_tokens": m.total_tokens_in + m.total_tokens_out,
                        "estimated_cost_usd": round(m.total_cost, 4),
                        "calls_by_model": dict(m.calls_by_model),
                        "errors": dict(m.errors),
                    }
                    for name, m in self._function_metrics.items()
                },
                "by_model": self._get_model_summary(),
                "recent_calls": self._get_recent_calls(10),
                "active_calls": len(self._active_calls),
            }

    def _get_model_summary(self) -> Dict[str, Dict]:
        """Get metrics grouped by model"""
        model_metrics = defaultdict(lambda: {
            "calls": 0,
            "total_duration_ms": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost": 0,
            "errors": 0
        })

        for record in self._calls:
            m = model_metrics[record.model]
            m["calls"] += 1
            m["total_duration_ms"] += record.duration_ms or 0
            m["tokens_in"] += record.tokens_in
            m["tokens_out"] += record.tokens_out
            m["cost"] += record.cost_estimate
            if not record.success:
                m["errors"] += 1

        # Calculate averages
        for model, m in model_metrics.items():
            m["avg_duration_ms"] = round(m["total_duration_ms"] / m["calls"], 2) if m["calls"] > 0 else 0
            m["total_duration_ms"] = round(m["total_duration_ms"], 2)
            m["cost"] = round(m["cost"], 4)

        return dict(model_metrics)

    def _get_recent_calls(self, n: int) -> List[Dict]:
        """Get the most recent N calls"""
        recent = self._calls[-n:] if len(self._calls) >= n else self._calls
        return [
            {
                "function": r.function_name,
                "model": r.model,
                "duration_ms": round(r.duration_ms, 2) if r.duration_ms else None,
                "success": r.success,
                "tokens": r.tokens_in + r.tokens_out,
                "cost_usd": round(r.cost_estimate, 6),
            }
            for r in reversed(recent)
        ]

    def get_function_report(self, function_name: str) -> Dict[str, Any]:
        """Get detailed report for a specific function"""
        with self._lock:
            metrics = self._function_metrics.get(function_name)
            if not metrics:
                return {"error": f"No metrics found for function: {function_name}"}

            function_calls = [r for r in self._calls if r.function_name == function_name]

            return {
                "function_name": function_name,
                "summary": {
                    "total_calls": metrics.total_calls,
                    "successful_calls": metrics.successful_calls,
                    "failed_calls": metrics.failed_calls,
                    "success_rate": f"{(metrics.successful_calls / metrics.total_calls * 100):.1f}%",
                    "avg_duration_ms": round(metrics.avg_duration_ms, 2),
                    "min_duration_ms": round(metrics.min_duration_ms, 2) if metrics.min_duration_ms != float('inf') else 0,
                    "max_duration_ms": round(metrics.max_duration_ms, 2),
                    "total_tokens": metrics.total_tokens_in + metrics.total_tokens_out,
                    "estimated_cost_usd": round(metrics.total_cost, 4),
                },
                "by_model": dict(metrics.calls_by_model),
                "errors": dict(metrics.errors),
                "last_10_calls": [
                    {
                        "duration_ms": round(r.duration_ms, 2) if r.duration_ms else None,
                        "success": r.success,
                        "model": r.model,
                        "tokens": r.tokens_in + r.tokens_out,
                    }
                    for r in function_calls[-10:]
                ]
            }

    def export_to_json(self, filepath: str = "llm_metrics.json"):
        """Export all metrics to JSON file"""
        metrics = self.get_metrics()
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"[LLM_MONITOR] Exported metrics to {filepath}")

    def reset(self):
        """Reset all metrics"""
        with self._lock:
            self._calls.clear()
            self._active_calls.clear()
            self._function_metrics.clear()
            self._total_calls = 0
            self._session_start = datetime.now()
        logger.info("[LLM_MONITOR] Metrics reset")

    def print_summary(self):
        """Print a formatted summary to console"""
        metrics = self.get_metrics()

        print("\n" + "=" * 80)
        print("LLM CALL MONITORING SUMMARY")
        print("=" * 80)

        summary = metrics["summary"]
        print(f"\nOVERALL METRICS:")
        print(f"  Total Calls:        {summary['total_calls']}")
        print(f"  Success Rate:       {summary['success_rate']}")
        print(f"  Total Duration:     {summary['total_duration_ms']:.0f}ms")
        print(f"  Avg Duration:       {summary['avg_duration_ms']:.0f}ms")
        print(f"  Total Tokens:       {summary['total_tokens']:,}")
        print(f"  Estimated Cost:     ${summary['estimated_cost_usd']:.4f}")
        print(f"  Calls/Minute:       {summary['calls_per_minute']:.1f}")

        print(f"\nBY FUNCTION:")
        print("-" * 80)
        print(f"{'Function':<35} {'Calls':>8} {'Avg(ms)':>10} {'Cost($)':>10}")
        print("-" * 80)
        for name, func_metrics in sorted(metrics["by_function"].items(), key=lambda x: -x[1]["total_calls"]):
            print(f"{name:<35} {func_metrics['total_calls']:>8} {func_metrics['avg_duration_ms']:>10.0f} {func_metrics['estimated_cost_usd']:>10.4f}")

        print(f"\nBY MODEL:")
        print("-" * 80)
        print(f"{'Model':<30} {'Calls':>8} {'Avg(ms)':>10} {'Cost($)':>10}")
        print("-" * 80)
        for model, model_metrics in sorted(metrics["by_model"].items(), key=lambda x: -x[1]["calls"]):
            print(f"{model:<30} {model_metrics['calls']:>8} {model_metrics['avg_duration_ms']:>10.0f} {model_metrics['cost']:>10.4f}")

        print("\n" + "=" * 80)


class _CallTracker:
    """Helper class for context manager tracking"""
    def __init__(self, call_id: str):
        self.call_id = call_id
        self.tokens_in = 0
        self.tokens_out = 0

    def set_tokens(self, tokens_in: int = 0, tokens_out: int = 0):
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out


# =============================================================================
# DECORATOR FOR EASY TRACKING
# =============================================================================

def track_llm_call(
    function_name: Optional[str] = None,
    model: str = "unknown",
    monitor: Optional[LLMMonitor] = None
):
    """
    Decorator to track LLM calls in a function.

    Usage:
        @track_llm_call(function_name="my_func", model="gemini-2.5-flash")
        def my_func():
            return llm.invoke(prompt)
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal function_name
            if function_name is None:
                function_name = func.__name__

            _monitor = monitor or llm_monitor

            call_id = _monitor.start_call(function_name, model)
            try:
                result = func(*args, **kwargs)
                _monitor.end_call(call_id, success=True)
                return result
            except Exception as e:
                _monitor.end_call(call_id, success=False, error=str(e))
                raise
        return wrapper
    return decorator


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Singleton instance for application-wide monitoring
llm_monitor = LLMMonitor(enable_phoenix=True)


# =============================================================================
# INTEGRATION WITH LLM_FALLBACK.PY
# =============================================================================

def patch_llm_fallback_for_monitoring():
    """
    Patch the llm_fallback module to automatically track LLM calls.
    Call this at application startup to enable automatic monitoring.
    """
    try:
        import llm_fallback
        original_invoke = llm_fallback.LLMWithTimeout._invoke_with_timeout

        def monitored_invoke(self, method_name: str, *args, **kwargs):
            model = getattr(self, '_model_name', 'unknown')
            function_name = f"LLMWithTimeout.{method_name}"

            call_id = llm_monitor.start_call(function_name, model)
            try:
                result = original_invoke(self, method_name, *args, **kwargs)

                # Try to extract token usage from result
                tokens_in = 0
                tokens_out = 0
                if hasattr(result, 'usage_metadata'):
                    usage = result.usage_metadata
                    tokens_in = getattr(usage, 'input_tokens', 0) or getattr(usage, 'prompt_tokens', 0) or 0
                    tokens_out = getattr(usage, 'output_tokens', 0) or getattr(usage, 'completion_tokens', 0) or 0

                llm_monitor.end_call(call_id, success=True, tokens_in=tokens_in, tokens_out=tokens_out)
                return result
            except Exception as e:
                llm_monitor.end_call(call_id, success=False, error=str(e))
                raise

        llm_fallback.LLMWithTimeout._invoke_with_timeout = monitored_invoke
        logger.info("[LLM_MONITOR] Successfully patched llm_fallback for automatic monitoring")
        return True

    except Exception as e:
        logger.warning(f"[LLM_MONITOR] Failed to patch llm_fallback: {e}")
        return False


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Monitoring Utility")
    parser.add_argument("--demo", action="store_true", help="Run demo to show monitoring capabilities")
    parser.add_argument("--export", type=str, help="Export metrics to JSON file")
    args = parser.parse_args()

    if args.demo:
        print("Running LLM Monitor Demo...")

        # Simulate some LLM calls
        monitor = LLMMonitor(enable_phoenix=False)

        # Simulated calls
        test_functions = [
            ("identify_instruments", "gemini-2.5-pro", 4500, True, 1500, 800),
            ("identify_accessories", "gemini-2.5-pro", 8200, True, 2000, 1200),
            ("generate_schema", "gemini-2.5-flash", 1200, True, 500, 300),
            ("validate_response", "gemini-2.5-flash", 800, True, 400, 100),
            ("standards_rag_query", "gemini-2.5-flash", 2500, True, 1000, 600),
            ("standards_rag_query", "gemini-2.5-flash", 2200, False, 1000, 0),
            ("deep_agent_enrich", "gemini-2.5-flash", 3500, True, 800, 500),
        ]

        for func_name, model, duration_ms, success, tokens_in, tokens_out in test_functions:
            call_id = monitor.start_call(func_name, model)
            time.sleep(duration_ms / 10000)  # Simulate some work (scaled down)
            monitor.end_call(
                call_id,
                success=success,
                error=None if success else "Simulated error",
                tokens_in=tokens_in,
                tokens_out=tokens_out
            )

        # Print summary
        monitor.print_summary()

        # Export if requested
        if args.export:
            monitor.export_to_json(args.export)

    elif args.export:
        # Export current metrics
        llm_monitor.export_to_json(args.export)
        print(f"Metrics exported to {args.export}")
    else:
        parser.print_help()
