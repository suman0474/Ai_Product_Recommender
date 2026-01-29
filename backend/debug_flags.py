"""
Debug Flags Module for Product Search Workflow
==============================================

Centralized debug flag management for all backend functions.
Provides decorators for function entry/exit logging and performance timing.

Environment Variables:
- DEBUG_ALL=1                 Enable all debug flags
- DEBUG_VALIDATION_TOOL=1     Enable validation tool debugging
- DEBUG_ADVANCED_PARAMS=1     Enable advanced parameters debugging
- DEBUG_VENDOR_ANALYSIS=1     Enable vendor analysis debugging
- DEBUG_RANKING_TOOL=1        Enable ranking tool debugging
- DEBUG_WORKFLOW=1            Enable workflow orchestration debugging
- DEBUG_INTENT_TOOLS=1        Enable intent tools debugging
- DEBUG_LLM_FALLBACK=1        Enable LLM fallback debugging
- DEBUG_API_ENDPOINTS=1       Enable API endpoint debugging

Usage:
    from debug_flags import debug_log, timed_execution, is_debug_enabled

    @debug_log("VALIDATION_TOOL")
    def my_function(arg1, arg2):
        pass

    @timed_execution("WORKFLOW")
    def my_workflow():
        pass
"""

import os
import time
import logging
import functools
from typing import Any, Callable, Optional, Dict

logger = logging.getLogger(__name__)


# ============================================================================
# DEBUG FLAGS CONFIGURATION
# ============================================================================

DEBUG_FLAGS: Dict[str, bool] = {
    # Module-level flags (set via environment variables)
    "VALIDATION_TOOL": os.getenv("DEBUG_VALIDATION_TOOL", "0") == "1",
    "ADVANCED_PARAMS": os.getenv("DEBUG_ADVANCED_PARAMS", "0") == "1",
    "VENDOR_ANALYSIS": os.getenv("DEBUG_VENDOR_ANALYSIS", "0") == "1",
    "RANKING_TOOL": os.getenv("DEBUG_RANKING_TOOL", "0") == "1",
    "WORKFLOW": os.getenv("DEBUG_WORKFLOW", "0") == "1",
    "INTENT_TOOLS": os.getenv("DEBUG_INTENT_TOOLS", "0") == "1",
    "LLM_FALLBACK": os.getenv("DEBUG_LLM_FALLBACK", "0") == "1",
    "API_ENDPOINTS": os.getenv("DEBUG_API_ENDPOINTS", "0") == "1",
    "SALES_AGENT": os.getenv("DEBUG_SALES_AGENT", "0") == "1",
    "PPI_WORKFLOW": os.getenv("DEBUG_PPI_WORKFLOW", "0") == "1",
    "STANDARDS_RAG": os.getenv("DEBUG_STANDARDS_RAG", "0") == "1",
    "STRATEGY_RAG": os.getenv("DEBUG_STRATEGY_RAG", "0") == "1",

    # Global flag to enable all debugging
    "ALL": os.getenv("DEBUG_ALL", "0") == "1",
}


# ============================================================================
# FLAG MANAGEMENT FUNCTIONS
# ============================================================================

def is_debug_enabled(module: str) -> bool:
    """
    Check if debugging is enabled for a specific module.

    Args:
        module: Module name (e.g., "VALIDATION_TOOL", "WORKFLOW")

    Returns:
        True if debugging is enabled for the module or globally
    """
    return DEBUG_FLAGS.get("ALL", False) or DEBUG_FLAGS.get(module, False)


def get_debug_flag(module: str) -> bool:
    """
    Get the current debug flag value for a module.

    Args:
        module: Module name

    Returns:
        Current flag value (True/False)
    """
    return DEBUG_FLAGS.get(module, False)


def set_debug_flag(module: str, enabled: bool) -> None:
    """
    Set a debug flag programmatically (useful for testing).

    Args:
        module: Module name
        enabled: Whether to enable (True) or disable (False) debugging
    """
    DEBUG_FLAGS[module] = enabled
    logger.info(f"[DEBUG_FLAGS] Set {module} = {enabled}")


def enable_all_debug() -> None:
    """Enable debugging for all modules."""
    DEBUG_FLAGS["ALL"] = True
    logger.info("[DEBUG_FLAGS] All debugging enabled")


def disable_all_debug() -> None:
    """Disable global debugging (individual flags still apply)."""
    DEBUG_FLAGS["ALL"] = False
    logger.info("[DEBUG_FLAGS] Global debugging disabled")


def get_enabled_flags() -> Dict[str, bool]:
    """
    Get a dictionary of all currently enabled debug flags.

    Returns:
        Dictionary of flag_name -> enabled status
    """
    return {k: v for k, v in DEBUG_FLAGS.items() if v}


# ============================================================================
# DECORATORS
# ============================================================================

def debug_log(module: str, log_args: bool = True, log_result: bool = False):
    """
    Decorator for function entry/exit logging.

    Logs function entry with arguments and exit with result (if enabled).
    Only logs when debugging is enabled for the module.

    Args:
        module: Module name for debug flag check
        log_args: Whether to log function arguments (default: True)
        log_result: Whether to log return value (default: False, can be verbose)

    Usage:
        @debug_log("VALIDATION_TOOL")
        def validate(user_input, session_id=None):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_debug_enabled(module):
                return func(*args, **kwargs)

            func_name = func.__name__

            # Log entry
            if log_args:
                # Truncate long arguments for readability
                args_str = ", ".join(repr(a)[:100] for a in args)
                kwargs_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in kwargs.items())
                all_args = f"({args_str}{', ' if args_str and kwargs_str else ''}{kwargs_str})"
                logger.debug(f"[{module}] >> ENTER {func_name}{all_args}")
            else:
                logger.debug(f"[{module}] >> ENTER {func_name}()")

            try:
                result = func(*args, **kwargs)

                # Log exit
                if log_result and result is not None:
                    result_str = repr(result)[:200]
                    logger.debug(f"[{module}] << EXIT {func_name} => {result_str}")
                else:
                    logger.debug(f"[{module}] << EXIT {func_name} => OK")

                return result

            except Exception as e:
                logger.debug(f"[{module}] << EXIT {func_name} => EXCEPTION: {type(e).__name__}: {e}")
                raise

        return wrapper
    return decorator


def timed_execution(module: str, threshold_ms: Optional[float] = None):
    """
    Decorator for performance timing.

    Measures and logs function execution time.
    Optionally warns if execution exceeds threshold.

    Args:
        module: Module name for debug flag check
        threshold_ms: Optional threshold in milliseconds. If exceeded, logs a warning.

    Usage:
        @timed_execution("WORKFLOW", threshold_ms=5000)
        def run_workflow():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_debug_enabled(module):
                return func(*args, **kwargs)

            func_name = func.__name__
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000

                # Log timing
                if threshold_ms and elapsed_ms > threshold_ms:
                    logger.warning(
                        f"[{module}] SLOW: {func_name} took {elapsed_ms:.2f}ms "
                        f"(threshold: {threshold_ms}ms)"
                    )
                else:
                    logger.debug(f"[{module}] TIMING: {func_name} took {elapsed_ms:.2f}ms")

                return result

            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"[{module}] TIMING: {func_name} failed after {elapsed_ms:.2f}ms "
                    f"({type(e).__name__})"
                )
                raise

        return wrapper
    return decorator


def debug_state(module: str, state_name: str = "state"):
    """
    Decorator for logging workflow state at entry and exit.

    Useful for debugging LangGraph-style workflows where state is passed through.

    Args:
        module: Module name for debug flag check
        state_name: Name of the state parameter (default: "state")

    Usage:
        @debug_state("WORKFLOW")
        def process_step(state: Dict):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_debug_enabled(module):
                return func(*args, **kwargs)

            func_name = func.__name__

            # Try to find state in args or kwargs
            state = kwargs.get(state_name) or (args[0] if args else None)

            if state and isinstance(state, dict):
                state_keys = list(state.keys())
                logger.debug(f"[{module}] STATE_IN {func_name}: keys={state_keys}")
            else:
                logger.debug(f"[{module}] STATE_IN {func_name}: (no dict state)")

            result = func(*args, **kwargs)

            if result and isinstance(result, dict):
                result_keys = list(result.keys())
                logger.debug(f"[{module}] STATE_OUT {func_name}: keys={result_keys}")

            return result

        return wrapper
    return decorator


# ============================================================================
# UI DECISION PATTERN DETECTION (For validation_tool.py Fix)
# ============================================================================

# Patterns that indicate UI decision states, NOT product requirements
UI_DECISION_PATTERNS = [
    "user selected:",
    "user clicked:",
    "user chose:",
    "decision:",
    "action:",
    "button clicked:",
    "continue",
    "proceed",
    "go back",
    "cancel",
    "skip",
    "confirm",
]


def is_ui_decision_input(user_input: str) -> bool:
    """
    Check if the input is a UI decision pattern rather than product requirements.

    UI decision patterns are inputs like "User selected: continue" that come from
    the frontend when a user clicks a button. These should NOT be processed as
    product requirements.

    Args:
        user_input: The user input string to check

    Returns:
        True if the input matches a UI decision pattern
    """
    if not user_input:
        return False

    normalized = user_input.lower().strip()

    # Check for exact matches (single-word decisions)
    single_word_decisions = {"continue", "proceed", "skip", "cancel", "confirm", "back"}
    if normalized in single_word_decisions:
        return True

    # Check for pattern matches
    for pattern in UI_DECISION_PATTERNS:
        if pattern in normalized:
            return True

    return False


def get_ui_decision_error_message(user_input: str) -> str:
    """
    Get an appropriate error message for UI decision inputs.

    Args:
        user_input: The UI decision input

    Returns:
        Descriptive error message
    """
    return (
        f"Input '{user_input}' appears to be a UI navigation action, not a product requirement. "
        "Please provide actual product requirements (e.g., 'I need a pressure transmitter with 4-20mA output') "
        "or use the appropriate API endpoint for the current workflow state."
    )


# ============================================================================
# INITIALIZATION LOGGING
# ============================================================================

def _log_debug_status():
    """Log current debug flag status on module load."""
    enabled = get_enabled_flags()
    if enabled:
        logger.info(f"[DEBUG_FLAGS] Enabled flags: {list(enabled.keys())}")
    else:
        logger.debug("[DEBUG_FLAGS] No debug flags enabled")


# Log status on import
_log_debug_status()
