#!/usr/bin/env python
"""
Orchestration Test Runner with Debug Flags

This script runs the orchestration implementation tests with various debug options.
Reusable for future test runs after code changes.

Usage:
    python run_orchestration_tests.py                    # Run all tests with debug
    python run_orchestration_tests.py --quiet            # Run without debug output
    python run_orchestration_tests.py --test logger      # Run only logger tests
    python run_orchestration_tests.py --test lock        # Run only lock tests
    python run_orchestration_tests.py --test storage     # Run only storage tests
    python run_orchestration_tests.py --test router      # Run only router tests
    python run_orchestration_tests.py --test multiuser   # Run only multi-user tests
    python run_orchestration_tests.py --quick            # Quick smoke test
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

# Add backend to path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)


def print_header(title: str):
    """Print a formatted header."""
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)
    print()


def print_section(title: str):
    """Print a section header."""
    print()
    print("-" * 50)
    print(f" {title}")
    print("-" * 50)


def run_tests(
    test_filter: str = None,
    debug: bool = True,
    verbose: bool = True,
    stop_on_fail: bool = True,
    coverage: bool = False
):
    """
    Run the orchestration tests.
    
    Args:
        test_filter: Optional filter for specific test class
        debug: Enable debug output
        verbose: Verbose pytest output
        stop_on_fail: Stop on first failure
        coverage: Enable coverage reporting
    """
    print_header("ORCHESTRATION IMPLEMENTATION TESTS")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Debug Mode: {debug}")
    print(f"Test Filter: {test_filter or 'all'}")
    
    # Set environment
    env = os.environ.copy()
    env["DEBUG_TESTS"] = "1" if debug else "0"
    env["PYTHONPATH"] = BACKEND_DIR
    
    # Build pytest command
    test_file = os.path.join(BACKEND_DIR, "tests", "test_orchestration_impl.py")
    
    cmd = ["python", "-m", "pytest", test_file]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    if stop_on_fail:
        cmd.append("-x")
    
    cmd.append("--tb=long")
    
    # Add test filter
    if test_filter:
        filter_map = {
            "logger": "TestStructuredLogger",
            "lock": "TestDistributedLock or TestOptimisticLock",
            "distributed": "TestDistributedLock",
            "optimistic": "TestOptimisticLock",
            "storage": "TestStateStorage",
            "router": "TestRouterAgent",
            "multiuser": "TestMultiUserConcurrency",
        }
        
        if test_filter in filter_map:
            cmd.extend(["-k", filter_map[test_filter]])
        else:
            cmd.extend(["-k", test_filter])
    
    if coverage:
        cmd.extend([
            "--cov=observability",
            "--cov=concurrency",
            "--cov=agentic.state_storage",
            "--cov=agentic.router_agent",
            "--cov-report=term-missing"
        ])
    
    print()
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run tests
    result = subprocess.run(cmd, env=env, cwd=BACKEND_DIR)
    
    return result.returncode


def run_quick_test():
    """Run a quick smoke test to verify imports work."""
    print_header("QUICK SMOKE TEST")
    
    errors = []
    
    # Test imports
    print_section("Testing Imports")
    
    try:
        from observability.structured_logger import StructuredLogger, set_correlation_context
        print("✓ observability.structured_logger")
    except Exception as e:
        print(f"✗ observability.structured_logger: {e}")
        errors.append(str(e))
    
    try:
        from concurrency.distributed_lock import InMemoryDistributedLock, create_distributed_lock
        print("✓ concurrency.distributed_lock")
    except Exception as e:
        print(f"✗ concurrency.distributed_lock: {e}")
        errors.append(str(e))
    
    try:
        from concurrency.optimistic_lock import InMemoryOptimisticLockManager, VersionedState
        print("✓ concurrency.optimistic_lock")
    except Exception as e:
        print(f"✗ concurrency.optimistic_lock: {e}")
        errors.append(str(e))
    
    try:
        from agentic.state_storage import InMemoryStateStorage, SessionState, InstanceState
        print("✓ agentic.state_storage")
    except Exception as e:
        print(f"✗ agentic.state_storage: {e}")
        errors.append(str(e))
    
    try:
        from agentic.router_agent import RouterAgent, WorkflowType, RoutingResult
        print("✓ agentic.router_agent")
    except Exception as e:
        print(f"✗ agentic.router_agent: {e}")
        errors.append(str(e))
    
    # Quick functional tests
    print_section("Quick Functional Tests")
    
    try:
        # Test logger
        from observability.structured_logger import StructuredLogger
        logger = StructuredLogger("QuickTest")
        logger.info("Quick test message")
        print("✓ Logger works")
    except Exception as e:
        print(f"✗ Logger: {e}")
        errors.append(str(e))
    
    try:
        # Test lock
        from concurrency.distributed_lock import InMemoryDistributedLock
        lock = InMemoryDistributedLock()
        with lock.acquire("quick-test"):
            pass
        print("✓ Distributed lock works")
    except Exception as e:
        print(f"✗ Distributed lock: {e}")
        errors.append(str(e))
    
    try:
        # Test storage
        from agentic.state_storage import InMemoryStateStorage, SessionState
        storage = InMemoryStateStorage()
        session = SessionState(session_id="test", user_id="user")
        storage.save_session("test", session)
        retrieved = storage.get_session("test")
        assert retrieved is not None
        print("✓ State storage works")
    except Exception as e:
        print(f"✗ State storage: {e}")
        errors.append(str(e))
    
    try:
        # Test router
        from agentic.router_agent import RouterAgent
        router = RouterAgent(use_memory_fallback=True)
        result = router.route("test-session", "test query", "user")
        assert result is not None
        print("✓ Router agent works")
    except Exception as e:
        print(f"✗ Router agent: {e}")
        errors.append(str(e))
    
    # Summary
    print_section("Summary")
    if errors:
        print(f"✗ {len(errors)} error(s) occurred:")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("✓ All quick tests passed!")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run orchestration implementation tests"
    )
    
    parser.add_argument(
        "--test", "-t",
        type=str,
        choices=["logger", "lock", "distributed", "optimistic", 
                 "storage", "router", "multiuser"],
        help="Run specific test class"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Disable debug output"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick smoke test only"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Enable coverage reporting"
    )
    
    parser.add_argument(
        "--no-stop",
        action="store_true",
        help="Don't stop on first failure"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        return run_quick_test()
    
    return run_tests(
        test_filter=args.test,
        debug=not args.quiet,
        stop_on_fail=not args.no_stop,
        coverage=args.coverage
    )


if __name__ == "__main__":
    sys.exit(main())
