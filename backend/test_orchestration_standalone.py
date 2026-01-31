#!/usr/bin/env python
"""
Standalone Test Suite for Orchestration Implementation

This file runs WITHOUT pytest - just pure Python. 
Works with any Python 3.7+ installation.

Usage:
    python test_orchestration_standalone.py
    python test_orchestration_standalone.py --verbose
"""

import os
import sys
import time
import uuid
import json
import threading
import traceback
from datetime import datetime
from dataclasses import dataclass
from typing import List, Callable, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Add backend to path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

# =============================================================================
# TEST FRAMEWORK (Minimal, no dependencies)
# =============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    error: Optional[str] = None
    traceback: Optional[str] = None


class TestRunner:
    """Simple test runner without external dependencies."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[TestResult] = []
    
    def log(self, msg: str):
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}")
    
    def run_test(self, name: str, test_func: Callable) -> TestResult:
        """Run a single test function."""
        self.log(f"Running: {name}")
        start = time.time()
        
        try:
            test_func()
            duration = (time.time() - start) * 1000
            result = TestResult(name=name, passed=True, duration_ms=duration)
            self.log(f"  ✓ PASSED ({duration:.1f}ms)")
        except AssertionError as e:
            duration = (time.time() - start) * 1000
            result = TestResult(
                name=name,
                passed=False,
                duration_ms=duration,
                error=str(e),
                traceback=traceback.format_exc()
            )
            self.log(f"  ✗ FAILED: {e}")
        except Exception as e:
            duration = (time.time() - start) * 1000
            result = TestResult(
                name=name,
                passed=False,
                duration_ms=duration,
                error=f"{type(e).__name__}: {e}",
                traceback=traceback.format_exc()
            )
            self.log(f"  ✗ ERROR: {type(e).__name__}: {e}")
        
        self.results.append(result)
        return result
    
    def summary(self):
        """Print test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        print()
        print("=" * 60)
        print(f" TEST SUMMARY: {passed}/{total} passed, {failed} failed")
        print("=" * 60)
        
        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.error}")
                    if self.verbose and r.traceback:
                        for line in r.traceback.split('\n')[-5:]:
                            if line.strip():
                                print(f"      {line}")
        
        return failed == 0


# =============================================================================
# TESTS
# =============================================================================

def test_structured_logger_import():
    """Test that structured logger can be imported."""
    from observability.structured_logger import StructuredLogger
    logger = StructuredLogger("Test")
    assert logger.name == "Test"


def test_structured_logger_correlation_context():
    """Test correlation context management."""
    from observability.structured_logger import (
        set_correlation_context,
        get_correlation_id,
        get_session_id,
        clear_correlation_context
    )
    
    set_correlation_context("corr-123", "sess-456", "inst-789")
    
    assert get_correlation_id() == "corr-123"
    assert get_session_id() == "sess-456"
    
    clear_correlation_context()
    assert get_correlation_id() == ""


def test_structured_logger_log_entry():
    """Test log entry structure."""
    from observability.structured_logger import (
        StructuredLogger,
        set_correlation_context
    )
    
    set_correlation_context("corr-test")
    logger = StructuredLogger("TestEntry")
    
    entry = logger._build_log_entry("INFO", "Test message", custom="value")
    
    assert entry["level"] == "INFO"
    assert entry["message"] == "Test message"
    assert entry["correlation_id"] == "corr-test"
    assert entry["custom"] == "value"
    assert "timestamp" in entry


def test_distributed_lock_import():
    """Test distributed lock can be imported."""
    from concurrency.distributed_lock import (
        InMemoryDistributedLock,
        create_distributed_lock
    )
    
    lock = InMemoryDistributedLock()
    assert lock is not None


def test_distributed_lock_acquire_release():
    """Test basic lock acquire and release."""
    from concurrency.distributed_lock import InMemoryDistributedLock
    
    lock = InMemoryDistributedLock()
    acquired = False
    
    with lock.acquire("test-lock", timeout=5):
        acquired = True
    
    assert acquired, "Lock should have been acquired"


def test_distributed_lock_prevents_concurrent():
    """Test lock prevents concurrent access."""
    from concurrency.distributed_lock import InMemoryDistributedLock
    
    lock = InMemoryDistributedLock()
    sequence = []
    
    def worker(worker_id: int):
        with lock.acquire("shared-lock", timeout=3):
            sequence.append(f"{worker_id}-start")
            time.sleep(0.05)
            sequence.append(f"{worker_id}-end")
    
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Verify sequential execution (no interleaving)
    for i in range(0, len(sequence), 2):
        if i + 1 < len(sequence):
            start_id = sequence[i].split("-")[0]
            end_id = sequence[i + 1].split("-")[0]
            assert start_id == end_id, f"Interleaving detected: {sequence}"


def test_optimistic_lock_import():
    """Test optimistic lock can be imported."""
    from concurrency.optimistic_lock import (
        InMemoryOptimisticLockManager,
        VersionedState
    )
    
    manager = InMemoryOptimisticLockManager()
    assert manager is not None


def test_optimistic_lock_write_read():
    """Test optimistic lock write and read."""
    from concurrency.optimistic_lock import InMemoryOptimisticLockManager
    
    manager = InMemoryOptimisticLockManager()
    
    # Write
    state = manager.write_new("key1", {"value": 42})
    assert state.version == 1
    assert state.data["value"] == 42
    
    # Read
    read_state = manager.read_with_version("key1")
    assert read_state.version == 1
    assert read_state.data["value"] == 42


def test_optimistic_lock_conflict_detection():
    """Test optimistic lock detects conflicts."""
    from concurrency.optimistic_lock import InMemoryOptimisticLockManager
    
    manager = InMemoryOptimisticLockManager()
    
    # Create initial
    manager.write_new("conflict-key", {"counter": 0})
    
    # Two concurrent readers
    reader1 = manager.read_with_version("conflict-key")
    reader2 = manager.read_with_version("conflict-key")
    
    # First update succeeds
    success1 = manager.write_if_unchanged("conflict-key", reader1, {"counter": 1})
    assert success1, "First update should succeed"
    
    # Second update fails (stale version)
    success2 = manager.write_if_unchanged("conflict-key", reader2, {"counter": 2})
    assert not success2, "Second update should fail"
    
    # Final value should be 1
    final = manager.read_with_version("conflict-key")
    assert final.data["counter"] == 1


def test_state_storage_import():
    """Test state storage can be imported."""
    from agentic.state_storage import (
        InMemoryStateStorage,
        SessionState,
        InstanceState
    )
    
    storage = InMemoryStateStorage()
    assert storage is not None


def test_state_storage_session_crud():
    """Test session CRUD operations."""
    from agentic.state_storage import (
        InMemoryStateStorage,
        SessionState
    )
    
    storage = InMemoryStateStorage()
    
    # Create
    session = SessionState(
        session_id="sess-123",
        user_id="user-abc",
        intent="solution",
        locked=True
    )
    storage.save_session("sess-123", session)
    
    # Read
    retrieved = storage.get_session("sess-123")
    assert retrieved is not None
    assert retrieved.user_id == "user-abc"
    assert retrieved.intent == "solution"
    
    # Update
    retrieved.intent = "question"
    storage.save_session("sess-123", retrieved)
    updated = storage.get_session("sess-123")
    assert updated.intent == "question"
    
    # Delete
    storage.delete_session("sess-123")
    assert storage.get_session("sess-123") is None


def test_state_storage_instance_parent_child():
    """Test instance parent-child relationships."""
    from agentic.state_storage import (
        InMemoryStateStorage,
        InstanceState
    )
    
    storage = InMemoryStateStorage()
    
    # Create parent
    parent = InstanceState(
        instance_id="parent-1",
        session_id="sess-123",
        workflow_type="solution",
        intent="solution"
    )
    storage.save_instance(parent)
    
    # Create child
    child = InstanceState(
        instance_id="child-1",
        session_id="sess-123",
        workflow_type="product_search",
        parent_id="parent-1",
        trigger_source="flow_sensor",
        intent="solution"
    )
    storage.save_instance(child)
    
    # Verify parent has child
    updated_parent = storage.get_instance("parent-1")
    assert "child-1" in updated_parent.children
    
    # Get children
    children = storage.get_children("parent-1")
    assert len(children) == 1
    assert children[0].instance_id == "child-1"


def test_state_storage_trigger_dedup():
    """Test trigger-based deduplication."""
    from agentic.state_storage import (
        InMemoryStateStorage,
        InstanceState
    )
    
    storage = InMemoryStateStorage()
    
    # Create instance with trigger
    inst = InstanceState(
        instance_id="inst-1",
        session_id="sess-123",
        workflow_type="product_search",
        trigger_source="temperature_sensor"
    )
    storage.save_instance(inst)
    
    # Lookup by trigger
    found = storage.get_instance_by_trigger(
        "sess-123",
        "product_search",
        "temperature_sensor"
    )
    
    assert found is not None
    assert found.instance_id == "inst-1"


def test_router_agent_import():
    """Test router agent can be imported."""
    from agentic.router_agent import (
        RouterAgent,
        WorkflowType,
        RoutingResult
    )
    
    router = RouterAgent(use_memory_fallback=True)
    assert router is not None


def test_router_agent_route():
    """Test basic routing."""
    from agentic.router_agent import RouterAgent, WorkflowType
    
    router = RouterAgent(use_memory_fallback=True)
    
    result = router.route(
        session_id=f"test-{uuid.uuid4().hex[:8]}",
        query="I need a pressure sensor",
        user_id="user-abc"
    )
    
    assert result is not None
    assert result.is_new_session == True
    assert result.workflow_type in WorkflowType


def test_router_agent_session_locking():
    """Test session locking after classification."""
    from agentic.router_agent import RouterAgent, WorkflowType
    
    router = RouterAgent(use_memory_fallback=True)
    session_id = f"lock-{uuid.uuid4().hex[:8]}"
    
    # First request - classifies
    result1 = router.route(
        session_id=session_id,
        query="Design a flow measurement system",
        user_id="user-abc"
    )
    
    # If routed to solution, verify session is locked
    if result1.workflow_type == WorkflowType.SOLUTION:
        # Second request should be locked to same workflow
        result2 = router.route(
            session_id=session_id,
            query="What about temperature?",
            user_id="user-abc"
        )
        
        assert result2.is_locked == True
        assert result2.workflow_type == WorkflowType.SOLUTION


def test_router_agent_child_workflow():
    """Test child workflow routing."""
    from agentic.router_agent import RouterAgent, WorkflowType
    from agentic.state_storage import InstanceState, SessionState
    
    router = RouterAgent(use_memory_fallback=True)
    
    # Create parent instance
    parent = InstanceState(
        instance_id="parent-test",
        session_id="sess-test",
        workflow_type="solution",
        intent="solution"
    )
    router.storage.save_instance(parent)
    
    # Create session
    session = SessionState(
        session_id="sess-test",
        user_id="user-abc",
        intent="solution"
    )
    router.storage.save_session("sess-test", session)
    
    # Route child
    child_result = router.route_child(
        parent_instance_id="parent-test",
        workflow_type=WorkflowType.PRODUCT_SEARCH,
        trigger_source="flow_transmitter"
    )
    
    assert child_result.parent_instance_id == "parent-test"
    assert child_result.workflow_type == WorkflowType.PRODUCT_SEARCH
    assert child_result.intent == "solution"  # Inherited


def test_router_agent_unlock():
    """Test session unlock."""
    from agentic.router_agent import RouterAgent
    
    router = RouterAgent(use_memory_fallback=True)
    session_id = f"unlock-{uuid.uuid4().hex[:8]}"
    
    # Route and lock
    router.route(session_id=session_id, query="Design a solution", user_id="user")
    
    # Unlock
    router.unlock_session(session_id)
    
    # Verify unlocked
    session = router.storage.get_session(session_id)
    assert session is None or session.locked == False


def test_multi_user_parallel():
    """Test multiple users routing in parallel."""
    from agentic.router_agent import RouterAgent
    
    router = RouterAgent(use_memory_fallback=True)
    results = []
    errors = []
    
    def user_workflow(user_id: str, session_id: str, query: str):
        try:
            result = router.route(
                session_id=session_id,
                query=query,
                user_id=user_id
            )
            results.append({"user": user_id, "workflow": result.workflow_type.value})
        except Exception as e:
            errors.append({"user": user_id, "error": str(e)})
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                user_workflow,
                f"user-{i}",
                f"session-{i}-{uuid.uuid4().hex[:8]}",
                f"Query from user {i}"
            )
            for i in range(5)
        ]
        for f in futures:
            f.result()
    
    assert len(errors) == 0, f"Errors: {errors}"
    assert len(results) == 5


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", default=True)
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    print()
    print("=" * 60)
    print(" ORCHESTRATION IMPLEMENTATION - STANDALONE TESTS")
    print("=" * 60)
    print(f" Time: {datetime.now().isoformat()}")
    print(f" Python: {sys.version}")
    print(f" Verbose: {verbose}")
    print("=" * 60)
    print()
    
    runner = TestRunner(verbose=verbose)
    
    # All tests to run
    tests = [
        ("Logger: Import", test_structured_logger_import),
        ("Logger: Correlation Context", test_structured_logger_correlation_context),
        ("Logger: Log Entry", test_structured_logger_log_entry),
        ("Distributed Lock: Import", test_distributed_lock_import),
        ("Distributed Lock: Acquire/Release", test_distributed_lock_acquire_release),
        ("Distributed Lock: Concurrent", test_distributed_lock_prevents_concurrent),
        ("Optimistic Lock: Import", test_optimistic_lock_import),
        ("Optimistic Lock: Write/Read", test_optimistic_lock_write_read),
        ("Optimistic Lock: Conflict Detection", test_optimistic_lock_conflict_detection),
        ("State Storage: Import", test_state_storage_import),
        ("State Storage: Session CRUD", test_state_storage_session_crud),
        ("State Storage: Parent-Child", test_state_storage_instance_parent_child),
        ("State Storage: Trigger Dedup", test_state_storage_trigger_dedup),
        ("Router Agent: Import", test_router_agent_import),
        ("Router Agent: Route", test_router_agent_route),
        ("Router Agent: Session Locking", test_router_agent_session_locking),
        ("Router Agent: Child Workflow", test_router_agent_child_workflow),
        ("Router Agent: Unlock", test_router_agent_unlock),
        ("Multi-User: Parallel", test_multi_user_parallel),
    ]
    
    for name, test_func in tests:
        runner.run_test(name, test_func)
    
    success = runner.summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
