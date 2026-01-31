"""
Test Suite for Production-Grade Orchestration Implementation

Tests:
- Structured Logger (observability)
- Distributed Lock (concurrency)
- Optimistic Lock (concurrency)
- State Storage (persistence)
- Router Agent (routing)

Run with debug: python -m pytest tests/test_orchestration_impl.py -v -s --tb=long
"""

import pytest
import time
import uuid
import json
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Debug flag - set to True for verbose output
DEBUG = os.environ.get("DEBUG_TESTS", "1") == "1"

def debug_print(message: str, **kwargs):
    """Print debug message if DEBUG is enabled."""
    if DEBUG:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        extra = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"[{timestamp}] [DEBUG] {message}" + (f" | {extra}" if extra else ""))


# =============================================================================
# TEST: STRUCTURED LOGGER
# =============================================================================

class TestStructuredLogger:
    """Tests for the structured logging system."""
    
    def test_logger_creation(self):
        """Test logger can be created."""
        debug_print("Testing logger creation")
        
        from observability.structured_logger import StructuredLogger
        
        logger = StructuredLogger("TestComponent", service="test-service")
        assert logger.name == "TestComponent"
        assert logger.service == "test-service"
        
        debug_print("Logger created successfully", name=logger.name)
    
    def test_correlation_context(self):
        """Test correlation context is set and retrieved correctly."""
        debug_print("Testing correlation context")
        
        from observability.structured_logger import (
            set_correlation_context,
            get_correlation_id,
            get_session_id,
            get_instance_id,
            clear_correlation_context
        )
        
        # Set context
        set_correlation_context(
            correlation_id="test-corr-123",
            session_id="session-abc",
            instance_id="inst-xyz"
        )
        
        # Verify retrieval
        assert get_correlation_id() == "test-corr-123"
        assert get_session_id() == "session-abc"
        assert get_instance_id() == "inst-xyz"
        
        debug_print(
            "Context set and retrieved",
            corr_id=get_correlation_id(),
            session=get_session_id()
        )
        
        # Clear and verify
        clear_correlation_context()
        assert get_correlation_id() == ""
        
        debug_print("Context cleared successfully")
    
    def test_log_entry_format(self):
        """Test log entry has correct JSON format."""
        debug_print("Testing log entry format")
        
        from observability.structured_logger import (
            StructuredLogger,
            set_correlation_context
        )
        
        set_correlation_context("corr-123", "sess-456")
        logger = StructuredLogger("TestFormat")
        
        # Build entry
        entry = logger._build_log_entry(
            "INFO",
            "Test message",
            custom_field="custom_value"
        )
        
        # Verify structure
        assert "timestamp" in entry
        assert entry["level"] == "INFO"
        assert entry["message"] == "Test message"
        assert entry["correlation_id"] == "corr-123"
        assert entry["session_id"] == "sess-456"
        assert entry["custom_field"] == "custom_value"
        
        debug_print("Entry format correct", entry=json.dumps(entry)[:100])
    
    def test_generate_correlation_id(self):
        """Test correlation ID generation."""
        debug_print("Testing correlation ID generation")
        
        from observability.structured_logger import generate_correlation_id
        
        id1 = generate_correlation_id()
        id2 = generate_correlation_id()
        
        assert id1.startswith("req-")
        assert id1 != id2
        
        debug_print("Generated IDs", id1=id1, id2=id2)


# =============================================================================
# TEST: DISTRIBUTED LOCK (In-Memory)
# =============================================================================

class TestDistributedLock:
    """Tests for the distributed locking system (in-memory mode)."""
    
    def test_lock_acquire_release(self):
        """Test basic lock acquire and release."""
        debug_print("Testing lock acquire/release")
        
        from concurrency.distributed_lock import InMemoryDistributedLock
        
        lock = InMemoryDistributedLock()
        acquired = False
        
        with lock.acquire("test-lock-1", timeout=5):
            acquired = True
            debug_print("Lock acquired", lock_name="test-lock-1")
        
        assert acquired
        debug_print("Lock released successfully")
    
    def test_lock_prevents_concurrent_access(self):
        """Test that lock prevents concurrent access."""
        debug_print("Testing lock prevents concurrent access")
        
        from concurrency.distributed_lock import InMemoryDistributedLock
        
        lock = InMemoryDistributedLock()
        results = []
        errors = []
        
        def worker(worker_id: int):
            try:
                with lock.acquire("shared-lock", timeout=2):
                    results.append(f"worker-{worker_id}-start")
                    time.sleep(0.1)  # Simulate work
                    results.append(f"worker-{worker_id}-end")
            except TimeoutError as e:
                errors.append(str(e))
        
        # Run concurrent workers
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        
        debug_print(
            "Concurrent test complete",
            results=results,
            errors=errors
        )
        
        # Verify sequential execution (no interleaving)
        # Each worker's start should be followed by its end
        for i in range(0, len(results), 2):
            if i + 1 < len(results):
                start = results[i]
                end = results[i + 1]
                worker_id = start.split("-")[1]
                assert end == f"worker-{worker_id}-end"
    
    def test_lock_timeout(self):
        """Test lock timeout when already held."""
        debug_print("Testing lock timeout")
        
        from concurrency.distributed_lock import InMemoryDistributedLock
        
        lock = InMemoryDistributedLock()
        timeout_occurred = False
        
        def holder():
            with lock.acquire("held-lock", timeout=5):
                debug_print("Lock holder acquired lock")
                time.sleep(2)  # Hold for 2 seconds
        
        def waiter():
            nonlocal timeout_occurred
            time.sleep(0.1)  # Let holder grab first
            try:
                with lock.acquire("held-lock", timeout=0.5):
                    pass
            except TimeoutError:
                timeout_occurred = True
                debug_print("Waiter timed out as expected")
        
        holder_thread = threading.Thread(target=holder)
        waiter_thread = threading.Thread(target=waiter)
        
        holder_thread.start()
        waiter_thread.start()
        
        holder_thread.join(timeout=5)
        waiter_thread.join(timeout=5)
        
        assert timeout_occurred, "Waiter should have timed out"


# =============================================================================
# TEST: OPTIMISTIC LOCK (In-Memory)
# =============================================================================

class TestOptimisticLock:
    """Tests for the optimistic locking system."""
    
    def test_write_and_read(self):
        """Test basic write and read."""
        debug_print("Testing optimistic lock write/read")
        
        from concurrency.optimistic_lock import InMemoryOptimisticLockManager
        
        manager = InMemoryOptimisticLockManager()
        
        # Write new
        state = manager.write_new("test-key-1", {"value": 42})
        assert state.version == 1
        assert state.data["value"] == 42
        
        debug_print("Written state", key=state.key, version=state.version)
        
        # Read back
        read_state = manager.read_with_version("test-key-1")
        assert read_state.data["value"] == 42
        assert read_state.version == 1
        
        debug_print("Read state", version=read_state.version)
    
    def test_optimistic_conflict_detection(self):
        """Test that concurrent modifications are detected."""
        debug_print("Testing optimistic conflict detection")
        
        from concurrency.optimistic_lock import InMemoryOptimisticLockManager
        
        manager = InMemoryOptimisticLockManager()
        
        # Create initial state
        manager.write_new("conflict-key", {"counter": 0})
        
        # Two readers get the same version
        reader1_state = manager.read_with_version("conflict-key")
        reader2_state = manager.read_with_version("conflict-key")
        
        debug_print(
            "Both readers got version",
            v1=reader1_state.version,
            v2=reader2_state.version
        )
        
        # Reader 1 updates successfully
        success1 = manager.write_if_unchanged(
            "conflict-key",
            reader1_state,
            {"counter": 1}
        )
        assert success1, "First update should succeed"
        debug_print("Reader 1 update succeeded")
        
        # Reader 2 tries to update with stale version
        success2 = manager.write_if_unchanged(
            "conflict-key",
            reader2_state,
            {"counter": 2}
        )
        assert not success2, "Second update should fail (stale version)"
        debug_print("Reader 2 update correctly rejected")
        
        # Verify final value
        final = manager.read_with_version("conflict-key")
        assert final.data["counter"] == 1
        assert final.version == 2


# =============================================================================
# TEST: STATE STORAGE (In-Memory)
# =============================================================================

class TestStateStorage:
    """Tests for the state storage system."""
    
    def test_session_crud(self):
        """Test session create, read, update, delete."""
        debug_print("Testing session CRUD operations")
        
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
        debug_print("Session created", session_id="sess-123")
        
        # Read
        retrieved = storage.get_session("sess-123")
        assert retrieved is not None
        assert retrieved.user_id == "user-abc"
        assert retrieved.intent == "solution"
        assert retrieved.locked == True
        debug_print("Session retrieved", intent=retrieved.intent)
        
        # Update
        retrieved.intent = "question"
        storage.save_session("sess-123", retrieved)
        updated = storage.get_session("sess-123")
        assert updated.intent == "question"
        debug_print("Session updated", new_intent=updated.intent)
        
        # Delete
        storage.delete_session("sess-123")
        deleted = storage.get_session("sess-123")
        assert deleted is None
        debug_print("Session deleted")
    
    def test_instance_with_parent(self):
        """Test instance creation with parent-child relationship."""
        debug_print("Testing instance parent-child relationship")
        
        from agentic.state_storage import (
            InMemoryStateStorage,
            InstanceState
        )
        
        storage = InMemoryStateStorage()
        
        # Create parent instance
        parent = InstanceState(
            instance_id="inst-parent",
            session_id="sess-123",
            workflow_type="solution",
            intent="solution"
        )
        storage.save_instance(parent)
        debug_print("Parent instance created", id=parent.instance_id)
        
        # Create child instance
        child = InstanceState(
            instance_id="inst-child-1",
            session_id="sess-123",
            workflow_type="product_search",
            parent_id="inst-parent",
            trigger_source="flow_transmitter",
            intent="solution"  # Inherited
        )
        storage.save_instance(child)
        debug_print(
            "Child instance created",
            id=child.instance_id,
            parent_id=child.parent_id
        )
        
        # Verify parent has child in list
        updated_parent = storage.get_instance("inst-parent")
        assert "inst-child-1" in updated_parent.children
        debug_print("Parent children list", children=updated_parent.children)
        
        # Get children
        children = storage.get_children("inst-parent")
        assert len(children) == 1
        assert children[0].instance_id == "inst-child-1"
    
    def test_trigger_deduplication(self):
        """Test instance deduplication by trigger source."""
        debug_print("Testing trigger-based deduplication")
        
        from agentic.state_storage import (
            InMemoryStateStorage,
            InstanceState
        )
        
        storage = InMemoryStateStorage()
        
        # Create first instance with trigger
        inst1 = InstanceState(
            instance_id="inst-1",
            session_id="sess-123",
            workflow_type="product_search",
            trigger_source="temperature_sensor"
        )
        storage.save_instance(inst1)
        
        # Look up by trigger
        found = storage.get_instance_by_trigger(
            "sess-123",
            "product_search",
            "temperature_sensor"
        )
        
        assert found is not None
        assert found.instance_id == "inst-1"
        debug_print("Deduplication lookup succeeded", found_id=found.instance_id)


# =============================================================================
# TEST: ROUTER AGENT (In-Memory)
# =============================================================================

class TestRouterAgent:
    """Tests for the router agent."""
    
    def test_route_new_session(self):
        """Test routing creates new session."""
        debug_print("Testing route with new session")
        
        from agentic.router_agent import RouterAgent, WorkflowType
        
        router = RouterAgent(use_memory_fallback=True)
        
        result = router.route(
            session_id="new-sess-123",
            query="I need a pressure sensor for my application",
            user_id="user-abc"
        )
        
        assert result.is_new_session == True
        assert result.session_id == "new-sess-123"
        assert result.workflow_type in WorkflowType
        
        debug_print(
            "Route result",
            workflow=result.workflow_type.value,
            is_new=result.is_new_session,
            intent=result.intent
        )
    
    def test_session_locking(self):
        """Test that session gets locked after first classification."""
        debug_print("Testing session locking")
        
        from agentic.router_agent import RouterAgent, WorkflowType
        
        router = RouterAgent(use_memory_fallback=True)
        session_id = f"lock-test-{uuid.uuid4().hex[:8]}"
        
        # First request - classifies and locks
        result1 = router.route(
            session_id=session_id,
            query="Design a flow measurement system for my plant",
            user_id="user-abc"
        )
        
        debug_print(
            "First request",
            workflow=result1.workflow_type.value,
            locked=result1.is_locked
        )
        
        # Second request - should be locked
        result2 = router.route(
            session_id=session_id,
            query="Add temperature monitoring",
            user_id="user-abc"
        )
        
        debug_print(
            "Second request",
            workflow=result2.workflow_type.value,
            locked=result2.is_locked,
            same_workflow=(result1.workflow_type == result2.workflow_type)
        )
        
        # If first was solution, second should also be solution (locked)
        if result1.workflow_type == WorkflowType.SOLUTION:
            assert result2.workflow_type == WorkflowType.SOLUTION
            assert result2.is_locked == True
    
    def test_child_workflow_routing(self):
        """Test child workflow routing with parent linkage."""
        debug_print("Testing child workflow routing")
        
        from agentic.router_agent import RouterAgent, WorkflowType
        from agentic.state_storage import InstanceState
        
        router = RouterAgent(use_memory_fallback=True)
        
        # Create parent instance
        parent = InstanceState(
            instance_id="parent-inst-001",
            session_id="sess-abc",
            workflow_type="solution",
            intent="solution"
        )
        router.storage.save_instance(parent)
        
        # Also create session
        from agentic.state_storage import SessionState
        session = SessionState(
            session_id="sess-abc",
            user_id="user-abc",
            intent="solution",
            locked=True
        )
        router.storage.save_session("sess-abc", session)
        
        # Route child
        child_result = router.route_child(
            parent_instance_id="parent-inst-001",
            workflow_type=WorkflowType.PRODUCT_SEARCH,
            trigger_source="flow_transmitter"
        )
        
        assert child_result.parent_instance_id == "parent-inst-001"
        assert child_result.workflow_type == WorkflowType.PRODUCT_SEARCH
        assert child_result.intent == "solution"  # Inherited
        assert child_result.instance_id is not None
        
        debug_print(
            "Child routed",
            child_id=child_result.instance_id,
            parent_id=child_result.parent_instance_id,
            inherited_intent=child_result.intent
        )
    
    def test_session_unlock(self):
        """Test session can be unlocked."""
        debug_print("Testing session unlock")
        
        from agentic.router_agent import RouterAgent
        
        router = RouterAgent(use_memory_fallback=True)
        session_id = f"unlock-test-{uuid.uuid4().hex[:8]}"
        
        # Route and lock
        router.route(
            session_id=session_id,
            query="Design a solution",
            user_id="user-abc"
        )
        
        # Verify locked
        session = router.storage.get_session(session_id)
        was_locked = session.locked if session else False
        debug_print("Before unlock", locked=was_locked)
        
        # Unlock
        router.unlock_session(session_id)
        
        # Verify unlocked
        session = router.storage.get_session(session_id)
        is_unlocked = not session.locked if session else True
        debug_print("After unlock", locked=not is_unlocked)
        
        assert is_unlocked


# =============================================================================
# TEST: MULTI-USER CONCURRENT ROUTING
# =============================================================================

class TestMultiUserConcurrency:
    """Tests for multi-user concurrent operations."""
    
    def test_multiple_users_parallel(self):
        """Test multiple users can route in parallel without conflict."""
        debug_print("Testing multi-user parallel routing")
        
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
                results.append({
                    "user_id": user_id,
                    "session_id": session_id,
                    "workflow": result.workflow_type.value,
                    "instance_id": result.instance_id,
                    "is_new": result.is_new_session
                })
            except Exception as e:
                errors.append({"user_id": user_id, "error": str(e)})
        
        # Simulate 5 concurrent users
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
            
            for future in as_completed(futures):
                future.result()  # Propagate exceptions
        
        debug_print(
            "Multi-user test complete",
            num_results=len(results),
            num_errors=len(errors)
        )
        
        # Log each result
        for r in results:
            debug_print(f"  User {r['user_id']}: {r['workflow']}")
        
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Enable debug mode
    os.environ["DEBUG_TESTS"] = "1"
    
    print("=" * 70)
    print("ORCHESTRATION IMPLEMENTATION TEST SUITE")
    print("=" * 70)
    print(f"Debug Mode: {DEBUG}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)
    print()
    
    # Run with pytest
    exit_code = pytest.main([
        __file__,
        "-v",           # Verbose
        "-s",           # Show print statements
        "--tb=long",    # Long traceback
        "-x",           # Stop on first failure
    ])
    
    sys.exit(exit_code)
