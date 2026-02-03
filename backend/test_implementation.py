#!/usr/bin/env python
"""
Quick test of the distributed enrichment implementation
"""

import sys
import time
from datetime import datetime

print("=" * 70)
print("TESTING DISTRIBUTED ENRICHMENT IMPLEMENTATION")
print("=" * 70)

# Test 1: State Manager Import and Basic Operations
print("\n[TEST 1] State Manager")
print("-" * 70)

try:
    from agentic.deep_agent.state_manager import InMemoryStateManager

    manager = InMemoryStateManager()
    print("[PASS] InMemoryStateManager imported and initialized")

    # Create session
    manager.create_session("test-session-1", {"total_items": 2})
    print("[PASS] Session created")

    # Update item state
    manager.update_item_state("test-session-1", "item-1", {
        "specifications": {"temp_range": "0-100C"},
        "spec_count": 1
    })
    print("[PASS] Item state updated")

    # Retrieve state
    item_state = manager.get_item_state("test-session-1", "item-1")
    assert item_state is not None
    assert item_state["spec_count"] == 1
    print("[PASS] Item state retrieved correctly")

    # Mark complete
    manager.mark_item_complete("test-session-1", "item-1", {"temp": "100C"})
    manager.update_item_state("test-session-1", "item-2", {"spec_count": 0})
    manager.mark_item_complete("test-session-1", "item-2", {"temp": "50C"})

    # Check completion
    is_complete = manager.is_session_complete("test-session-1")
    assert is_complete, "Session should be complete"
    print("[PASS] Session completion tracking works")

    print("\n[SUMMARY] State Manager: ALL TESTS PASSED")

except Exception as e:
    print(f"[FAIL] State Manager Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Task Queue Import and Basic Operations
print("\n[TEST 2] Task Queue")
print("-" * 70)

try:
    from agentic.deep_agent.task_queue import InMemoryTaskQueue, EnrichmentTask

    queue = InMemoryTaskQueue()
    print("[PASS] InMemoryTaskQueue imported and initialized")

    # Create and push task
    task = {
        "task_id": "task-1",
        "task_type": "initial_enrichment",
        "session_id": "test-session",
        "item_id": "item-1",
        "payload": {"user_requirement": "Test"},
        "priority": 1,
        "created_at": datetime.utcnow().isoformat(),
        "retry_count": 0
    }

    queue.push_task(task)
    print("[PASS] Task pushed to queue")
    assert queue.get_queue_depth() == 1, "Queue depth should be 1"
    print("[PASS] Queue depth tracked correctly")

    # Pop task
    popped_task = queue.pop_task(timeout=1)
    assert popped_task is not None, "Should get task from queue"
    assert popped_task["task_id"] == "task-1", "Should be correct task"
    print("[PASS] Task popped from queue")
    assert queue.get_processing_count() == 1, "Should be in processing"
    print("[PASS] Task in processing state")

    # ACK task
    queue.ack_task("task-1")
    assert queue.get_processing_count() == 0, "Should not be processing"
    print("[PASS] Task acknowledged and removed from processing")

    # Test priority ordering
    queue.push_task({**task, "task_id": "low-priority", "priority": 10})
    queue.push_task({**task, "task_id": "high-priority", "priority": 1})

    high = queue.pop_task(timeout=1)
    assert high["task_id"] == "high-priority", "Should pop high priority first"
    print("[PASS] Priority queue ordering works")

    print("\n[SUMMARY] Task Queue: ALL TESTS PASSED")

except Exception as e:
    print(f"[FAIL] Task Queue Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Enrichment Worker Import
print("\n[TEST 3] Enrichment Worker")
print("-" * 70)

try:
    from agentic.deep_agent.enrichment_worker import EnrichmentWorker
    from agentic.deep_agent.state_manager import InMemoryStateManager
    from agentic.deep_agent.task_queue import InMemoryTaskQueue

    state_mgr = InMemoryStateManager()
    task_q = InMemoryTaskQueue()

    worker = EnrichmentWorker(task_q, state_mgr, "test-worker")
    print("[PASS] EnrichmentWorker imported and initialized")

    assert worker.worker_id == "test-worker", "Worker ID should be set"
    print("[PASS] Worker ID set correctly")

    assert not worker.running, "Worker should not be running initially"
    print("[PASS] Worker state initialized correctly")

    print("\n[SUMMARY] Enrichment Worker: ALL TESTS PASSED")

except Exception as e:
    print(f"[FAIL] Enrichment Worker Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Orchestrator Import
print("\n[TEST 4] Stateless Orchestrator")
print("-" * 70)

try:
    from agentic.deep_agent.stateless_orchestrator import StatelessEnrichmentOrchestrator
    from agentic.deep_agent.state_manager import InMemoryStateManager
    from agentic.deep_agent.task_queue import InMemoryTaskQueue

    state_mgr = InMemoryStateManager()
    task_q = InMemoryTaskQueue()

    orchestrator = StatelessEnrichmentOrchestrator(state_mgr, task_q)
    print("[PASS] StatelessEnrichmentOrchestrator imported and initialized")

    # Start session
    items = [
        {"name": "Test Item 1", "category": "Test"},
        {"name": "Test Item 2", "category": "Test"}
    ]

    session_id = orchestrator.start_enrichment_session(items, session_id="test-session")
    assert session_id == "test-session", "Session ID should match"
    print("[PASS] Session started successfully")

    # Check status
    status = orchestrator.get_session_status("test-session")
    assert status["total_items"] == 2, "Should have 2 items"
    assert status["completed_items"] == 0, "Should have no completed items yet"
    assert not status["is_complete"], "Should not be complete"
    print("[PASS] Session status reported correctly")

    # Check queue has initial tasks
    assert task_q.get_queue_depth() == 2, "Should have 2 initial tasks"
    print("[PASS] Initial tasks pushed to queue")

    print("\n[SUMMARY] Orchestrator: ALL TESTS PASSED")

except Exception as e:
    print(f"[FAIL] Orchestrator Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Compatibility Wrapper
print("\n[TEST 5] Compatibility Wrapper")
print("-" * 70)

try:
    from agentic.deep_agent.compatibility_wrapper import is_distributed_enabled

    enabled = is_distributed_enabled()
    print(f"[PASS] Compatibility wrapper imported")
    print(f"  - Distributed mode: {enabled}")

    print("\n[SUMMARY] Compatibility Wrapper: ALL TESTS PASSED")

except Exception as e:
    print(f"[FAIL] Compatibility Wrapper Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final Summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print("""
Test Results:
  [PASS] State Manager (4/4 subtests)
  [PASS] Task Queue (6/6 subtests)
  [PASS] Enrichment Worker (3/3 subtests)
  [PASS] Orchestrator (3/3 subtests)
  [PASS] Compatibility Wrapper (1/1 subtests)

Total: 17 subtests PASSED
Status: READY FOR INTEGRATION

Next Steps:
  1. Run full pytest suite: pytest tests/test_state_manager.py tests/test_task_queue.py -v
  2. Set up Redis: docker run -d -p 6379:6379 redis:7-alpine
  3. Test with Redis backend
  4. Run performance benchmarks
""")
print("=" * 70)
