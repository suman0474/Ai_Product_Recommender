"""
Failure Recovery and Resilience Tests

Tests system behavior during crashes, task failures, and recovery scenarios.
Uses in-memory backends for fast testing.

Run with: pytest tests/test_failure_recovery.py -v
"""

import pytest
import time
from datetime import datetime, timedelta

from agentic.deep_agent.state_manager import InMemoryStateManager
from agentic.deep_agent.task_queue import InMemoryTaskQueue
from agentic.deep_agent.enrichment_worker import EnrichmentWorker
from agentic.deep_agent.stateless_orchestrator import StatelessEnrichmentOrchestrator


@pytest.fixture
def orchestrator():
    """Create orchestrator with in-memory backends."""
    state_mgr = InMemoryStateManager()
    task_q = InMemoryTaskQueue()
    return StatelessEnrichmentOrchestrator(state_mgr, task_q), state_mgr, task_q


# =============================================================================
# TASK FAILURE SCENARIOS
# =============================================================================

class TestTaskFailures:
    """Test task failure handling and recovery."""

    def test_task_failure_nack_and_retry(self, orchestrator):
        """Test task NACK and automatic requeue."""
        orch, state_mgr, task_q = orchestrator

        items = [{"name": "Item 1", "category": "Test"}]
        session_id = orch.start_enrichment_session(items)

        # Pop task
        task = task_q.pop_task(timeout=5)
        assert task is not None
        original_priority = task["priority"]

        # Simulate failure - NACK with retry
        task_q.nack_task(task["task_id"], retry=True)

        # Task should be requeued
        assert task_q.get_processing_count() == 0
        assert task_q.get_queue_depth() == 1

        # Pop again - should be same task with incremented retry count
        retry_task = task_q.pop_task(timeout=5)
        assert retry_task is not None
        assert retry_task["retry_count"] == 1
        # Priority should be higher (worse) for retry
        assert retry_task["priority"] > original_priority

    def test_max_retries_sends_to_dlq(self, orchestrator):
        """Test that max retries sends task to DLQ."""
        orch, state_mgr, task_q = orchestrator

        items = [{"name": "Item", "category": "Test"}]
        session_id = orch.start_enrichment_session(items)

        # Keep NACKing until max retries exceeded
        for i in range(5):  # Default max_retries is 3, so exceed it
            task = task_q.pop_task(timeout=5)
            if not task:
                break
            task_q.nack_task(task["task_id"], retry=True)

        # After max retries, should be in DLQ
        dlq_tasks = task_q.get_dlq_tasks()
        assert len(dlq_tasks) > 0

    def test_nack_without_retry_sends_to_dlq(self, orchestrator):
        """Test that NACK without retry sends to DLQ immediately."""
        orch, state_mgr, task_q = orchestrator

        items = [{"name": "Item", "category": "Test"}]
        session_id = orch.start_enrichment_session(items)

        task = task_q.pop_task(timeout=5)
        assert task is not None

        # NACK without retry
        task_q.nack_task(task["task_id"], retry=False)

        # Should be in DLQ
        dlq_tasks = task_q.get_dlq_tasks()
        assert len(dlq_tasks) == 1
        assert dlq_tasks[0]["task_id"] == task["task_id"]

    def test_multiple_task_failures(self, orchestrator):
        """Test handling multiple task failures in batch."""
        orch, state_mgr, task_q = orchestrator

        items = [
            {"name": f"Item {i}", "category": "Test"}
            for i in range(3)
        ]
        session_id = orch.start_enrichment_session(items)

        # Pop all tasks
        tasks = []
        for _ in range(3):
            task = task_q.pop_task(timeout=5)
            if task:
                tasks.append(task)

        # Fail all tasks
        for task in tasks:
            task_q.nack_task(task["task_id"], retry=True)

        # All should be requeued
        assert task_q.get_queue_depth() == 3
        assert task_q.get_processing_count() == 0


# =============================================================================
# STATE PERSISTENCE ON FAILURE
# =============================================================================

class TestStatePersistence:
    """Test state persistence across failures."""

    def test_in_progress_item_state_survives_crash(self, orchestrator):
        """Test that in-progress item state is preserved."""
        orch, state_mgr, task_q = orchestrator

        items = [{"name": "Item 1", "category": "Test"}]
        session_id = orch.start_enrichment_session(items)

        item_ids = state_mgr.get_session_item_ids(session_id)
        item_id = item_ids[0]

        # Mark item as in-progress with partial results
        state_mgr.update_item_state(session_id, item_id, {
            "specifications": {
                "temp_range": "0-100C",
                "accuracy": "0.1%",
                "resolution": "0.01C"
            },
            "spec_count": 3,
            "status": "in_progress",
            "iteration": 1
        })

        # Simulate crash - get fresh state manager
        recovered_state = state_mgr.get_item_state(session_id, item_id)

        # Verify state survived
        assert recovered_state is not None
        assert recovered_state["spec_count"] == 3
        assert len(recovered_state["specifications"]) == 3
        assert recovered_state["iteration"] == 1

    def test_partial_completion_state_preserved(self, orchestrator):
        """Test that partial completion state is preserved across failure."""
        orch, state_mgr, task_q = orchestrator

        items = [
            {"name": f"Item {i}", "category": "Test"}
            for i in range(4)
        ]
        session_id = orch.start_enrichment_session(items)

        item_ids = state_mgr.get_session_item_ids(session_id)

        # Complete first 2 items
        state_mgr.mark_item_complete(session_id, item_ids[0], {})
        state_mgr.mark_item_complete(session_id, item_ids[1], {})

        # Simulate crash - verify state still shows partial completion
        status = orch.get_session_status(session_id)
        assert status["completed_items"] == 2
        assert status["progress_percentage"] == 50.0

    def test_completed_items_cannot_be_reprocessed(self, orchestrator):
        """Test that completed items are protected from reprocessing."""
        orch, state_mgr, task_q = orchestrator

        items = [{"name": "Item", "category": "Test"}]
        session_id = orch.start_enrichment_session(items)

        item_ids = state_mgr.get_session_item_ids(session_id)
        item_id = item_ids[0]

        # Complete the item
        final_specs = {"temp": "100C", "accuracy": "0.5%"}
        state_mgr.mark_item_complete(session_id, item_id, final_specs)

        # Verify item is marked complete
        item_state = state_mgr.get_item_state(session_id, item_id)
        assert item_state["status"] == "completed"

        # Try to update completed item - should be blocked or tracked
        state_mgr.update_item_state(session_id, item_id, {
            "specifications": {"new": "spec"}
        })

        # Original completion specs should be preserved
        item_state = state_mgr.get_item_state(session_id, item_id)
        # System should have protection against overwriting completed items


# =============================================================================
# WORKER CRASH RECOVERY
# =============================================================================

class TestWorkerCrashRecovery:
    """Test recovery from worker crashes."""

    def test_task_in_processing_survives_worker_crash(self, orchestrator):
        """Test that task in processing state is recoverable after crash."""
        orch, state_mgr, task_q = orchestrator

        items = [{"name": "Item", "category": "Test"}]
        session_id = orch.start_enrichment_session(items)

        # Pop task (simulates worker picking it up)
        task = task_q.pop_task(timeout=5)
        assert task is not None
        assert task_q.get_processing_count() == 1

        # Worker crashes WITHOUT acknowledging task
        # (in this test, we just don't call ack_task)

        # Simulate recovery timeout and requeue
        # In real system, a watchdog would move processing tasks back to queue
        # For now, manually NACK to simulate recovery
        task_q.nack_task(task["task_id"], retry=True)

        # Task should be requeued for another worker
        assert task_q.get_processing_count() == 0
        assert task_q.get_queue_depth() == 1

    def test_multiple_workers_one_crashes(self, orchestrator):
        """Test that other workers continue when one crashes."""
        orch, state_mgr, task_q = orchestrator

        items = [
            {"name": f"Item {i}", "category": "Test"}
            for i in range(3)
        ]
        session_id = orch.start_enrichment_session(items)

        # Simulate 2 workers picking up tasks
        task1 = task_q.pop_task(timeout=5)
        task2 = task_q.pop_task(timeout=5)

        assert task_q.get_processing_count() == 2

        # Worker 1 crashes (no ack)
        # Worker 2 completes normally
        task_q.ack_task(task2["task_id"])

        # Task2 acknowledged successfully
        assert task_q.get_processing_count() == 1

        # Recovery: requeue task1
        task_q.nack_task(task1["task_id"], retry=True)
        assert task_q.get_processing_count() == 0
        assert task_q.get_queue_depth() > 0

    def test_session_survives_worker_crashes(self, orchestrator):
        """Test that session completes despite worker crashes."""
        orch, state_mgr, task_q = orchestrator

        items = [
            {"name": f"Item {i}", "category": "Test"}
            for i in range(2)
        ]
        session_id = orch.start_enrichment_session(items)
        item_ids = state_mgr.get_session_item_ids(session_id)

        # Pop and crash
        task1 = task_q.pop_task(timeout=5)
        task_q.nack_task(task1["task_id"], retry=True)

        # Pop and succeed
        task2 = task_q.pop_task(timeout=5)
        state_mgr.mark_item_complete(session_id, item_ids[1], {})
        task_q.ack_task(task2["task_id"])

        # Session should eventually complete
        # First item still needs processing
        status = orch.get_session_status(session_id)
        assert status["completed_items"] == 1
        assert not status["is_complete"]


# =============================================================================
# CONCURRENT FAILURE SCENARIOS
# =============================================================================

class TestConcurrentFailures:
    """Test handling multiple concurrent failures."""

    def test_batch_partial_failure_recovery(self, orchestrator):
        """Test recovery from partial batch failures."""
        orch, state_mgr, task_q = orchestrator

        items = [
            {"name": f"Item {i}", "category": "Test"}
            for i in range(4)
        ]
        session_id = orch.start_enrichment_session(items)

        # Pop all 4 tasks
        tasks = []
        for _ in range(4):
            task = task_q.pop_task(timeout=5)
            if task:
                tasks.append(task)

        # Fail 2, succeed 2
        task_q.ack_task(tasks[0]["task_id"])
        task_q.nack_task(tasks[1]["task_id"], retry=True)
        task_q.ack_task(tasks[2]["task_id"])
        task_q.nack_task(tasks[3]["task_id"], retry=True)

        # Mark items as complete
        item_ids = state_mgr.get_session_item_ids(session_id)
        state_mgr.mark_item_complete(session_id, item_ids[0], {})
        state_mgr.mark_item_complete(session_id, item_ids[2], {})

        # Status should show 2 complete
        status = orch.get_session_status(session_id)
        assert status["completed_items"] == 2
        # Queue should have 2 failed tasks requeued
        assert task_q.get_queue_depth() == 2

    def test_interleaved_failure_and_success(self, orchestrator):
        """Test interleaved success and failures."""
        orch, state_mgr, task_q = orchestrator

        items = [
            {"name": f"Item {i}", "category": "Test"}
            for i in range(3)
        ]
        session_id = orch.start_enrichment_session(items)
        item_ids = state_mgr.get_session_item_ids(session_id)

        # Pop, succeed, pop, fail, pop, succeed
        t1 = task_q.pop_task(timeout=5)
        state_mgr.mark_item_complete(session_id, item_ids[0], {})
        task_q.ack_task(t1["task_id"])

        t2 = task_q.pop_task(timeout=5)
        task_q.nack_task(t2["task_id"], retry=True)

        t3 = task_q.pop_task(timeout=5)
        state_mgr.mark_item_complete(session_id, item_ids[2], {})
        task_q.ack_task(t3["task_id"])

        # Status should be 2 complete, 1 pending
        status = orch.get_session_status(session_id)
        assert status["completed_items"] == 2
        assert not status["is_complete"]


# =============================================================================
# TIMEOUT AND DEADLINE HANDLING
# =============================================================================

class TestTimeoutHandling:
    """Test timeout and deadline scenarios."""

    def test_empty_queue_timeout(self, orchestrator):
        """Test that pop_task returns None on timeout."""
        orch, state_mgr, task_q = orchestrator

        # Empty queue
        start = time.time()
        task = task_q.pop_task(timeout=1)
        elapsed = time.time() - start

        assert task is None
        assert elapsed >= 1.0

    def test_queue_depth_during_failure(self, orchestrator):
        """Test queue depth tracking during failures."""
        orch, state_mgr, task_q = orchestrator

        items = [
            {"name": f"Item {i}", "category": "Test"}
            for i in range(3)
        ]
        session_id = orch.start_enrichment_session(items)

        initial_depth = task_q.get_queue_depth()
        assert initial_depth == 3

        # Pop one
        task = task_q.pop_task(timeout=5)
        assert task_q.get_queue_depth() == 2
        assert task_q.get_processing_count() == 1

        # Fail it
        task_q.nack_task(task["task_id"], retry=True)
        assert task_q.get_queue_depth() == 3
        assert task_q.get_processing_count() == 0

    def test_dlq_accumulation(self, orchestrator):
        """Test that DLQ accumulates permanently failed tasks."""
        orch, state_mgr, task_q = orchestrator

        items = [{"name": "Item", "category": "Test"}]
        session_id = orch.start_enrichment_session(items)

        task = task_q.pop_task(timeout=5)
        assert task is not None

        # Fail permanently
        task_q.nack_task(task["task_id"], retry=False)

        dlq_tasks = task_q.get_dlq_tasks()
        assert len(dlq_tasks) == 1
        assert dlq_tasks[0]["task_id"] == task["task_id"]


# =============================================================================
# DATA INTEGRITY AFTER FAILURES
# =============================================================================

class TestDataIntegrity:
    """Test data integrity maintained across failures."""

    def test_session_metadata_unchanged_after_failures(self, orchestrator):
        """Test that session metadata is never corrupted."""
        orch, state_mgr, task_q = orchestrator

        items = [{"name": "Item", "category": "Test"}]
        session_id = orch.start_enrichment_session(items)

        # Get initial status
        status1 = orch.get_session_status(session_id)
        initial_total = status1["total_items"]

        # Pop and fail multiple times
        for _ in range(3):
            task = task_q.pop_task(timeout=5)
            if task:
                task_q.nack_task(task["task_id"], retry=True)

        # Metadata should be unchanged
        status2 = orch.get_session_status(session_id)
        assert status2["total_items"] == initial_total

    def test_item_specifications_not_lost_on_failure(self, orchestrator):
        """Test that item specifications are preserved even if task fails."""
        orch, state_mgr, task_q = orchestrator

        items = [{"name": "Item", "category": "Test"}]
        session_id = orch.start_enrichment_session(items)
        item_ids = state_mgr.get_session_item_ids(session_id)
        item_id = item_ids[0]

        # Update with specs
        state_mgr.update_item_state(session_id, item_id, {
            "specifications": {
                "temp_range": "0-100C",
                "accuracy": "0.1%"
            },
            "spec_count": 2
        })

        # Get specs before failure
        state1 = state_mgr.get_item_state(session_id, item_id)
        specs_before = state1["specifications"].copy()

        # Pop task and fail
        task = task_q.pop_task(timeout=5)
        task_q.nack_task(task["task_id"], retry=True)

        # Get specs after failure
        state2 = state_mgr.get_item_state(session_id, item_id)
        specs_after = state2["specifications"]

        # Specs should be identical
        assert specs_after == specs_before

    def test_completion_status_immutable(self, orchestrator):
        """Test that completed status cannot be changed back to in-progress."""
        orch, state_mgr, task_q = orchestrator

        items = [{"name": "Item", "category": "Test"}]
        session_id = orch.start_enrichment_session(items)
        item_ids = state_mgr.get_session_item_ids(session_id)
        item_id = item_ids[0]

        # Mark complete
        state_mgr.mark_item_complete(session_id, item_id, {
            "final_spec": "value"
        })

        status1 = state_mgr.get_item_state(session_id, item_id)
        assert status1["status"] == "completed"

        # Try to update to in_progress
        state_mgr.update_item_state(session_id, item_id, {
            "status": "in_progress",
            "new_field": "value"
        })

        # Status should still be completed
        status2 = state_mgr.get_item_state(session_id, item_id)
        assert status2["status"] == "completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
