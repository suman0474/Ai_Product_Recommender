# tests/test_end_to_end.py
"""
End-to-End Workflow Tests

Tests complete enrichment workflows from start to finish.
Uses in-memory backends for fast testing.

Run with: pytest tests/test_end_to_end.py -v
"""

import pytest
import time
from datetime import datetime

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
# SINGLE ITEM WORKFLOWS
# =============================================================================

class TestSingleItemWorkflow:
    """Test single item enrichment workflows."""

    def test_single_item_start_to_status(self, orchestrator):
        """Test starting enrichment and checking status for single item."""
        orch, state_mgr, task_q = orchestrator

        items = [{"name": "Test Item", "category": "Test"}]

        # Start session
        session_id = orch.start_enrichment_session(items)
        assert session_id is not None

        # Check initial status
        status = orch.get_session_status(session_id)
        assert status["total_items"] == 1
        assert status["completed_items"] == 0
        assert status["progress_percentage"] == 0.0
        assert not status["is_complete"]

        # Verify task in queue
        assert task_q.get_queue_depth() == 1

    def test_single_item_completion(self, orchestrator):
        """Test completing a single item enrichment."""
        orch, state_mgr, task_q = orchestrator

        items = [{"name": "Test Item", "category": "Test"}]

        # Start
        session_id = orch.start_enrichment_session(items)

        # Get item ID
        item_ids = state_mgr.get_session_item_ids(session_id)
        item_id = item_ids[0]

        # Simulate enrichment completion
        state_mgr.mark_item_complete(session_id, item_id, {
            "temp_range": "0-100C",
            "accuracy": "0.5%"
        })

        # Check final status
        status = orch.get_session_status(session_id)
        assert status["completed_items"] == 1
        assert status["progress_percentage"] == 100.0
        assert status["is_complete"]

    def test_single_item_progress_tracking(self, orchestrator):
        """Test progress tracking throughout enrichment."""
        orch, state_mgr, task_q = orchestrator

        items = [{"name": "Test Item", "category": "Test"}]

        session_id = orch.start_enrichment_session(items)
        item_ids = state_mgr.get_session_item_ids(session_id)
        item_id = item_ids[0]

        # Initial status
        status1 = orch.get_session_status(session_id)
        assert status1["progress_percentage"] == 0.0

        # Update to in-progress
        state_mgr.update_item_state(session_id, item_id, {
            "specifications": {"temp": "100C"},
            "spec_count": 1,
            "status": "in_progress"
        })

        # Still not complete
        status2 = orch.get_session_status(session_id)
        assert status2["progress_percentage"] == 0.0

        # Complete
        state_mgr.mark_item_complete(session_id, item_id, {"temp": "100C"})

        # Now complete
        status3 = orch.get_session_status(session_id)
        assert status3["progress_percentage"] == 100.0


# =============================================================================
# BATCH WORKFLOWS
# =============================================================================

class TestBatchWorkflow:
    """Test batch processing workflows."""

    def test_batch_initialization(self, orchestrator):
        """Test initializing a batch of items."""
        orch, state_mgr, task_q = orchestrator

        items = [
            {"name": f"Item {i}", "category": "Test"}
            for i in range(5)
        ]

        session_id = orch.start_enrichment_session(items)

        # Verify all items created
        item_ids = state_mgr.get_session_item_ids(session_id)
        assert len(item_ids) == 5

        # Verify all tasks queued
        assert task_q.get_queue_depth() == 5

    def test_batch_parallel_completion(self, orchestrator):
        """Test completing multiple items in parallel."""
        orch, state_mgr, task_q = orchestrator

        items = [
            {"name": f"Item {i}", "category": "Test"}
            for i in range(3)
        ]

        session_id = orch.start_enrichment_session(items)
        item_ids = state_mgr.get_session_item_ids(session_id)

        # Complete all items
        for i, item_id in enumerate(item_ids):
            state_mgr.mark_item_complete(session_id, item_id, {
                "param": f"value-{i}"
            })

        # Verify completion
        status = orch.get_session_status(session_id)
        assert status["completed_items"] == 3
        assert status["is_complete"]

    def test_batch_partial_completion(self, orchestrator):
        """Test progress with partial completion."""
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

        # Check progress
        status = orch.get_session_status(session_id)
        assert status["completed_items"] == 2
        assert status["progress_percentage"] == 50.0
        assert not status["is_complete"]

        # Complete remaining
        state_mgr.mark_item_complete(session_id, item_ids[2], {})
        state_mgr.mark_item_complete(session_id, item_ids[3], {})

        # Now complete
        status = orch.get_session_status(session_id)
        assert status["completed_items"] == 4
        assert status["progress_percentage"] == 100.0
        assert status["is_complete"]

    def test_batch_category_domain_planning(self, orchestrator):
        """Test domain planning for different categories."""
        orch, state_mgr, task_q = orchestrator

        test_cases = [
            ({"name": "Pressure Transmitter", "category": "Pressure"}, ["safety", "pressure"]),
            ({"name": "Temperature Sensor", "category": "Temperature"}, ["safety", "temperature"]),
            ({"name": "Flow Meter", "category": "Flow"}, ["safety", "flow"]),
            ({"name": "Unknown Device", "category": "Unknown"}, ["safety", "accessories"]),
        ]

        for item, expected_domains in test_cases:
            domains = orch._plan_initial_domains(item)
            assert any(d in domains for d in expected_domains), \
                f"Expected domains {expected_domains} not found in {domains}"


# =============================================================================
# TASK QUEUE INTEGRATION
# =============================================================================

class TestTaskQueueIntegration:
    """Test task queue integration with orchestrator."""

    def test_task_priority_distribution(self, orchestrator):
        """Test that initial tasks have correct priority."""
        orch, state_mgr, task_q = orchestrator

        items = [{"name": "Item", "category": "Test"}]
        session_id = orch.start_enrichment_session(items)

        # Pop task and check priority
        task = task_q.pop_task(timeout=5)
        assert task is not None
        assert task["priority"] == 1  # Initial tasks have high priority

    def test_task_payload_structure(self, orchestrator):
        """Test that task payloads are structured correctly."""
        orch, state_mgr, task_q = orchestrator

        items = [{"name": "Test Item", "category": "Test"}]
        session_id = orch.start_enrichment_session(items)

        task = task_q.pop_task(timeout=5)

        # Verify payload structure
        assert "user_requirement" in task["payload"]
        assert "initial_domains" in task["payload"]
        assert task["task_type"] == "initial_enrichment"
        assert task["session_id"] == session_id


# =============================================================================
# SESSION LIFECYCLE
# =============================================================================

class TestSessionLifecycle:
    """Test complete session lifecycle."""

    def test_session_creation_to_completion(self, orchestrator):
        """Test complete lifecycle from creation to completion."""
        orch, state_mgr, task_q = orchestrator

        items = [{"name": "Item", "category": "Test"}]

        # Create
        session_id = orch.start_enrichment_session(items)
        assert session_id is not None

        # Get initial status
        status = orch.get_session_status(session_id)
        assert status["status"] == "in_progress"

        # Complete
        item_ids = state_mgr.get_session_item_ids(session_id)
        for item_id in item_ids:
            state_mgr.mark_item_complete(session_id, item_id, {})

        # Get final status
        status = orch.get_session_status(session_id)
        assert status["is_complete"]

    def test_session_cancellation(self, orchestrator):
        """Test session cancellation."""
        orch, state_mgr, task_q = orchestrator

        items = [{"name": "Item", "category": "Test"}]

        session_id = orch.start_enrichment_session(items)

        # Cancel
        result = orch.cancel_session(session_id)
        assert result["status"] == "cancelled"

        # Verify still in queue (cancellation doesn't remove in-progress tasks)
        assert task_q.get_queue_depth() == 1

    def test_multiple_concurrent_sessions(self, orchestrator):
        """Test handling multiple concurrent sessions."""
        orch, state_mgr, task_q = orchestrator

        # Create multiple sessions
        session_ids = []
        for i in range(3):
            items = [{"name": f"Session {i} Item", "category": "Test"}]
            session_id = orch.start_enrichment_session(items)
            session_ids.append(session_id)

        # Verify all independent
        for session_id in session_ids:
            status = orch.get_session_status(session_id)
            assert status["total_items"] == 1
            assert status["completed_items"] == 0

        # Verify queue has all tasks
        assert task_q.get_queue_depth() == 3

        # Complete in different order
        state_mgr.mark_item_complete(session_ids[2], "item-0", {})
        state_mgr.mark_item_complete(session_ids[0], "item-0", {})
        state_mgr.mark_item_complete(session_ids[1], "item-0", {})

        # All should be complete
        for session_id in session_ids:
            status = orch.get_session_status(session_id)
            assert status["is_complete"]


# =============================================================================
# ERROR CONDITIONS
# =============================================================================

class TestErrorConditions:
    """Test error handling and edge cases."""

    def test_nonexistent_session(self, orchestrator):
        """Test accessing non-existent session."""
        orch, state_mgr, task_q = orchestrator

        status = orch.get_session_status("nonexistent-session")
        assert "error" in status

    def test_empty_items_list(self, orchestrator):
        """Test starting enrichment with empty items list."""
        orch, state_mgr, task_q = orchestrator

        session_id = orch.start_enrichment_session([])

        status = orch.get_session_status(session_id)
        assert status["total_items"] == 0

    def test_items_with_missing_fields(self, orchestrator):
        """Test items with missing optional fields."""
        orch, state_mgr, task_q = orchestrator

        items = [
            {"name": "Item 1"},  # No category
            {"category": "Test"},  # No name
            {},  # Empty
        ]

        session_id = orch.start_enrichment_session(items)
        status = orch.get_session_status(session_id)

        # Should still create session
        assert status["total_items"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
