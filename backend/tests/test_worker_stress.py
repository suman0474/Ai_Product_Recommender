"""
Worker Stress and Scaling Tests

Tests system behavior under high load and with multiple concurrent workers.
Uses in-memory backends for fast testing.

Run with: pytest tests/test_worker_stress.py -v
"""

import pytest
import threading
import time
from datetime import datetime
from typing import List

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
# HIGH VOLUME ITEM PROCESSING
# =============================================================================

class TestHighVolumeProcessing:
    """Test system with high volume of items."""

    def test_100_items_session(self, orchestrator):
        """Test session with 100 items."""
        orch, state_mgr, task_q = orchestrator

        items = [
            {"name": f"Item {i}", "category": "Test"}
            for i in range(100)
        ]

        session_id = orch.start_enrichment_session(items)

        # Verify all items created
        item_ids = state_mgr.get_session_item_ids(session_id)
        assert len(item_ids) == 100

        # Verify all tasks queued
        assert task_q.get_queue_depth() == 100

        # Complete all items
        for item_id in item_ids:
            state_mgr.mark_item_complete(session_id, item_id, {})

        # Verify completion
        status = orch.get_session_status(session_id)
        assert status["completed_items"] == 100
        assert status["is_complete"]

    def test_500_items_session(self, orchestrator):
        """Test session with 500 items."""
        orch, state_mgr, task_q = orchestrator

        items = [
            {"name": f"Item {i}", "category": "Test"}
            for i in range(500)
        ]

        session_id = orch.start_enrichment_session(items)
        item_ids = state_mgr.get_session_item_ids(session_id)

        assert len(item_ids) == 500

        # Complete all
        for i, item_id in enumerate(item_ids):
            state_mgr.mark_item_complete(session_id, item_id, {})

            # Check progress periodically
            if (i + 1) % 100 == 0:
                status = orch.get_session_status(session_id)
                assert status["completed_items"] == i + 1

        status = orch.get_session_status(session_id)
        assert status["is_complete"]

    def test_1000_items_session(self, orchestrator):
        """Test session with 1000 items."""
        orch, state_mgr, task_q = orchestrator

        items = [
            {"name": f"Item {i}", "category": "Test"}
            for i in range(1000)
        ]

        session_id = orch.start_enrichment_session(items)
        item_ids = state_mgr.get_session_item_ids(session_id)

        assert len(item_ids) == 1000

        # Complete in batches
        batch_size = 100
        for i in range(0, len(item_ids), batch_size):
            batch = item_ids[i:i+batch_size]
            for item_id in batch:
                state_mgr.mark_item_complete(session_id, item_id, {})

            # Verify batch completion
            status = orch.get_session_status(session_id)
            assert status["completed_items"] == min(i + batch_size, 1000)

        # Final verification
        status = orch.get_session_status(session_id)
        assert status["completed_items"] == 1000
        assert status["is_complete"]

    def test_5000_items_session(self, orchestrator):
        """Test session with 5000 items (extreme load)."""
        orch, state_mgr, task_q = orchestrator

        items = [
            {"name": f"Item {i}", "category": "Test"}
            for i in range(5000)
        ]

        session_id = orch.start_enrichment_session(items)
        item_ids = state_mgr.get_session_item_ids(session_id)

        assert len(item_ids) == 5000

        # Complete all quickly
        for item_id in item_ids:
            state_mgr.mark_item_complete(session_id, item_id, {})

        status = orch.get_session_status(session_id)
        assert status["is_complete"]


# =============================================================================
# CONCURRENT SESSION HANDLING
# =============================================================================

class TestConcurrentSessions:
    """Test multiple concurrent sessions."""

    def test_10_concurrent_sessions(self, orchestrator):
        """Test 10 concurrent sessions."""
        orch, state_mgr, task_q = orchestrator

        session_ids = []
        for i in range(10):
            items = [
                {"name": f"Session {i} Item {j}", "category": "Test"}
                for j in range(10)
            ]
            session_id = orch.start_enrichment_session(items)
            session_ids.append(session_id)

        # Verify all sessions exist
        assert len(session_ids) == 10

        # Complete all items in all sessions
        for session_id in session_ids:
            item_ids = state_mgr.get_session_item_ids(session_id)
            for item_id in item_ids:
                state_mgr.mark_item_complete(session_id, item_id, {})

        # Verify all sessions complete
        for session_id in session_ids:
            status = orch.get_session_status(session_id)
            assert status["is_complete"]

    def test_50_concurrent_sessions(self, orchestrator):
        """Test 50 concurrent sessions."""
        orch, state_mgr, task_q = orchestrator

        session_ids = []
        for i in range(50):
            items = [
                {"name": f"Item {j}", "category": "Test"}
                for j in range(5)
            ]
            session_id = orch.start_enrichment_session(items)
            session_ids.append(session_id)

        assert len(session_ids) == 50

        # Complete all
        for session_id in session_ids:
            item_ids = state_mgr.get_session_item_ids(session_id)
            for item_id in item_ids:
                state_mgr.mark_item_complete(session_id, item_id, {})

        # All should be complete
        completed = 0
        for session_id in session_ids:
            status = orch.get_session_status(session_id)
            if status["is_complete"]:
                completed += 1

        assert completed == 50

    def test_100_concurrent_sessions_with_interleaving(self, orchestrator):
        """Test 100 sessions with interleaved completion."""
        orch, state_mgr, task_q = orchestrator

        session_ids = []
        for i in range(100):
            items = [{"name": f"Item", "category": "Test"}]
            session_id = orch.start_enrichment_session(items)
            session_ids.append(session_id)

        # Interleave completion - complete every 5th session
        all_completed = set()
        for i, session_id in enumerate(session_ids):
            if i % 5 == 0:
                item_ids = state_mgr.get_session_item_ids(session_id)
                for item_id in item_ids:
                    state_mgr.mark_item_complete(session_id, item_id, {})
                all_completed.add(session_id)

        # Complete remaining
        for session_id in session_ids:
            if session_id not in all_completed:
                item_ids = state_mgr.get_session_item_ids(session_id)
                for item_id in item_ids:
                    state_mgr.mark_item_complete(session_id, item_id, {})

        # All should be complete
        for session_id in session_ids:
            status = orch.get_session_status(session_id)
            assert status["is_complete"]


# =============================================================================
# QUEUE STRESS UNDER LOAD
# =============================================================================

class TestQueueStress:
    """Test queue performance under stress."""

    def test_10000_task_queue(self, orchestrator):
        """Test queue with 10,000 tasks."""
        orch, state_mgr, task_q = orchestrator

        # Push 10,000 tasks
        for i in range(10000):
            task = {
                "task_id": f"task-{i}",
                "task_type": "initial_enrichment",
                "session_id": "session-1",
                "item_id": f"item-{i}",
                "payload": {},
                "priority": i % 10,
                "created_at": datetime.utcnow().isoformat(),
                "retry_count": 0
            }
            task_q.push_task(task)

        assert task_q.get_queue_depth() == 10000

        # Pop all tasks
        count = 0
        while True:
            task = task_q.pop_task(timeout=1)
            if not task:
                break
            count += 1
            task_q.ack_task(task["task_id"])

        assert count == 10000

    def test_queue_with_mixed_priorities(self, orchestrator):
        """Test queue ordering with mixed priorities."""
        orch, state_mgr, task_q = orchestrator

        # Push 1000 tasks with random priorities
        import random
        priorities = [random.randint(1, 20) for _ in range(1000)]

        for i, priority in enumerate(priorities):
            task = {
                "task_id": f"task-{i}",
                "task_type": "initial_enrichment",
                "session_id": "session-1",
                "item_id": f"item-{i}",
                "payload": {},
                "priority": priority,
                "created_at": datetime.utcnow().isoformat(),
                "retry_count": 0
            }
            task_q.push_task(task)

        # Pop and verify priority ordering
        prev_priority = 0
        correctly_ordered = 0
        count = 0

        while True:
            task = task_q.pop_task(timeout=1)
            if not task:
                break

            count += 1
            current_priority = task["priority"]

            if current_priority >= prev_priority:
                correctly_ordered += 1

            prev_priority = current_priority
            task_q.ack_task(task["task_id"])

        # Most should be in correct order (allowing some deviation)
        assert correctly_ordered >= count * 0.95  # 95%+ correct

    def test_queue_retry_surge(self, orchestrator):
        """Test queue behavior during retry surge."""
        orch, state_mgr, task_q = orchestrator

        # Push 100 tasks
        for i in range(100):
            task = {
                "task_id": f"task-{i}",
                "task_type": "initial_enrichment",
                "session_id": "session-1",
                "item_id": f"item-{i}",
                "payload": {},
                "priority": 1,
                "created_at": datetime.utcnow().isoformat(),
                "retry_count": 0
            }
            task_q.push_task(task)

        # Pop all and NACK with retry
        for _ in range(100):
            task = task_q.pop_task(timeout=1)
            if task:
                task_q.nack_task(task["task_id"], retry=True)

        # All should be requeued
        assert task_q.get_queue_depth() == 100


# =============================================================================
# STATE MANAGER STRESS
# =============================================================================

class TestStateManagerStress:
    """Test state manager under heavy load."""

    def test_large_specifications_update(self, orchestrator):
        """Test updating items with large specification sets."""
        orch, state_mgr, task_q = orchestrator

        items = [{"name": "Item", "category": "Test"}]
        session_id = orch.start_enrichment_session(items)
        item_ids = state_mgr.get_session_item_ids(session_id)
        item_id = item_ids[0]

        # Create large specification set (1000 specs)
        large_specs = {f"spec_{i}": f"value_{i}" for i in range(1000)}

        state_mgr.update_item_state(session_id, item_id, {
            "specifications": large_specs,
            "spec_count": 1000
        })

        item_state = state_mgr.get_item_state(session_id, item_id)
        assert len(item_state["specifications"]) == 1000

    def test_many_items_updates(self, orchestrator):
        """Test state manager with many item updates."""
        orch, state_mgr, task_q = orchestrator

        items = [
            {"name": f"Item {i}", "category": "Test"}
            for i in range(100)
        ]
        session_id = orch.start_enrichment_session(items)
        item_ids = state_mgr.get_session_item_ids(session_id)

        # Update all items 10 times each
        for iteration in range(10):
            for item_id in item_ids:
                state_mgr.update_item_state(session_id, item_id, {
                    "specifications": {f"spec_{iteration}": f"value_{iteration}"},
                    "iteration": iteration
                })

        # All items should have latest iteration
        for item_id in item_ids:
            item_state = state_mgr.get_item_state(session_id, item_id)
            assert item_state["iteration"] == 9

    def test_deep_nested_specifications(self, orchestrator):
        """Test deeply nested specification structures."""
        orch, state_mgr, task_q = orchestrator

        items = [{"name": "Item", "category": "Test"}]
        session_id = orch.start_enrichment_session(items)
        item_ids = state_mgr.get_session_item_ids(session_id)
        item_id = item_ids[0]

        # Create nested specs
        nested_specs = {
            "temperature": {
                "range": {"min": 0, "max": 100},
                "unit": "C",
                "precision": 0.1
            },
            "pressure": {
                "range": {"min": 0, "max": 1000},
                "unit": "Pa",
                "precision": 1.0
            },
            "flow": {
                "range": {"min": 0, "max": 100},
                "unit": "L/min",
                "precision": 0.01
            }
        }

        state_mgr.update_item_state(session_id, item_id, {
            "specifications": nested_specs,
            "spec_count": 3
        })

        item_state = state_mgr.get_item_state(session_id, item_id)
        assert item_state["specifications"]["temperature"]["range"]["max"] == 100


# =============================================================================
# MULTI-THREADED SCENARIO (Simulated)
# =============================================================================

class TestMultiWorkerSimulation:
    """Test simulated multi-worker scenarios."""

    def test_two_workers_processing(self, orchestrator):
        """Simulate two workers processing tasks."""
        orch, state_mgr, task_q = orchestrator

        items = [
            {"name": f"Item {i}", "category": "Test"}
            for i in range(20)
        ]
        session_id = orch.start_enrichment_session(items)
        item_ids = state_mgr.get_session_item_ids(session_id)

        # Simulate worker 1: process items 0-9
        for item_id in item_ids[:10]:
            task = task_q.pop_task(timeout=1)
            if task:
                state_mgr.mark_item_complete(session_id, item_id, {})
                task_q.ack_task(task["task_id"])

        # Simulate worker 2: process items 10-19
        for item_id in item_ids[10:]:
            task = task_q.pop_task(timeout=1)
            if task:
                state_mgr.mark_item_complete(session_id, item_id, {})
                task_q.ack_task(task["task_id"])

        # All should be complete
        status = orch.get_session_status(session_id)
        assert status["is_complete"]

    def test_four_workers_interleaved(self, orchestrator):
        """Simulate four workers with interleaved processing."""
        orch, state_mgr, task_q = orchestrator

        items = [
            {"name": f"Item {i}", "category": "Test"}
            for i in range(40)
        ]
        session_id = orch.start_enrichment_session(items)
        item_ids = state_mgr.get_session_item_ids(session_id)

        # Interleave processing: round-robin 4 workers
        for i, item_id in enumerate(item_ids):
            worker_id = i % 4

            task = task_q.pop_task(timeout=1)
            if task:
                state_mgr.mark_item_complete(session_id, item_id, {})
                task_q.ack_task(task["task_id"])

        status = orch.get_session_status(session_id)
        assert status["is_complete"]
        assert status["completed_items"] == 40

    def test_worker_uneven_load_distribution(self, orchestrator):
        """Test with uneven work distribution among workers."""
        orch, state_mgr, task_q = orchestrator

        items = [
            {"name": f"Item {i}", "category": "Test"}
            for i in range(100)
        ]
        session_id = orch.start_enrichment_session(items)
        item_ids = state_mgr.get_session_item_ids(session_id)

        # Worker 1: 50% of load
        for item_id in item_ids[:50]:
            task = task_q.pop_task(timeout=1)
            if task:
                state_mgr.mark_item_complete(session_id, item_id, {})
                task_q.ack_task(task["task_id"])

        # Worker 2: 30% of load
        for item_id in item_ids[50:80]:
            task = task_q.pop_task(timeout=1)
            if task:
                state_mgr.mark_item_complete(session_id, item_id, {})
                task_q.ack_task(task["task_id"])

        # Worker 3: 20% of load
        for item_id in item_ids[80:]:
            task = task_q.pop_task(timeout=1)
            if task:
                state_mgr.mark_item_complete(session_id, item_id, {})
                task_q.ack_task(task["task_id"])

        status = orch.get_session_status(session_id)
        assert status["is_complete"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
