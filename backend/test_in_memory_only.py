#!/usr/bin/env python
"""
In-Memory Only Tests (No Redis Required)
"""

import sys
import pytest

# Mark all Redis tests to be skipped
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

def test_state_manager_create_session():
    """Test session creation."""
    from agentic.deep_agent.state_manager import InMemoryStateManager

    manager = InMemoryStateManager()
    manager.create_session("test-session-1", {"total_items": 5, "status": "in_progress"})

    session_state = manager.get_session_state("test-session-1")
    assert session_state is not None
    assert session_state["total_items"] == 5
    assert session_state["status"] == "in_progress"
    assert "created_at" in session_state


def test_state_manager_update_item():
    """Test item state update."""
    from agentic.deep_agent.state_manager import InMemoryStateManager

    manager = InMemoryStateManager()
    manager.create_session("test-session-2", {"total_items": 1})

    manager.update_item_state("test-session-2", "item-1", {
        "specifications": {"temp_range": "0-100C"},
        "spec_count": 1,
        "status": "in_progress"
    })

    item_state = manager.get_item_state("test-session-2", "item-1")
    assert item_state is not None
    assert item_state["spec_count"] == 1
    assert item_state["specifications"]["temp_range"] == "0-100C"


def test_state_manager_completion():
    """Test item completion tracking."""
    from agentic.deep_agent.state_manager import InMemoryStateManager

    manager = InMemoryStateManager()
    manager.create_session("test-session-3", {"total_items": 2})

    # Add items
    for i in range(2):
        manager.update_item_state("test-session-3", f"item-{i}", {"spec_count": 1})

    # Should not be complete
    assert not manager.is_session_complete("test-session-3")

    # Complete both items
    manager.mark_item_complete("test-session-3", "item-0", {"temp": "100C"})
    manager.mark_item_complete("test-session-3", "item-1", {"temp": "50C"})

    # Should be complete now
    assert manager.is_session_complete("test-session-3")


def test_task_queue_push_pop():
    """Test pushing and popping tasks."""
    from agentic.deep_agent.task_queue import InMemoryTaskQueue
    from datetime import datetime

    queue = InMemoryTaskQueue()

    task = {
        "task_id": "test-task-1",
        "task_type": "initial_enrichment",
        "session_id": "test-session",
        "item_id": "item-1",
        "payload": {"user_requirement": "Test"},
        "priority": 1,
        "created_at": datetime.utcnow().isoformat(),
        "retry_count": 0
    }

    queue.push_task(task)
    assert queue.get_queue_depth() == 1

    popped_task = queue.pop_task(timeout=1)
    assert popped_task is not None
    assert popped_task["task_id"] == "test-task-1"
    assert queue.get_processing_count() == 1

    queue.ack_task("test-task-1")
    assert queue.get_processing_count() == 0


def test_task_queue_priority():
    """Test priority queue ordering."""
    from agentic.deep_agent.task_queue import InMemoryTaskQueue
    from datetime import datetime

    queue = InMemoryTaskQueue()

    base_task = {
        "task_type": "initial_enrichment",
        "session_id": "test-session",
        "item_id": "item-1",
        "payload": {},
        "created_at": datetime.utcnow().isoformat(),
        "retry_count": 0
    }

    # Push with different priorities
    queue.push_task({**base_task, "task_id": "low", "priority": 10})
    queue.push_task({**base_task, "task_id": "high", "priority": 1})
    queue.push_task({**base_task, "task_id": "medium", "priority": 5})

    # Pop in priority order
    t1 = queue.pop_task(timeout=1)
    assert t1["task_id"] == "high"

    t2 = queue.pop_task(timeout=1)
    assert t2["task_id"] == "medium"

    t3 = queue.pop_task(timeout=1)
    assert t3["task_id"] == "low"


def test_task_queue_retry():
    """Test task retry logic."""
    from agentic.deep_agent.task_queue import InMemoryTaskQueue
    from datetime import datetime

    queue = InMemoryTaskQueue(max_retries=2)

    task = {
        "task_id": "retry-task",
        "task_type": "initial_enrichment",
        "session_id": "test-session",
        "item_id": "item-1",
        "payload": {},
        "priority": 1,
        "created_at": datetime.utcnow().isoformat(),
        "retry_count": 0
    }

    queue.push_task(task)
    popped = queue.pop_task(timeout=1)

    # NACK with retry
    queue.nack_task(popped["task_id"], retry=True)

    # Should be re-queued
    assert queue.get_queue_depth() == 1
    assert queue.get_processing_count() == 0

    # Check retry count increased
    retry_task = queue.pop_task(timeout=1)
    assert retry_task["retry_count"] == 1


def test_orchestrator_start_session():
    """Test orchestrator session start."""
    from agentic.deep_agent.stateless_orchestrator import StatelessEnrichmentOrchestrator
    from agentic.deep_agent.state_manager import InMemoryStateManager
    from agentic.deep_agent.task_queue import InMemoryTaskQueue

    state_mgr = InMemoryStateManager()
    task_q = InMemoryTaskQueue()

    orchestrator = StatelessEnrichmentOrchestrator(state_mgr, task_q)

    items = [
        {"name": "Item 1", "category": "Test"},
        {"name": "Item 2", "category": "Test"}
    ]

    session_id = orchestrator.start_enrichment_session(items)
    assert len(session_id) > 0

    # Check status
    status = orchestrator.get_session_status(session_id)
    assert status["total_items"] == 2
    assert status["completed_items"] == 0
    assert not status["is_complete"]

    # Check queue has tasks
    assert task_q.get_queue_depth() == 2


def test_orchestrator_session_completion():
    """Test orchestrator completion tracking."""
    from agentic.deep_agent.stateless_orchestrator import StatelessEnrichmentOrchestrator
    from agentic.deep_agent.state_manager import InMemoryStateManager
    from agentic.deep_agent.task_queue import InMemoryTaskQueue

    state_mgr = InMemoryStateManager()
    task_q = InMemoryTaskQueue()

    orchestrator = StatelessEnrichmentOrchestrator(state_mgr, task_q)

    items = [{"name": "Item 1", "category": "Test"}]
    session_id = orchestrator.start_enrichment_session(items)

    # Get initial status
    status = orchestrator.get_session_status(session_id)
    assert status["progress_percentage"] == 0.0

    # Mark item as complete
    item_ids = state_mgr.get_session_item_ids(session_id)
    for item_id in item_ids:
        state_mgr.mark_item_complete(session_id, item_id, {"temp": "100C"})

    # Check status again
    status = orchestrator.get_session_status(session_id)
    assert status["completed_items"] == 1
    assert status["is_complete"]
    assert status["progress_percentage"] == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
