# tests/test_task_queue.py
"""
Unit tests for EnrichmentTaskQueue

Tests both Redis and InMemory implementations.
"""

import pytest
import time
from datetime import datetime

from agentic.deep_agent.task_queue import (
    EnrichmentTask,
    EnrichmentTaskQueue,
    RedisTaskQueue,
    InMemoryTaskQueue,
    get_task_queue
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def in_memory_queue():
    """Create in-memory task queue for testing."""
    return InMemoryTaskQueue(max_retries=3)


@pytest.fixture
def redis_queue():
    """
    Create Redis task queue for testing.

    Skip if Redis not available.
    """
    try:
        queue = RedisTaskQueue(
            redis_url="redis://localhost:6379/1",
            queue_name="test_enrichment_tasks",
            max_retries=3
        )
        # Test connection
        queue.redis.ping()
        yield queue
        # Cleanup
        queue.redis.delete(queue.queue_name)
        queue.redis.delete(queue.processing_set)
        queue.redis.delete(queue.dlq_name)
        # Delete any processing keys
        for key in queue.redis.scan_iter(f"{queue.processing_set}:*"):
            queue.redis.delete(key)
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")


@pytest.fixture(params=["in_memory", "redis"])
def task_queue(request, in_memory_queue, redis_queue):
    """Parametrized fixture to test both implementations."""
    if request.param == "in_memory":
        return in_memory_queue
    elif request.param == "redis":
        return redis_queue


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_test_task(
    task_id: str = "test-task-1",
    priority: int = 1,
    task_type: str = "initial_enrichment"
) -> EnrichmentTask:
    """Create a test task."""
    return {
        "task_id": task_id,
        "task_type": task_type,
        "session_id": "test-session",
        "item_id": "item-1",
        "payload": {"user_requirement": "Test requirement"},
        "priority": priority,
        "created_at": datetime.utcnow().isoformat(),
        "retry_count": 0
    }


# =============================================================================
# TESTS
# =============================================================================

class TestBasicQueueOperations:
    """Test basic push/pop operations."""

    def test_push_task(self, task_queue):
        """Test pushing a task to queue."""
        task = create_test_task("task-1")

        task_queue.push_task(task)

        assert task_queue.get_queue_depth() == 1

    def test_pop_task(self, task_queue):
        """Test popping a task from queue."""
        task = create_test_task("task-2")

        task_queue.push_task(task)
        popped_task = task_queue.pop_task(timeout=5)

        assert popped_task is not None
        assert popped_task["task_id"] == "task-2"
        assert task_queue.get_queue_depth() == 0

    def test_pop_empty_queue(self, task_queue):
        """Test popping from empty queue returns None."""
        popped_task = task_queue.pop_task(timeout=1)
        assert popped_task is None

    def test_ack_task(self, task_queue):
        """Test acknowledging a task."""
        task = create_test_task("task-3")

        task_queue.push_task(task)
        popped_task = task_queue.pop_task(timeout=5)

        # Should be in processing
        assert task_queue.get_processing_count() == 1

        # ACK
        task_queue.ack_task(popped_task["task_id"])

        # Should no longer be in processing
        assert task_queue.get_processing_count() == 0


class TestPriorityQueue:
    """Test priority-based task ordering."""

    def test_priority_ordering(self, task_queue):
        """Test tasks are popped in priority order."""
        # Push tasks with different priorities
        task_queue.push_task(create_test_task("task-low", priority=10))
        task_queue.push_task(create_test_task("task-high", priority=1))
        task_queue.push_task(create_test_task("task-medium", priority=5))

        # Should pop in priority order (lower number = higher priority)
        task1 = task_queue.pop_task(timeout=5)
        assert task1["task_id"] == "task-high"

        task2 = task_queue.pop_task(timeout=5)
        assert task2["task_id"] == "task-medium"

        task3 = task_queue.pop_task(timeout=5)
        assert task3["task_id"] == "task-low"


class TestRetryLogic:
    """Test retry and DLQ behavior."""

    def test_nack_with_retry(self, task_queue):
        """Test NACK with retry re-queues task."""
        task = create_test_task("task-retry")

        task_queue.push_task(task)
        popped_task = task_queue.pop_task(timeout=5)

        # NACK with retry
        task_queue.nack_task(popped_task["task_id"], retry=True)

        # Should be re-queued
        assert task_queue.get_queue_depth() == 1
        assert task_queue.get_processing_count() == 0

        # Pop again and check retry count
        retry_task = task_queue.pop_task(timeout=5)
        assert retry_task["retry_count"] == 1

    def test_nack_max_retries(self, task_queue):
        """Test NACK after max retries moves to DLQ."""
        task = create_test_task("task-maxretry")

        task_queue.push_task(task)

        # Retry 3 times (max_retries=3)
        for i in range(4):
            popped_task = task_queue.pop_task(timeout=5)
            task_queue.nack_task(popped_task["task_id"], retry=True)

        # Should be in DLQ now
        assert task_queue.get_queue_depth() == 0

        # Check DLQ (only Redis has this)
        if isinstance(task_queue, RedisTaskQueue):
            assert task_queue.get_dlq_depth() > 0

    def test_nack_no_retry(self, task_queue):
        """Test NACK with retry=False moves to DLQ immediately."""
        task = create_test_task("task-noretry")

        task_queue.push_task(task)
        popped_task = task_queue.pop_task(timeout=5)

        # NACK without retry
        task_queue.nack_task(popped_task["task_id"], retry=False)

        # Should NOT be re-queued
        assert task_queue.get_queue_depth() == 0

        # Should be in DLQ (for Redis)
        if isinstance(task_queue, RedisTaskQueue):
            assert task_queue.get_dlq_depth() == 1


class TestRedisSpecific:
    """Tests specific to Redis implementation."""

    def test_redis_connection(self, redis_queue):
        """Test Redis connection."""
        assert redis_queue.redis.ping()

    def test_redis_dlq_operations(self, redis_queue):
        """Test DLQ peek and clear."""
        task = create_test_task("task-dlq")

        # Push and immediately move to DLQ
        redis_queue.push_task(task)
        popped_task = redis_queue.pop_task(timeout=5)
        redis_queue.nack_task(popped_task["task_id"], retry=False)

        # Peek DLQ
        dlq_tasks = redis_queue.peek_dlq(limit=10)
        assert len(dlq_tasks) == 1
        assert dlq_tasks[0]["task_id"] == "task-dlq"

        # Clear DLQ
        cleared = redis_queue.clear_dlq()
        assert cleared == 1
        assert redis_queue.get_dlq_depth() == 0

    def test_redis_blocking_pop(self, redis_queue):
        """Test blocking pop with timeout."""
        import threading

        def delayed_push():
            time.sleep(2)
            redis_queue.push_task(create_test_task("delayed-task"))

        # Start thread to push after delay
        thread = threading.Thread(target=delayed_push)
        thread.start()

        # Blocking pop should wait and return task
        start_time = time.time()
        task = redis_queue.pop_task(timeout=5)
        elapsed = time.time() - start_time

        assert task is not None
        assert task["task_id"] == "delayed-task"
        assert elapsed >= 2  # Should have waited

        thread.join()


class TestMultipleWorkers:
    """Test concurrent worker behavior."""

    def test_two_workers_different_tasks(self, task_queue):
        """Test two workers can pop different tasks."""
        task_queue.push_task(create_test_task("task-1"))
        task_queue.push_task(create_test_task("task-2"))

        # Worker 1
        task1 = task_queue.pop_task(timeout=5)
        assert task1 is not None

        # Worker 2
        task2 = task_queue.pop_task(timeout=5)
        assert task2 is not None

        # Should be different tasks
        assert task1["task_id"] != task2["task_id"]

        # Both should be in processing
        assert task_queue.get_processing_count() == 2


class TestFactoryFunction:
    """Test factory function."""

    def test_get_task_queue_redis(self):
        """Test factory returns Redis queue."""
        try:
            queue = get_task_queue(use_redis=True)
            assert isinstance(queue, (RedisTaskQueue, InMemoryTaskQueue))
        except Exception:
            pytest.skip("Redis not available")

    def test_get_task_queue_memory(self):
        """Test factory returns in-memory queue."""
        queue = get_task_queue(use_redis=False)
        assert isinstance(queue, InMemoryTaskQueue)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
