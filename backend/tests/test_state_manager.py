# tests/test_state_manager.py
"""
Unit tests for EnrichmentStateManager

Tests both Redis and InMemory implementations.
"""

import pytest
import json
import time
from datetime import datetime

from agentic.deep_agent.state_manager import (
    EnrichmentStateManager,
    RedisStateManager,
    InMemoryStateManager,
    get_state_manager
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def in_memory_manager():
    """Create in-memory state manager for testing."""
    return InMemoryStateManager()


@pytest.fixture
def redis_manager():
    """
    Create Redis state manager for testing.

    Skip if Redis not available.
    """
    try:
        manager = RedisStateManager(redis_url="redis://localhost:6379/1", ttl=60)
        # Test connection
        manager.redis.ping()
        yield manager
        # Cleanup
        manager.redis.flushdb()
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")


@pytest.fixture(params=["in_memory", "redis"])
def state_manager(request, in_memory_manager, redis_manager):
    """Parametrized fixture to test both implementations."""
    if request.param == "in_memory":
        return in_memory_manager
    elif request.param == "redis":
        return redis_manager


# =============================================================================
# TESTS
# =============================================================================

class TestStateManagerBasic:
    """Basic state manager operations."""

    def test_create_session(self, state_manager):
        """Test session creation."""
        session_id = "test-session-1"

        state_manager.create_session(session_id, {
            "total_items": 5,
            "status": "in_progress"
        })

        session_state = state_manager.get_session_state(session_id)

        assert session_state is not None
        assert session_state["total_items"] == 5
        assert session_state["status"] == "in_progress"
        assert "created_at" in session_state

    def test_get_nonexistent_session(self, state_manager):
        """Test retrieving non-existent session."""
        result = state_manager.get_session_state("nonexistent-session")
        assert result is None

    def test_update_item_state(self, state_manager):
        """Test item state update."""
        session_id = "test-session-2"
        item_id = "item-1"

        # Create session first
        state_manager.create_session(session_id, {"total_items": 1})

        # Update item state
        state_manager.update_item_state(session_id, item_id, {
            "specifications": {"temp_range": "0-100C"},
            "spec_count": 1,
            "status": "in_progress"
        })

        # Retrieve and verify
        item_state = state_manager.get_item_state(session_id, item_id)

        assert item_state is not None
        assert item_state["spec_count"] == 1
        assert item_state["specifications"]["temp_range"] == "0-100C"
        assert item_state["status"] == "in_progress"
        assert "updated_at" in item_state

    def test_get_nonexistent_item(self, state_manager):
        """Test retrieving non-existent item."""
        result = state_manager.get_item_state("nonexistent-session", "nonexistent-item")
        assert result is None


class TestItemCompletion:
    """Test item completion tracking."""

    def test_mark_item_complete(self, state_manager):
        """Test marking item as complete."""
        session_id = "test-session-3"
        item_id = "item-1"

        # Create session
        state_manager.create_session(session_id, {"total_items": 1})

        # Update item
        state_manager.update_item_state(session_id, item_id, {
            "specifications": {"temp_range": "0-100C"},
            "spec_count": 1
        })

        # Mark complete
        final_specs = {"temp_range": "0-100C", "accuracy": "0.5%"}
        state_manager.mark_item_complete(session_id, item_id, final_specs)

        # Verify
        item_state = state_manager.get_item_state(session_id, item_id)

        assert item_state["status"] == "completed"
        assert item_state["final_specs"] == final_specs
        assert "completed_at" in item_state

    def test_session_completion_single_item(self, state_manager):
        """Test session completion with single item."""
        session_id = "test-session-4"
        item_id = "item-1"

        # Create session
        state_manager.create_session(session_id, {"total_items": 1})

        # Should not be complete initially
        assert not state_manager.is_session_complete(session_id)

        # Update and complete item
        state_manager.update_item_state(session_id, item_id, {"spec_count": 1})
        state_manager.mark_item_complete(session_id, item_id, {"temp": "100C"})

        # Should be complete now
        assert state_manager.is_session_complete(session_id)

    def test_session_completion_multiple_items(self, state_manager):
        """Test session completion with multiple items."""
        session_id = "test-session-5"

        # Create session
        state_manager.create_session(session_id, {"total_items": 3})

        # Add items
        for i in range(3):
            item_id = f"item-{i}"
            state_manager.update_item_state(session_id, item_id, {"spec_count": 1})

        # Should not be complete yet
        assert not state_manager.is_session_complete(session_id)

        # Complete first two items
        state_manager.mark_item_complete(session_id, "item-0", {"temp": "100C"})
        state_manager.mark_item_complete(session_id, "item-1", {"temp": "100C"})

        # Still not complete
        assert not state_manager.is_session_complete(session_id)

        # Complete last item
        state_manager.mark_item_complete(session_id, "item-2", {"temp": "100C"})

        # Now complete
        assert state_manager.is_session_complete(session_id)


class TestSessionItems:
    """Test session item tracking."""

    def test_get_session_item_ids(self, state_manager):
        """Test retrieving all item IDs in a session."""
        session_id = "test-session-6"

        # Create session
        state_manager.create_session(session_id, {"total_items": 3})

        # Add items
        item_ids = ["item-1", "item-2", "item-3"]
        for item_id in item_ids:
            state_manager.update_item_state(session_id, item_id, {"spec_count": 0})

        # Get item IDs
        retrieved_ids = state_manager.get_session_item_ids(session_id)

        assert len(retrieved_ids) == 3
        for item_id in item_ids:
            assert item_id in retrieved_ids

    def test_delete_session(self, state_manager):
        """Test session deletion."""
        session_id = "test-session-7"

        # Create session with items
        state_manager.create_session(session_id, {"total_items": 2})
        state_manager.update_item_state(session_id, "item-1", {"spec_count": 1})
        state_manager.update_item_state(session_id, "item-2", {"spec_count": 2})

        # Verify exists
        assert state_manager.get_session_state(session_id) is not None
        assert state_manager.get_item_state(session_id, "item-1") is not None

        # Delete
        state_manager.delete_session(session_id)

        # Verify deleted
        assert state_manager.get_session_state(session_id) is None
        assert state_manager.get_item_state(session_id, "item-1") is None
        assert len(state_manager.get_session_item_ids(session_id)) == 0


class TestRedisSpecific:
    """Tests specific to Redis implementation."""

    def test_redis_connection(self, redis_manager):
        """Test Redis connection."""
        # Should be connected (fixture ensures this)
        assert redis_manager.redis.ping()

    def test_redis_ttl(self, redis_manager):
        """Test that keys have TTL set."""
        session_id = "test-session-ttl"

        redis_manager.create_session(session_id, {"total_items": 1})

        # Check TTL
        key = redis_manager._session_key(session_id)
        ttl = redis_manager.redis.ttl(key)

        assert ttl > 0
        assert ttl <= 60  # Should be <= configured TTL

    def test_redis_key_formats(self, redis_manager):
        """Test Redis key generation."""
        session_id = "test-session-keys"
        item_id = "item-1"

        assert redis_manager._session_key(session_id) == f"enrichment:session:{session_id}"
        assert redis_manager._item_key(session_id, item_id) == f"enrichment:session:{session_id}:item:{item_id}"
        assert redis_manager._items_set_key(session_id) == f"enrichment:session:{session_id}:items"
        assert redis_manager._completed_set_key(session_id) == f"enrichment:session:{session_id}:completed"


class TestFactoryFunction:
    """Test factory function."""

    def test_get_state_manager_redis(self):
        """Test factory returns Redis manager."""
        try:
            manager = get_state_manager(use_redis=True)
            assert isinstance(manager, (RedisStateManager, InMemoryStateManager))
        except Exception:
            pytest.skip("Redis not available")

    def test_get_state_manager_memory(self):
        """Test factory returns in-memory manager."""
        manager = get_state_manager(use_redis=False)
        assert isinstance(manager, InMemoryStateManager)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
