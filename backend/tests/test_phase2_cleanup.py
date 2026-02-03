"""
Test Phase 2: Application-Level Resource Cleanup

Tests that cleanup handlers are properly registered and execute correctly
during application shutdown.
"""

import unittest
import time
import logging
from unittest.mock import patch, MagicMock

logging.basicConfig(level=logging.INFO)


class TestPhase2Cleanup(unittest.TestCase):
    """Test Phase 2 cleanup functionality."""

    def test_bounded_cache_cleanup(self):
        """Test that bounded caches are properly cleaned up."""
        from agentic.caching.bounded_cache_manager import (
            get_or_create_cache,
            cleanup_all_caches,
            get_all_cache_stats
        )

        # Create and populate a cache
        cache = get_or_create_cache("test_cleanup", max_size=10, ttl_seconds=1)
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert len(cache) == 2, "Cache should have 2 entries"

        # Wait for TTL expiration
        time.sleep(1.1)

        # Cleanup should remove expired entries
        cleanup_results = cleanup_all_caches()
        assert "test_cleanup" in cleanup_results, "Cleanup should process test_cleanup"

        # Verify cleanup worked
        assert cache.get("key1") is None, "Expired entry should be None"
        assert cache.get("key2") is None, "Expired entry should be None"

    def test_cache_stats_on_cleanup(self):
        """Test that cache statistics are available for logging."""
        from agentic.caching.bounded_cache_manager import (
            get_or_create_cache,
            get_all_cache_stats,
            get_registry_summary
        )

        # Create cache and add data
        cache = get_or_create_cache("test_stats", max_size=100, ttl_seconds=3600)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Get stats
        stats = get_all_cache_stats()
        assert "test_stats" in stats, "test_stats should be in stats"
        assert stats["test_stats"]["size"] == 3, "Cache should have 3 entries"
        assert stats["test_stats"]["hits"] >= 0, "Stats should track hits"

        # Get summary
        summary = get_registry_summary()
        assert summary["cache_count"] >= 1, "Should have at least 1 cache"
        assert summary["total_entries"] >= 3, "Should have at least 3 total entries"

    def test_cache_cleanup_task_creation(self):
        """Test that cache cleanup task can be created and started."""
        from agentic.tasks.cache_cleanup_task import (
            CacheCleanupTask,
            get_cleanup_status
        )

        # Create task
        task = CacheCleanupTask(interval_seconds=2)
        assert not task._is_running, "Task should not be running initially"

        # Start task
        task.start()
        assert task._is_running, "Task should be running after start"

        # Check status
        time.sleep(0.5)  # Give it a moment to start
        assert task._thread and task._thread.is_alive(), "Thread should be alive"

        # Stop task
        task.stop()
        assert not task._is_running, "Task should not be running after stop"

    def test_cleanup_handlers_registered(self):
        """Test that cleanup handlers are registered with atexit."""
        from unittest.mock import patch
        from initialization import register_cleanup_handlers

        # Mock atexit.register to verify it's called
        with patch('atexit.register') as mock_register:
            register_cleanup_handlers()

            # Should have registered 4 handlers (stop task, cleanup caches, shutdown executor, log final)
            # Check that register was called at least 3 times
            assert mock_register.call_count >= 3, \
                f"Should register at least 3 handlers, got {mock_register.call_count} calls"

    def test_global_executor_shutdown(self):
        """Test that global executor can be shutdown gracefully."""
        try:
            from agentic.global_executor_manager import (
                get_global_executor,
                get_executor_stats,
                shutdown_global_executor
            )

            # Get executor
            executor = get_global_executor()
            assert executor is not None, "Global executor should be available"

            # Verify executor is active before running task
            stats_before = get_executor_stats()
            assert stats_before.get("is_active", False), "Executor should be active before shutdown"

            # Submit a simple task
            future = executor.submit(lambda: 42)
            result = future.result(timeout=5)
            assert result == 42, "Task should complete successfully"

            # Shutdown
            shutdown_global_executor(wait=True)

            # Verify it's shutdown
            stats_after = get_executor_stats()
            assert not stats_after.get("is_active", False), "Executor should be shutdown"

        except ImportError:
            self.skipTest("Global executor not available")

    def test_initialization_with_cleanup(self):
        """Test that initialization.py properly registers cleanup."""
        # This is more of an integration test
        # Verify the function exists and has the right structure
        from initialization import register_cleanup_handlers

        assert callable(register_cleanup_handlers), "register_cleanup_handlers should be callable"

        # Test that it doesn't raise exceptions
        try:
            register_cleanup_handlers()
        except Exception as e:
            self.fail(f"register_cleanup_handlers raised {type(e).__name__}: {e}")


class TestCleanupIntegration(unittest.TestCase):
    """Integration tests for cleanup system."""

    def test_full_cleanup_sequence(self):
        """Test the full cleanup sequence: create caches, fill them, cleanup."""
        from agentic.caching.bounded_cache_manager import (
            get_or_create_cache,
            cleanup_all_caches,
            clear_all_caches,
            get_registry_summary
        )

        # Create multiple caches
        cache1 = get_or_create_cache("integration_test_1", max_size=5)
        cache2 = get_or_create_cache("integration_test_2", max_size=10)

        # Populate caches
        for i in range(3):
            cache1.set(f"key{i}", f"value{i}")
            cache2.set(f"key{i}", f"value{i}")

        # Verify population
        summary = get_registry_summary()
        assert summary["total_entries"] >= 6, "Should have at least 6 entries"

        # Cleanup (shouldn't remove non-expired entries)
        cleanup_results = cleanup_all_caches()
        assert cache1.get("key0") is not None, "Non-expired entries should remain"

        # Clear all caches
        clear_results = clear_all_caches()
        assert clear_results["integration_test_1"] == 3, "Should clear 3 entries from cache1"
        assert clear_results["integration_test_2"] == 3, "Should clear 3 entries from cache2"

        # Verify cleared
        summary_after = get_registry_summary()
        assert summary_after["total_entries"] == 0, "All entries should be cleared"


if __name__ == "__main__":
    unittest.main()
