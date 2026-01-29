"""
API Key Manager - Thread-Safe Singleton
========================================

Centralized management of API keys with thread-safe rotation.
Replaces scattered global state in llm_fallback.py, config.py, and main.py.

Features:
- Thread-safe singleton pattern
- Automatic loading of multiple Google API keys (GOOGLE_API_KEY, GOOGLE_API_KEY2, ...)
- Thread-safe rotation with locks
- Single source of truth for all API keys

Usage:
    from config.api_key_manager import api_key_manager

    # Get current key
    key = api_key_manager.get_current_google_key()

    # Rotate to next key
    if api_key_manager.rotate_google_key():
        print("Rotated to next key")
"""

import os
import threading
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class APIKeyManager:
    """
    Thread-safe singleton for managing API key rotation.

    This class ensures that:
    1. Only one instance exists across the entire application
    2. API key rotation is thread-safe with proper locking
    3. All modules use the same rotation state
    4. No race conditions occur under concurrent requests
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """
        Singleton pattern: ensure only one instance exists.
        Thread-safe double-checked locking.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Initialize API keys from environment.
        Called only once when singleton is created.
        """
        self._google_keys = self._load_google_keys()
        self._openai_key = os.getenv("OPENAI_API_KEY")
        self._current_index = 0
        self._rotation_lock = threading.Lock()

        logger.info(f"[APIKeyManager] Initialized with {len(self._google_keys)} Google key(s)")
        if self._openai_key:
            logger.info("[APIKeyManager] OpenAI key available for fallback")

    def _load_google_keys(self) -> List[str]:
        """
        Load all Google API keys from environment.

        Loads:
        - GOOGLE_API_KEY
        - GOOGLE_API_KEY2
        - GOOGLE_API_KEY3
        - ... up to GOOGLE_API_KEY10

        Returns:
            List of API keys (empty list if none found)
        """
        keys = []

        # Load primary key
        key = os.getenv("GOOGLE_API_KEY")
        if key:
            keys.append(key)
            logger.debug("[APIKeyManager] Loaded GOOGLE_API_KEY")

        # Load GOOGLE_API_KEY1 (supports both naming conventions)
        key = os.getenv("GOOGLE_API_KEY1")
        if key and key not in keys:  # Avoid duplicates
            keys.append(key)
            logger.debug("[APIKeyManager] Loaded GOOGLE_API_KEY1")

        # Load additional keys (GOOGLE_API_KEY2 through GOOGLE_API_KEY10)
        for i in range(2, 11):
            key = os.getenv(f"GOOGLE_API_KEY{i}")
            if key:
                keys.append(key)
                logger.debug(f"[APIKeyManager] Loaded GOOGLE_API_KEY{i}")

        if not keys:
            logger.warning("[APIKeyManager] No Google API keys found in environment")

        return keys

    def get_current_google_key(self) -> Optional[str]:
        """
        Get current Google API key (thread-safe).

        Returns:
            Current API key or None if no keys available

        Thread-safe: Uses lock to prevent race conditions
        """
        with self._rotation_lock:
            if not self._google_keys:
                return None
            return self._google_keys[self._current_index % len(self._google_keys)]

    def get_google_key_by_index(self, index: int) -> Optional[str]:
        """
        Get specific Google API key by index (thread-safe).

        Args:
            index: Key index (0-based)

        Returns:
            API key at index or None if index out of range
        """
        with self._rotation_lock:
            if not self._google_keys or index < 0 or index >= len(self._google_keys):
                return None
            return self._google_keys[index]

    def rotate_google_key(self) -> bool:
        """
        Rotate to next Google API key (thread-safe).

        Returns:
            True if rotation succeeded, False if only one key available

        Thread-safe: Uses lock to ensure atomic increment
        """
        with self._rotation_lock:
            if len(self._google_keys) <= 1:
                logger.debug("[APIKeyManager] Cannot rotate - only one key available")
                return False

            old_idx = self._current_index
            self._current_index = (self._current_index + 1) % len(self._google_keys)

            logger.info(
                f"[APIKeyManager] 🔄 Rotated API key: "
                f"#{old_idx + 1} -> #{self._current_index + 1} "
                f"(of {len(self._google_keys)})"
            )
            return True

    def rotate_to_specific_key(self, index: int) -> bool:
        """
        Rotate to specific key index (thread-safe).

        Args:
            index: Target key index (0-based)

        Returns:
            True if rotation succeeded, False if index invalid
        """
        with self._rotation_lock:
            if index < 0 or index >= len(self._google_keys):
                logger.warning(f"[APIKeyManager] Invalid key index: {index}")
                return False

            old_idx = self._current_index
            self._current_index = index

            logger.info(
                f"[APIKeyManager] Rotated to specific key: "
                f"#{old_idx + 1} -> #{self._current_index + 1}"
            )
            return True

    def get_openai_key(self) -> Optional[str]:
        """
        Get OpenAI API key for fallback.

        Returns:
            OpenAI API key or None if not configured
        """
        return self._openai_key

    def get_google_key_count(self) -> int:
        """
        Get total number of Google API keys available.

        Returns:
            Number of keys
        """
        return len(self._google_keys)

    def get_current_key_index(self) -> int:
        """
        Get current key index (thread-safe).

        Returns:
            Current index (0-based)
        """
        with self._rotation_lock:
            return self._current_index

    def has_multiple_keys(self) -> bool:
        """
        Check if multiple Google API keys are available for rotation.

        Returns:
            True if 2+ keys available
        """
        return len(self._google_keys) > 1

    def reset_rotation(self) -> None:
        """
        Reset rotation to first key (thread-safe).
        Useful for testing or manual reset.
        """
        with self._rotation_lock:
            old_idx = self._current_index
            self._current_index = 0
            logger.info(f"[APIKeyManager] Reset rotation: #{old_idx + 1} -> #1")

    def get_status(self) -> dict:
        """
        Get current status of API key manager.

        Returns:
            Status dictionary with key counts and current index
        """
        with self._rotation_lock:
            return {
                "google_keys_available": len(self._google_keys),
                "current_key_index": self._current_index,
                "rotation_enabled": len(self._google_keys) > 1,
                "openai_available": self._openai_key is not None,
            }

    def __repr__(self) -> str:
        """String representation for debugging"""
        status = self.get_status()
        return (
            f"APIKeyManager("
            f"google_keys={status['google_keys_available']}, "
            f"current_index={status['current_key_index']}, "
            f"openai={status['openai_available']})"
        )


# Global singleton instance
# Import this in all modules that need API keys
api_key_manager = APIKeyManager()


# Convenience functions for backward compatibility
def get_current_google_api_key() -> Optional[str]:
    """
    Get current Google API key (convenience function).
    Uses singleton instance internally.
    """
    return api_key_manager.get_current_google_key()


def rotate_google_api_key() -> bool:
    """
    Rotate Google API key (convenience function).
    Uses singleton instance internally.
    """
    return api_key_manager.rotate_google_key()


def get_openai_api_key() -> Optional[str]:
    """
    Get OpenAI API key (convenience function).
    Uses singleton instance internally.
    """
    return api_key_manager.get_openai_key()


if __name__ == "__main__":
    # Test API key manager
    print("Testing APIKeyManager...")
    print(f"Status: {api_key_manager.get_status()}")
    print(f"Current Google key: {api_key_manager.get_current_google_key()[:20] if api_key_manager.get_current_google_key() else 'None'}...")
    print(f"OpenAI key available: {api_key_manager.get_openai_key() is not None}")

    if api_key_manager.has_multiple_keys():
        print("\nTesting rotation...")
        for i in range(3):
            print(f"  Rotation {i+1}: ", end="")
            if api_key_manager.rotate_google_key():
                print(f"Success (now at index {api_key_manager.get_current_key_index()})")
            else:
                print("Failed (only one key)")
    else:
        print("\nRotation not available (only one key)")

    print(f"\nFinal status: {api_key_manager}")
