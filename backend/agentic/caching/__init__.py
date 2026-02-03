# agentic/caching/__init__.py
# Caching module for AIPR backend

from .validation_cache import ValidationCacheManager
from .bounded_cache_manager import (
    BoundedCache,
    BoundedCacheManager,
    get_or_create_cache,
    get_cache,
    cleanup_all_caches,
    clear_all_caches,
    get_all_cache_stats,
    get_registry_summary,
    managed_cache,
)

__all__ = [
    'ValidationCacheManager',
    # Bounded cache exports
    'BoundedCache',
    'BoundedCacheManager',
    'get_or_create_cache',
    'get_cache',
    'cleanup_all_caches',
    'clear_all_caches',
    'get_all_cache_stats',
    'get_registry_summary',
    'managed_cache',
]
