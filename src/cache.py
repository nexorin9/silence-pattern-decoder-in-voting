"""
Cache module for expensive simulation computations.
Provides caching layer to avoid re-computing simulation results.
"""

import hashlib
import json
import os
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "cache"
DEFAULT_TTL_SECONDS = 3600  # 1 hour


class CacheEntry:
    """Represents a single cache entry with metadata."""

    def __init__(
        self,
        key: str,
        value: Any,
        created_at: float,
        last_accessed: float,
        access_count: int = 1,
    ):
        self.key = key
        self.value = value
        self.created_at = created_at
        self.last_accessed = last_accessed
        self.access_count = access_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create entry from dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            created_at=data["created_at"],
            last_accessed=data["last_accessed"],
            access_count=data.get("access_count", 1),
        )

    def touch(self):
        """Update last accessed time."""
        self.last_accessed = time.time()
        self.access_count += 1


class SimulationCache:
    """Cache for simulation results with TTL and cleanup support."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        default_ttl: int = DEFAULT_TTL_SECONDS,
        max_entries: int = 1000,
    ):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.default_ttl = default_ttl
        self.max_entries = max_entries
        self._entries: Dict[str, CacheEntry] = {}
        self._index_path = self.cache_dir / "index.json"
        self._ensure_cache_dir()

        # Load existing index if available
        self._load_index()

    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self):
        """Load cache index from disk."""
        if self._index_path.exists():
            try:
                with open(self._index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for entry_data in data:
                        entry = CacheEntry.from_dict(entry_data)
                        self._entries[entry.key] = entry
            except (json.JSONDecodeError, IOError) as e:
                # Invalid cache, start fresh
                self._entries = {}

    def _save_index(self):
        """Save cache index to disk."""
        try:
            entries_list = [entry.to_dict() for entry in self._entries.values()]
            with open(self._index_path, "w", encoding="utf-8") as f:
                json.dump(entries_list, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save cache index: {e}")

    def _compute_key(self, params: Dict[str, Any]) -> str:
        """Compute cache key from parameters."""
        # Sort keys for consistent hashing
        param_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]

    def _is_expired(self, entry: CacheEntry, ttl: Optional[int] = None) -> bool:
        """Check if cache entry has expired."""
        current_ttl = ttl if ttl is not None else self.default_ttl
        return time.time() - entry.created_at > current_ttl

    def get(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key
            ttl: Override TTL for this check

        Returns:
            Cached value or None if not found/expired
        """
        entry = self._entries.get(key)
        if entry is None:
            return None

        if self._is_expired(entry, ttl):
            del self._entries[key]
            self._save_index()
            return None

        entry.touch()
        self._save_index()
        return entry.value

    def set(
        self, key: str, value: Any, ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Override TTL for this entry

        Returns:
            True if successful
        """
        # Evict oldest entries if at capacity
        if len(self._entries) >= self.max_entries:
            self._evict_oldest()

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
        )
        self._entries[key] = entry
        self._save_index()
        return True

    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry existed and was deleted
        """
        if key in self._entries:
            del self._entries[key]
            self._save_index()
            return True
        return False

    def clear(self):
        """Clear all cache entries."""
        self._entries = {}
        if self._index_path.exists():
            self._index_path.unlink()

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = [
            key
            for key, entry in self._entries.items()
            if self._is_expired(entry)
        ]
        for key in expired_keys:
            del self._entries[key]
        self._save_index()
        return len(expired_keys)

    def _evict_oldest(self):
        """Remove oldest cache entry."""
        if not self._entries:
            return

        oldest_key = min(
            self._entries.keys(), key=lambda k: self._entries[k].created_at
        )
        del self._entries[oldest_key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_records = len(self._entries)
        total_accesses = sum(entry.access_count for entry in self._entries.values())
        avg_accesses = (
            total_accesses / total_records if total_records > 0 else 0
        )
        return {
            "total_entries": total_records,
            "total_accesses": total_accesses,
            "avg_accesses_per_entry": avg_accesses,
            "cache_dir": str(self.cache_dir),
            "max_entries": self.max_entries,
        }


# Global cache instance
_cache_instance: Optional[SimulationCache] = None


def get_cache(
    cache_dir: Optional[Path] = None, ttl: Optional[int] = None
) -> SimulationCache:
    """Get or create global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SimulationCache(cache_dir=cache_dir, default_ttl=ttl or DEFAULT_TTL_SECONDS)
    return _cache_instance


def cached(
    ttl: Optional[int] = None,
    cache: Optional[SimulationCache] = None,
    key_prefix: str = "",
) -> Callable:
    """
    Decorator for caching function results.

    Args:
        ttl: Cache TTL in seconds (uses default if not specified)
        cache: Cache instance (uses global if not specified)
        key_prefix: Prefix for cache keys

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        cache_instance = cache or get_cache()

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Compute cache key from function name and arguments
            params = {
                "function": f"{key_prefix}:{func.__name__}",
                "args": args,
                "kwargs": kwargs,
            }
            key = cache_instance._compute_key(params)

            # Try to get from cache
            cached_value = cache_instance.get(key, ttl=ttl)
            if cached_value is not None:
                return cached_value

            # Call function and cache result
            result = func(*args, **kwargs)
            cache_instance.set(key, result, ttl=ttl)
            return result

        return wrapper

    return decorator


def cached_method(
    ttl: Optional[int] = None, key_prefix: str = ""
) -> Callable:
    """
    Decorator for caching class method results.

    Args:
        ttl: Cache TTL in seconds
        key_prefix: Prefix for cache keys

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        cache_instance = get_cache()

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Compute cache key including class name
            params = {
                "class": self.__class__.__name__,
                "function": f"{key_prefix}:{func.__name__}",
                "args": args,
                "kwargs": kwargs,
            }
            key = cache_instance._compute_key(params)

            cached_value = cache_instance.get(key, ttl=ttl)
            if cached_value is not None:
                return cached_value

            result = func(self, *args, **kwargs)
            cache_instance.set(key, result, ttl=ttl)
            return result

        return wrapper

    return decorator


class CacheStatistics:
    """Track cache statistics across the application."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.invalidations = 0

    def record_hit(self):
        """Record a cache hit."""
        self.hits += 1

    def record_miss(self):
        """Record a cache miss."""
        self.misses += 1

    def record_invalidation(self):
        """Record a cache invalidation."""
        self.invalidations += 1

    def get_stats(self) -> Dict[str, int]:
        """Get statistics as dictionary."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "invalidations": self.invalidations,
            "hit_rate": round(hit_rate, 4),
            "total_requests": total,
        }

    def reset(self):
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
        self.invalidations = 0


# Global statistics instance
_statistics = CacheStatistics()


def get_statistics() -> CacheStatistics:
    """Get global statistics instance."""
    return _statistics
