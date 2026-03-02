"""
Data Cache Module

Efficient data caching system.
"""

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DataCache:
    """
    Caching system for data and computed values.

    Features:
    - File-based caching
    - Multiple formats (pickle, JSON)
    - Cache validation
    - Automatic cleanup
    """

    def __init__(self, cache_dir: str = ".cache"):
        """
        Initialize cache.

        Args:
            cache_dir: Cache directory path
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str, format: str = "pickle") -> Path:
        """
        Get cache file path for key.

        Args:
            key: Cache key
            format: File format ('pickle' or 'json')

        Returns:
            Path to cache file
        """
        # Create safe filename from key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        ext = ".pkl" if format == "pickle" else ".json"
        return self.cache_dir / f"{safe_key}{ext}"

    def save(self, key: str, data: Any, format: str = "pickle") -> bool:
        """
        Save data to cache.

        Args:
            key: Cache key
            data: Data to cache
            format: File format ('pickle' or 'json')

        Returns:
            True if successful
        """
        try:
            path = self._get_cache_path(key, format)

            if format == "pickle":
                with open(path, "wb") as f:
                    pickle.dump(data, f)
            elif format == "json":
                with open(path, "w") as f:
                    json.dump(data, f)
            else:
                raise ValueError(f"Unknown format: {format}")

            logger.info(f"Cached data to {path}")
            return True

        except Exception as e:
            logger.error(f"Error saving cache: {e}")
            return False

    def load(self, key: str, format: str = "pickle") -> Optional[Any]:
        """
        Load data from cache.

        Args:
            key: Cache key
            format: File format ('pickle' or 'json')

        Returns:
            Cached data or None
        """
        try:
            path = self._get_cache_path(key, format)

            if not path.exists():
                return None

            if format == "pickle":
                with open(path, "rb") as f:
                    data = pickle.load(f)
            elif format == "json":
                with open(path, "r") as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Unknown format: {format}")

            logger.info(f"Loaded cached data from {path}")
            return data

        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None

    def is_cached(self, key: str, format: str = "pickle") -> bool:
        """
        Check if key is cached.

        Args:
            key: Cache key
            format: File format

        Returns:
            True if cached
        """
        path = self._get_cache_path(key, format)
        return path.exists()

    def clear(self, key: Optional[str] = None) -> bool:
        """
        Clear cache.

        Args:
            key: Specific key to clear (all if None)

        Returns:
            True if successful
        """
        try:
            if key is None:
                # Clear all
                import shutil

                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared entire cache")
            else:
                # Clear specific key
                for path in self.cache_dir.glob(
                    f"{hashlib.md5(key.encode()).hexdigest()}.*"
                ):
                    path.unlink()
                    logger.info(f"Cleared cache for key: {key}")

            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        total = 0
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total

    def get_cache_info(self) -> dict:
        """Get cache information."""
        files = list(self.cache_dir.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())

        return {
            "cache_dir": str(self.cache_dir),
            "file_count": file_count,
            "total_size": self.get_cache_size(),
        }
