"""
Data Pipeline Tests

Unit tests for data loading, preprocessing, and caching.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import *


class TestDataLoader:
    """Test data loading functionality."""

    def test_get_loader_csv(self):
        """Test CSV loader retrieval."""
        loader = get_loader("csv")
        assert loader is not None

    def test_get_loader_yahoo(self):
        """Test Yahoo Finance loader retrieval."""
        loader = get_loader("yahoo")
        assert loader is not None

    def test_get_loader_parquet(self):
        """Test Parquet loader retrieval."""
        loader = get_loader("parquet")
        assert loader is not None

    def test_invalid_loader(self):
        """Test invalid loader."""
        with pytest.raises(ValueError):
            get_loader("invalid_source")


class TestDataPreprocessor:
    """Test data preprocessing functionality."""

    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor is not None

    def test_clean_forward_fill(self, sample_data):
        """Test forward fill cleaning."""
        # Introduce NaN
        data = sample_data.copy()
        data.loc[data.index[10], "close"] = np.nan

        cleaned = DataPreprocessor.clean(data, method="forward_fill")

        assert not cleaned.isnull().any().any()

    def test_remove_outliers(self, sample_data):
        """Test outlier removal."""
        data = sample_data.copy()

        cleaned = DataPreprocessor.remove_outliers(data, threshold=3.0)

        assert len(cleaned) <= len(data)

    def test_normalize_prices(self, sample_data):
        """Test price normalization."""
        data = sample_data.copy()

        normalized = DataPreprocessor.normalize_prices(data, method="minmax")

        # Check values are in [0, 1]
        assert normalized["close"].min() >= 0
        assert normalized["close"].max() <= 1

    def test_compute_returns(self, sample_data):
        """Test returns computation."""
        data = sample_data.copy()

        with_returns = DataPreprocessor.compute_returns(data)

        assert "returns" in with_returns.columns
        assert len(with_returns) == len(data) - 1


class TestDataSplitter:
    """Test data splitting functionality."""

    def test_init(self, sample_data):
        """Test splitter initialization."""
        splitter = DataSplitter(sample_data)
        assert splitter is not None
        assert len(splitter) == len(sample_data)

    def test_time_series_split(self, sample_data):
        """Test time series splitting."""
        splitter = DataSplitter(sample_data)
        train, val, test = splitter.time_series_split(train_ratio=0.6, val_ratio=0.2)

        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == len(sample_data)

    def test_rolling_split(self, sample_data):
        """Test rolling split."""
        splitter = DataSplitter(sample_data)

        splits = list(splitter.rolling_split(window_size=100, step_size=50))

        assert len(splits) > 0
        for train, test in splits:
            assert len(train) > 0
            assert len(test) > 0


class TestDataCache:
    """Test data caching functionality."""

    def test_save_and_load(self, tmp_cache_dir, sample_data):
        """Test saving and loading data."""
        cache = DataCache(cache_dir=str(tmp_cache_dir))

        cache.save("test_data", sample_data)
        loaded = cache.load("test_data")

        assert loaded is not None
        assert len(loaded) == len(sample_data)

    def test_is_cached(self, tmp_cache_dir, sample_data):
        """Test cache check."""
        cache = DataCache(cache_dir=str(tmp_cache_dir))

        assert not cache.is_cached("test_data")

        cache.save("test_data", sample_data)

        assert cache.is_cached("test_data")

    def test_clear_specific(self, tmp_cache_dir, sample_data):
        """Test clearing specific cache."""
        cache = DataCache(cache_dir=str(tmp_cache_dir))

        cache.save("test_data_1", sample_data)
        cache.save("test_data_2", sample_data)

        cache.clear("test_data_1")

        assert not cache.is_cached("test_data_1")
        assert cache.is_cached("test_data_2")

    def test_cache_info(self, tmp_cache_dir, sample_data):
        """Test cache info."""
        cache = DataCache(cache_dir=str(tmp_cache_dir))

        cache.save("test_data", sample_data)

        info = cache.get_cache_info()

        assert "cache_dir" in info
        assert "file_count" in info
        assert "total_size" in info


class TestDataEdgeCases:
    """Test edge cases in data handling."""

    def test_missing_columns(self):
        """Test handling of missing columns."""
        data = pd.DataFrame(
            {
                "open": [100],
                "close": [101],
                # Missing high, low, volume
            }
        )

        with pytest.raises(ValueError):
            DataLoader.validate_data(data)

    def test_duplicate_indices(self):
        """Test handling of duplicate indices."""
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        # Create duplicate index
        dates = dates.append(dates[:1])

        data = pd.DataFrame(
            {
                "open": [100] * 6,
                "high": [102] * 6,
                "low": [99] * 6,
                "close": [101] * 6,
                "volume": [1e6] * 6,
            },
            index=dates,
        )

        cleaned = DataLoader.validate_data(data)
        assert cleaned.index.is_unique
        assert len(cleaned) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
