"""
Pytest Configuration and Fixtures

Shared test configuration and fixtures for all tests.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data import DataPreprocessor, get_loader
from src.environment import TradingEnv


@pytest.fixture(scope="function")
def sample_data():
    """Create realistic sample trading data."""
    loader = get_loader(source="parquet")
    df = loader.load("data/^NSEI_2019-01-01_2023-12-31.parquet")

    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess(df)

    return df_processed


@pytest.fixture(scope="function")
def sample_env(sample_data):
    """Create test environment with default configuration."""
    return TradingEnv(
        df=sample_data.values,
        initial_capital=100000.0,
        max_position=10,
        transaction_cost_pct=0.001,
        lookback_window=20,
        render_mode=None,
    )


@pytest.fixture
def sample_observations():
    """Generate sample observations for environment testing."""
    n_samples = 10
    obs_dim = 50
    return np.random.randn(n_samples, obs_dim).astype(np.float32)


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Create temporary model directory."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


@pytest.fixture
def tmp_log_dir(tmp_path):
    """Create temporary logging directory."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Create temporary cache directory."""
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def torch_model():
    """Create a simple torch model for testing."""
    model = torch.nn.Sequential(
        torch.nn.Linear(50, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 3),
    )
    return model


@pytest.fixture
def sample_config():
    """Sample configuration dictionary."""
    return {
        "initial_capital": 100000.0,
        "transaction_cost": 0.001,
        "slippage": 0.0005,
        "learning_rate": 0.0003,
        "batch_size": 64,
        "epochs": 100,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "entropy_coeff": 0.01,
    }


@pytest.fixture(scope="session")
def test_data_path(file):
    """Get path to test data directory."""
    return Path(file).parent / "test_data"


# Pytest hooks for test output
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "smoke: mark test as a smoke test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        if "test_" in item.nodeid and "integration" not in item.nodeid:
            item.add_marker(pytest.mark.unit)
