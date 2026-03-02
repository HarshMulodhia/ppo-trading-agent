"""
Data Preprocessor Module

Data cleaning and normalization.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessing and feature engineering.

    Features:
    - Missing data handling
    - Outlier detection and removal
    - Price normalization
    - Feature engineering
    """

    def __init__(self):
        """Initialize preprocessor."""
        pass

    @staticmethod
    def clean(data: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame:
        """
        Clean data by handling missing values.

        Args:
            data: Input DataFrame
            method: Cleaning method ('forward_fill', 'interpolate', 'drop')

        Returns:
            Cleaned DataFrame
        """
        data = data.copy()

        if method == "forward_fill":
            data = data.ffill().bfill()
        elif method == "interpolate":
            data = data.interpolate(method="linear")
        elif method == "drop":
            data = data.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Data cleaned using {method}")
        return data

    @staticmethod
    def remove_outliers(
        data: pd.DataFrame,
        columns: Optional[list] = None,
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        """
        Remove outliers using z-score method.

        Args:
            data: Input DataFrame
            columns: Columns to check (all if None)
            threshold: Z-score threshold

        Returns:
            DataFrame with outliers removed
        """
        data = data.copy()

        if columns is None:
            columns = data.columns

        # Calculate z-scores
        z_scores = np.abs((data[columns] - data[columns].mean()) / data[columns].std())

        # Remove outliers
        mask = (z_scores < threshold).all(axis=1)
        removed = len(data) - mask.sum()
        data = data[mask]

        logger.info(f"Removed {removed} outlier records")
        return data

    @staticmethod
    def normalize_prices(
        data: pd.DataFrame,
        method: str = "minmax",
    ) -> pd.DataFrame:
        """
        Normalize price data.

        Args:
            data: Input DataFrame
            method: Normalization method ('minmax', 'zscore')

        Returns:
            Normalized DataFrame
        """
        data = data.copy()
        price_cols = ["open", "high", "low", "close"]

        if method == "minmax":
            for col in price_cols:
                if col in data.columns:
                    col_min = data[col].min()
                    col_max = data[col].max()
                    data[col] = (data[col] - col_min) / (col_max - col_min)

        elif method == "zscore":
            for col in price_cols:
                if col in data.columns:
                    data[col] = (data[col] - data[col].mean()) / data[col].std()

        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Prices normalized using {method}")
        return data

    @staticmethod
    def compute_returns(data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute log returns.

        Args:
            data: Input DataFrame with 'close' column

        Returns:
            DataFrame with 'returns' column added
        """
        data = data.copy()
        data["returns"] = np.log(data["close"] / data["close"].shift(1))
        data = data.dropna()

        logger.info(f"Computed returns for {len(data)} records")
        return data

    @staticmethod
    def normalize_volume(data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize trading volume.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with normalized volume
        """
        data = data.copy()

        if "volume" in data.columns:
            vol_mean = data["volume"].mean()
            vol_std = data["volume"].std()
            data["volume_norm"] = (data["volume"] - vol_mean) / vol_std

        return data

    @staticmethod
    def create_price_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with additional features
        """
        data = data.copy()

        # High-Low ratio
        data["hl_ratio"] = (data["high"] - data["low"]) / data["close"]

        # Close-Open ratio
        data["co_ratio"] = (data["close"] - data["open"]) / data["open"]

        # Upper and Lower shadows
        data["upper_shadow"] = (data["high"] - np.maximum(data["open"], data["close"])) / data[
            "close"
        ]
        data["lower_shadow"] = (np.minimum(data["open"], data["close"]) - data["low"]) / data[
            "close"
        ]

        return data

    def preprocess(
        self,
        data: pd.DataFrame,
        remove_outliers: bool = True,
        normalize: bool = True,
        compute_returns: bool = False,
        create_features: bool = False,
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline.

        Args:
            data: Input DataFrame
            remove_outliers: Whether to remove outliers
            normalize: Whether to normalize prices
            compute_returns: Whether to compute returns
            create_features: Whether to create features

        Returns:
            Preprocessed DataFrame
        """
        # Clean
        data = self.clean(data)

        # Remove outliers
        if remove_outliers:
            data = self.remove_outliers(data)

        # Normalize
        if normalize:
            data = self.normalize_prices(data)

        # Compute returns
        if compute_returns:
            data = self.compute_returns(data)

        # Create features
        if create_features:
            data = self.create_price_features(data)

        # Normalize volume
        data = self.normalize_volume(data)

        logger.info("Preprocessing complete")
        return data
