"""
Data Loader Module

Loading market data from various sources.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Base data loader class.

    Features:
    - Flexible data loading
    - Source abstraction
    - Data validation
    - Error handling
    - MultiIndex handling
    """

    def __init__(self):
        """Initialize base loader."""
        pass

    def load(self, **kwargs: Any) -> pd.DataFrame:
        """
        Load data.

        Args:
            identifier: Stock symbol or data identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            **kwargs: Additional arguments

        Returns:
            DataFrame with OHLCV data
        """
        raise NotImplementedError

    @staticmethod
    def flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten MultiIndex columns to single level.

        Args:
            data: DataFrame with potentially MultiIndex columns

        Returns:
            DataFrame with single-level columns
        """
        if isinstance(data.columns, pd.MultiIndex):
            # If MultiIndex, flatten it by taking first level
            # This handles yfinance multiple ticker downloads
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

        return data

    @staticmethod
    def validate_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data quality.

        Args:
            data: DataFrame to validate

        Returns:
            Dataframe
        """
        required_columns = ["open", "high", "low", "close", "volume"]

        # Check columns (case-insensitive)
        data_cols_lower = [col.lower() for col in data.columns]
        for col in required_columns:
            if col not in data_cols_lower:
                raise ValueError(f"Missing required column: {col}")

        # Check for NaN
        if data.isnull().any().any():
            logger.warning("Data contains NaN values - filling with forward fill")
            data = data.ffill().bfill()

        # Check for duplicates
        if data.index.duplicated().any():
            logger.warning("Data contains duplicate indices - removing duplicates")
            data = data[~data.index.duplicated(keep="first")]

        # Check monotonic index
        if not data.index.is_monotonic_increasing:
            logger.warning("Data index is not monotonic - sorting by index")
            data = data.sort_index()

        return data


class CSVLoader(DataLoader):
    """Load data from CSV files."""

    def load(
        self,
        filepath: str,
        date_column: str = "date",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from CSV.

        Args:
            filepath: Path to CSV file
            date_column: Column name for dates
            **kwargs: Additional pandas arguments

        Returns:
            DataFrame with OHLCV data
        """
        try:
            data = pd.read_csv(filepath, **kwargs)

            # Set date as index
            if date_column in data.columns:
                data[date_column] = pd.to_datetime(data[date_column])
                data = data.set_index(date_column)

            # Flatten columns if MultiIndex
            data = self.flatten_columns(data)

            # Lowercase column names
            data.columns = [col.lower() for col in data.columns]

            # Validate
            self.validate_data(data)

            logger.info(f"Loaded {len(data)} records from {filepath}")
            return data

        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise


class YahooFinanceLoader(DataLoader):
    """Load data from Yahoo Finance."""

    def load(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from Yahoo Finance.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', '^NSEI')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            **kwargs: Additional arguments

        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf

            # Download data
            data = yf.download(symbol, start=start_date, end=end_date, progress=False, **kwargs)

            # Handle empty data
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            # Flatten columns if MultiIndex
            data = self.flatten_columns(data)

            # Lowercase column names - check if .str accessor is available
            if isinstance(data.columns, pd.Index):
                data.columns = [col.lower() for col in data.columns]
            else:
                # Fallback for any other column type
                data.columns = [str(col).lower() for col in data.columns]

            # Validate
            self.validate_data(data)

            logger.info(f"Loaded {len(data)} records for {symbol} from Yahoo Finance")
            return data

        except ImportError:
            logger.error("yfinance not installed. Install with: pip install yfinance")
            raise
        except Exception as e:
            logger.error(f"Error loading from Yahoo Finance: {e}")
            raise


class NSELoader(DataLoader):
    """Load data from NSE India."""

    def load(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from NSE India.

        Args:
            symbol: Stock symbol or index (e.g., 'INFY', '^NSEI')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            **kwargs: Additional arguments

        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf

            # Handle index symbols (starting with ^)
            if symbol.startswith("^"):
                # Index symbol - use as is
                nse_symbol = symbol
            else:
                # Stock symbol - add .NS suffix if not present
                nse_symbol = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol

            # Download data
            data = yf.download(nse_symbol, start=start_date, end=end_date, progress=False, **kwargs)

            # Handle empty data
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            # Flatten columns if MultiIndex
            data = self.flatten_columns(data)

            # Lowercase column names
            if isinstance(data.columns, pd.Index):
                data.columns = [col.lower() for col in data.columns]
            else:
                data.columns = [str(col).lower() for col in data.columns]

            # Validate
            self.validate_data(data)

            logger.info(f"Loaded {len(data)} records for {symbol} from NSE")
            return data

        except ImportError:
            logger.error("yfinance not installed. Install with: pip install yfinance")
            raise
        except Exception as e:
            logger.error(f"Error loading from NSE: {e}")
            raise


class ParquetLoader(DataLoader):
    """Load data from Parquet files."""

    def load(
        self,
        filepath: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from Parquet.

        Args:
            filepath: Path to Parquet file
            **kwargs: Additional pandas arguments

        Returns:
            DataFrame with OHLCV data
        """
        try:
            data = pd.read_parquet(filepath, **kwargs)

            # Flatten columns if MultiIndex
            data = self.flatten_columns(data)

            # Lowercase column names
            if isinstance(data.columns, pd.Index):
                data.columns = [col.lower() for col in data.columns]
            else:
                data.columns = [str(col).lower() for col in data.columns]

            # Validate
            self.validate_data(data)

            logger.info(f"Loaded {len(data)} records from {filepath}")
            return data

        except Exception as e:
            logger.error(f"Error loading Parquet: {e}")
            raise


def get_loader(source: str) -> DataLoader:
    """
    Get appropriate loader for source.

    Args:
        source: Data source type ('csv', 'yahoo', 'nse', 'parquet')

    Returns:
        DataLoader instance
    """
    loaders = {
        "csv": CSVLoader,
        "yahoo": YahooFinanceLoader,
        "nse": NSELoader,
        "parquet": ParquetLoader,
    }

    if source.lower() not in loaders:
        raise ValueError(f"Unknown source: {source}")

    return loaders[source.lower()]()
