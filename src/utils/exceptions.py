"""
Exceptions Module

Custom exception classes for trading system.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


class TradingException(Exception):
    """Base exception for trading system."""

    pass


class InsufficientCapitalError(TradingException):
    """Raised when insufficient capital for trade."""

    def __init__(self, required: float, available: float):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient capital: required {required:.2f}, available {available:.2f}"
        )


class InvalidOrderError(TradingException):
    """Raised when order is invalid."""

    def __init__(self, message: str, order_details: Optional[Mapping[str, Any]] = None):
        self.order_details = order_details
        super().__init__(message)


class PositionLimitError(TradingException):
    """Raised when position limit exceeded."""

    def __init__(self, position_size: float, max_size: float):
        self.position_size = position_size
        self.max_size = max_size
        super().__init__(f"Position {position_size:.2f} exceeds limit {max_size:.2f}")


class DataQualityError(TradingException):
    """Raised when data quality check fails."""

    def __init__(self, message: str, column: Optional[str] = None):
        self.column = column
        super().__init__(message)


class DataLoadError(TradingException):
    """Raised when data loading fails."""

    def __init__(self, source: str, message: str):
        self.source = source
        super().__init__(f"Error loading data from {source}: {message}")


class ModelError(TradingException):
    """Raised when model operation fails."""

    def __init__(self, message: str):
        super().__init__(message)


class ConfigurationError(TradingException):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, config_key: Optional[str] = None):
        self.config_key = config_key
        super().__init__(message)


class CacheError(TradingException):
    """Raised when cache operation fails."""

    def __init__(self, operation: str, key: str, message: str):
        self.operation = operation
        self.key = key
        super().__init__(f"Cache {operation} error for key '{key}': {message}")


class EnvironmentError(TradingException):
    """Raised when environment operation fails."""

    def __init__(self, message: str):
        super().__init__(message)


class RewardError(TradingException):
    """Raised when reward computation fails."""

    def __init__(self, message: str):
        super().__init__(message)


class ValidationError(TradingException):
    """Raised when validation fails."""

    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        super().__init__(message)
