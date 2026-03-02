"""
Performance Monitoring Module

Live performance monitoring and drift detection.
"""

import logging
from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitor trading performance and detect issues.

    Features:
    - Real-time metric tracking
    - Distribution shift detection
    - Alert generation
    - Daily reporting
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize monitor.

        Args:
            window_size: Rolling window size
        """
        self.window_size: int = window_size
        self.returns: Deque[float] = deque(maxlen=window_size)
        self.profits: Deque = deque(maxlen=window_size)
        self.timestamps: Deque = deque(maxlen=window_size)
        self.baseline_mean: Optional[float] = None
        self.baseline_std: Optional[float] = None

    def track_performance(
        self,
        return_value: float,
        profit: float,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Track performance metric.

        Args:
            return_value: Daily return
            profit: Daily profit
            timestamp: Timestamp (optional)
        """
        self.returns.append(return_value)
        self.profits.append(profit)
        self.timestamps.append(timestamp)

    def set_baseline(self) -> None:
        """Set baseline from current metrics."""
        if len(self.returns) > 10:
            self.baseline_mean = np.mean(self.returns)
            self.baseline_std = np.std(self.returns)
            logger.info(f"Baseline set: mean={self.baseline_mean:.4f}, std={self.baseline_std:.4f}")

    def detect_drift(self, threshold: float = 2.0) -> bool:
        """
        Detect distribution shift.

        Args:
            threshold: Std dev threshold

        Returns:
            True if drift detected
        """
        if self.baseline_mean is None or len(self.returns) < 10:
            return False

        current_mean = np.mean(self.returns)
        drift = abs(current_mean - self.baseline_mean) / max(self.baseline_std, 1e-8)

        if drift > threshold:
            logger.warning(f"Distribution drift detected: {drift:.2f} std devs")
            return True

        return False

    def alert(self, message: str, level: str = "WARNING") -> None:
        """
        Send alert.

        Args:
            message: Alert message
            level: Alert level
        """
        log_func = getattr(logger, level.lower())
        log_func(f"ALERT: {message}")

    def get_daily_report(self) -> Dict[str, float]:
        """Get daily performance summary."""
        if len(self.returns) == 0:
            return {}

        return {
            "daily_return": float(self.returns[-1]) if self.returns else 0.0,
            "daily_profit": float(self.profits[-1]) if self.profits else 0.0,
            "avg_return": float(np.mean(self.returns)),
            "sharpe_ratio": float(np.mean(self.returns) / max(np.std(self.returns), 1e-8)),
            "max_profit": float(np.max(self.profits)) if self.profits else 0.0,
            "min_profit": float(np.min(self.profits)) if self.profits else 0.0,
        }
