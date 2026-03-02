"""
Training Logger Module

Metrics tracking and logging during training.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TrainingLogger:
    """
    Track and log training metrics.

    Features:
    - Real-time metric tracking
    - File and console logging
    - Metrics summary statistics
    - Results export
    """

    def __init__(self, log_dir: str = "logs"):
        """
        Initialize training logger.

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics: Dict[str, List[float]] = {}
        self.steps: List[int] = []
        self.current_step = 0

        # Setup file logging
        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self._setup_file_logging()

    def _setup_file_logging(self) -> None:
        """Setup file logging handler."""
        file_handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Log scalar metric.

        Args:
            name: Metric name
            value: Metric value
            step: Training step (optional)
        """
        if step is None:
            step = self.current_step

        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append(float(value))

        if step not in self.steps:
            self.steps.append(step)

    def log_histogram(
        self,
        name: str,
        values: np.ndarray,
        step: Optional[int] = None,
    ) -> None:
        """
        Log histogram statistics.

        Args:
            name: Histogram name
            values: Values array
            step: Training step
        """
        stats = {
            f"{name}/mean": float(np.mean(values)),
            f"{name}/std": float(np.std(values)),
            f"{name}/min": float(np.min(values)),
            f"{name}/max": float(np.max(values)),
            f"{name}/median": float(np.median(values)),
        }

        for stat_name, stat_value in stats.items():
            self.log_scalar(stat_name, stat_value, step)

    def update_step(self, step: int) -> None:
        """
        Update current training step.

        Args:
            step: Current training step
        """
        self.current_step = step

    def get_summary(self, metric_name: Optional[str] = None) -> Dict[str, float]:
        """
        Get summary statistics for metrics.

        Args:
            metric_name: Specific metric (all if None)

        Returns:
            Dictionary with summary statistics
        """
        if metric_name:
            if metric_name not in self.metrics:
                return {}

            values = np.array(self.metrics[metric_name])
            return {
                f"{metric_name}/mean": float(np.mean(values)),
                f"{metric_name}/std": float(np.std(values)),
                f"{metric_name}/min": float(np.min(values)),
                f"{metric_name}/max": float(np.max(values)),
            }
        else:
            summary = {}
            for name in self.metrics:
                summary.update(self.get_summary(name))
            return summary

    def save_to_file(self, path: Optional[str] = None) -> str:
        """
        Save metrics to JSON file.

        Args:
            path: Save path (auto-generated if None)

        Returns:
            Path to saved file
        """
        if path is None:
            path = self.log_dir / "metrics.json"
        else:
            path = Path(path)

        data = {
            "metrics": {k: v for k, v in self.metrics.items()},
            "steps": self.steps,
            "summary": self.get_summary(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Metrics saved to {path}")
        return str(path)

    def log_message(self, message: str, level: str = "INFO") -> None:
        """
        Log message.

        Args:
            message: Message to log
            level: Logging level
        """
        log_func = getattr(logger, level.lower())
        log_func(message)

    def get_latest_metrics(self, n: int = 10) -> Dict[str, float]:
        """
        Get latest N metric values.

        Args:
            n: Number of latest values

        Returns:
            Dictionary with latest metrics
        """
        latest = {}
        for name, values in self.metrics.items():
            if len(values) > 0:
                latest[name] = float(np.mean(values[-n:]))
        return latest
