"""
Evaluation Package - Performance Metrics and Backtesting

Provides tools for evaluating trading agent performance.
"""

from .backtester import Backtester, Portfolio, Trade
from .evaluator import AgentEvaluator, EvaluationResult
from .metrics import (
    PerformanceMetrics,
    clip_values,
    compute_metrics,
    normalize_features,
    validate_prices,
)
from .visualizer import ResultsVisualizer

__all__ = [
    "Trade",
    "Portfolio",
    "Backtester",
    "AgentEvaluator",
    "EvaluationResult",
    "PerformanceMetrics",
    "compute_metrics",
    "clip_values",
    "normalize_features",
    "validate_prices",
    "ResultsVisualizer",
]
