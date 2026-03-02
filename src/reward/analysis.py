"""
Reward Analysis Module

Analyzing reward signals during training.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RewardAnalyzer:
    """
    Analyze reward signals and quality.

    Features:
    - Distribution analysis
    - Outlier detection
    - Correlation analysis
    - Quality metrics
    """

    def __init__(self):
        """Initialize reward analyzer."""
        self.rewards_history: List[np.ndarray] = []

    def add_rewards(self, rewards: np.ndarray) -> None:
        """
        Add reward data.

        Args:
            rewards: Reward array
        """
        self.rewards_history.append(rewards.copy())

    def analyze_distribution(self, rewards: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Analyze reward distribution.

        Args:
            rewards: Reward array (uses latest if None)

        Returns:
            Dictionary with statistics
        """
        if rewards is None:
            if not self.rewards_history:
                return {}
            rewards = self.rewards_history[-1]

        if len(rewards) == 0:
            return {}

        analysis = {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "median": float(np.median(rewards)),
            "q25": float(np.percentile(rewards, 25)),
            "q75": float(np.percentile(rewards, 75)),
            "skewness": float(self._compute_skewness(rewards)),
            "kurtosis": float(self._compute_kurtosis(rewards)),
        }

        return analysis

    @staticmethod
    def _compute_skewness(data: np.ndarray) -> float:
        """Compute skewness."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))

    @staticmethod
    def _compute_kurtosis(data: np.ndarray) -> float:
        """Compute kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3)

    def detect_outliers(
        self,
        rewards: Optional[np.ndarray] = None,
        threshold: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Detect outliers in reward signal.

        Args:
            rewards: Reward array (uses latest if None)
            threshold: Outlier threshold (std deviations)

        Returns:
            Dictionary with outlier information
        """
        if rewards is None:
            if not self.rewards_history:
                return {}
            rewards = self.rewards_history[-1]

        mean = np.mean(rewards)
        std = np.std(rewards)

        # Identify outliers
        outlier_mask = np.abs(rewards - mean) > threshold * std
        outlier_indices = np.where(outlier_mask)[0]
        outlier_values = rewards[outlier_mask]

        return {
            "num_outliers": int(np.sum(outlier_mask)),
            "outlier_ratio": float(np.sum(outlier_mask) / len(rewards)),
            "outlier_indices": outlier_indices.tolist(),
            "outlier_values": outlier_values.tolist(),
        }

    def correlation_analysis(self) -> Dict[str, float]:
        """
        Analyze correlation between consecutive rewards.

        Returns:
            Dictionary with correlation metrics
        """
        if len(self.rewards_history) < 2:
            return {}

        # Use latest rewards
        rewards = self.rewards_history[-1]

        if len(rewards) < 2:
            return {}

        # Autocorrelation at lag 1
        lag1_corr = np.corrcoef(rewards[:-1], rewards[1:])[0, 1]

        # Serial correlation
        diffs = np.diff(rewards)
        mean_diff = np.mean(diffs)

        analysis = {
            "lag1_autocorr": float(lag1_corr) if not np.isnan(lag1_corr) else 0.0,
            "mean_diff": float(mean_diff),
            "reward_momentum": float(np.mean(np.sign(diffs))),
        }

        return analysis

    def reward_quality_metrics(self, rewards: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute reward quality metrics.

        Args:
            rewards: Reward array (uses latest if None)

        Returns:
            Dictionary with quality metrics
        """
        if rewards is None:
            if not self.rewards_history:
                return {}
            rewards = self.rewards_history[-1]

        # Check for dead rewards (all zeros)
        dead_ratio = float(np.sum(rewards == 0) / len(rewards))

        # Check for sparsity
        nonzero_ratio = float(np.sum(rewards != 0) / len(rewards))

        # Check variance
        variance = float(np.var(rewards))

        # Reward stability (low variation over windows)
        if len(rewards) > 10:
            window_vars = []
            for i in range(0, len(rewards) - 10, 10):
                window_vars.append(np.var(rewards[i : i + 10]))
            stability = float(np.mean(window_vars))
        else:
            stability = variance

        metrics = {
            "dead_ratio": dead_ratio,
            "nonzero_ratio": nonzero_ratio,
            "variance": variance,
            "stability": stability,
        }

        return metrics

    def generate_report(self) -> str:
        """
        Generate analysis report.

        Returns:
            Report string
        """
        if not self.rewards_history:
            return "No reward data available"

        latest_rewards = self.rewards_history[-1]

        dist_analysis = self.analyze_distribution(latest_rewards)
        outlier_analysis = self.detect_outliers(latest_rewards)
        corr_analysis = self.correlation_analysis()
        quality_metrics = self.reward_quality_metrics(latest_rewards)

        report = f"""
        ═══════════════════════════════════════════════════════════
                    REWARD ANALYSIS REPORT
        ═══════════════════════════════════════════════════════════
        DISTRIBUTION ANALYSIS
        ─────────────────────────────────────────────────────────
        Mean:               {dist_analysis['mean']:.4f}
        Std Dev:            {dist_analysis['std']:.4f}
        Min:                {dist_analysis['min']:.4f}
        Max:                {dist_analysis['max']:.4f}
        Median:             {dist_analysis['median']:.4f}
        Skewness:           {dist_analysis['skewness']:.4f}
        Kurtosis:           {dist_analysis['kurtosis']:.4f}
        OUTLIER DETECTION
        ─────────────────────────────────────────────────────────
        Num Outliers:       {outlier_analysis['num_outliers']}
        Outlier Ratio:      {outlier_analysis['outlier_ratio']:.4f}
        CORRELATION ANALYSIS
        ─────────────────────────────────────────────────────────
        Lag-1 Autocorr:     {corr_analysis.get('lag1_autocorr', 0.0):.4f}
        Reward Momentum:    {corr_analysis.get('reward_momentum', 0.0):.4f}
        REWARD QUALITY
        ─────────────────────────────────────────────────────────
        Dead Ratio:         {quality_metrics['dead_ratio']:.4f}
        Nonzero Ratio:      {quality_metrics['nonzero_ratio']:.4f}
        Variance:           {quality_metrics['variance']:.4f}
        Stability:          {quality_metrics['stability']:.4f}
        ═══════════════════════════════════════════════════════════
        """

        return report
