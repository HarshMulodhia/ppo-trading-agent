"""
Agent Evaluator Module

Comprehensive agent evaluation framework.
"""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class AgentEvaluator:
    """
    Comprehensive evaluation framework for trading agents.

    Features:
    - Multi-episode evaluation
    - Performance metrics computation
    - Baseline comparison
    - Detailed reporting
    """

    def __init__(self, env, agent):
        """
        Initialize evaluator.

        Args:
            env: Trading environment
            agent: Agent to evaluate
        """
        self.env = env
        self.agent = agent
        self.results = []

    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate agent performance.

        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy
            render: Render environment

        Returns:
            Dictionary with evaluation metrics
        """
        episode_returns = []
        episode_lengths = []

        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_return = 0
            steps = 0

            while not done:
                action, _ = self.agent.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, _ = self.env.step(action)

                episode_return += reward
                steps += 1

                if render:
                    self.env.render()

            episode_returns.append(episode_return)
            episode_lengths.append(steps)
            logger.info(f"Episode {episode+1}: Return={episode_return:.2f}, Length={steps}")

        metrics = self.compute_metrics(episode_returns, episode_lengths)
        self.results.append(metrics)
        return metrics

    def compute_metrics(
        self,
        returns: List[float],
        lengths: List[int],
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            returns: Episode returns
            lengths: Episode lengths

        Returns:
            Dictionary with metrics
        """
        returns_arr = np.array(returns)
        lengths_arr = np.array(lengths)

        metrics = {
            "mean_return": float(np.mean(returns_arr)),
            "std_return": float(np.std(returns_arr)),
            "min_return": float(np.min(returns_arr)),
            "max_return": float(np.max(returns_arr)),
            "median_return": float(np.median(returns_arr)),
            "mean_length": float(np.mean(lengths_arr)),
            "std_length": float(np.std(lengths_arr)),
        }

        return metrics

    def compare_baseline(
        self,
        baseline_agent,
        n_episodes: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare agent against baseline.

        Args:
            baseline_agent: Baseline agent for comparison
            n_episodes: Number of evaluation episodes

        Returns:
            Dictionary with comparison results
        """
        agent_metrics = self.evaluate(n_episodes=n_episodes, deterministic=True)

        # Evaluate baseline
        original_agent = self.agent
        self.agent = baseline_agent
        baseline_metrics = self.evaluate(n_episodes=n_episodes, deterministic=True)
        self.agent = original_agent

        comparison = {
            "agent": agent_metrics,
            "baseline": baseline_metrics,
            "improvement": {
                "return_improvement": agent_metrics["mean_return"]
                - baseline_metrics["mean_return"],
                "return_ratio": agent_metrics["mean_return"]
                / max(baseline_metrics["mean_return"], 1e-8),
            },
        }

        return comparison

    def generate_report(self) -> str:
        """
        Generate evaluation report.

        Returns:
            Report string
        """
        if not self.results:
            return "No evaluation results available"

        latest = self.results[-1]

        report = f"""
        ═══════════════════════════════════════════════════════════
                        EVALUATION REPORT
        ═══════════════════════════════════════════════════════════
        Mean Return:         {latest['mean_return']:.4f}
        Std Return:          {latest['std_return']:.4f}
        Min Return:          {latest['min_return']:.4f}
        Max Return:          {latest['max_return']:.4f}
        Median Return:       {latest['median_return']:.4f}
        Mean Episode Length: {latest['mean_length']:.0f}
        Std Episode Length:  {latest['std_length']:.0f}
        ═══════════════════════════════════════════════════════════
        """

        return report


class EvaluationResult:
    """Container for evaluation results."""

    def __init__(
        self,
        returns: List[float],
        lengths: List[int],
        actions: List[int],
        observations: List[np.ndarray],
    ):
        """
        Initialize result container.

        Args:
            returns: Episode returns
            lengths: Episode lengths
            actions: Actions taken
            observations: Observations
        """
        self.returns = np.array(returns)
        self.lengths = np.array(lengths)
        self.actions = np.array(actions)
        self.observations = np.array(observations)

    def summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "mean_return": float(np.mean(self.returns)),
            "std_return": float(np.std(self.returns)),
            "mean_length": float(np.mean(self.lengths)),
        }
