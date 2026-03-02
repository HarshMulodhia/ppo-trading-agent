#!/usr/bin/env python3
"""
Baseline Comparison Script

Compare PPO agent against buy-and-hold and other baselines.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import save_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineComparison:
    """Compare agent performance against baselines."""

    def __init__(self, initial_capital=100000):
        """Initialize comparison."""
        self.initial_capital = initial_capital
        self.results = {}

    def buy_and_hold(self, prices):
        """
        Buy-and-hold baseline.

        Args:
            prices: Price series

        Returns:
            Returns and equity curve
        """
        # Buy at start, hold until end
        initial_price = prices[0]
        shares = self.initial_capital / initial_price

        equity = shares * prices
        returns = (equity - self.initial_capital) / self.initial_capital

        return returns, equity

    def buy_and_hold_with_rebalance(self, prices, rebalance_freq=252):
        """
        Buy-and-hold with periodic rebalancing.

        Args:
            prices: Price series
            rebalance_freq: Rebalancing frequency (days)

        Returns:
            Returns and equity curve
        """
        equity = np.ones(len(prices)) * self.initial_capital

        for i in range(len(prices)):
            if i % rebalance_freq == 0:
                # Rebalance
                current_price = prices[i]
                shares = (
                    equity[i - 1] / current_price
                    if i > 0
                    else self.initial_capital / current_price
                )

            if i > 0:
                equity[i] = shares * prices[i]

        returns = (equity - self.initial_capital) / self.initial_capital
        return returns, equity

    def momentum(self, prices, lookback=20):
        """
        Simple momentum strategy.

        Args:
            prices: Price series
            lookback: Lookback window

        Returns:
            Returns and equity curve
        """
        equity = np.ones(len(prices)) * self.initial_capital
        position = 0  # 0: no position, 1: long

        for i in range(lookback, len(prices)):
            # Calculate momentum
            momentum = prices[i] - prices[i - lookback]

            if momentum > 0 and position == 0:
                # Go long
                position = 1
                shares = equity[i - 1] / prices[i]
            elif momentum <= 0 and position == 1:
                # Exit
                position = 0
                equity[i] = shares * prices[i]

            if position == 1:
                equity[i] = shares * prices[i]
            else:
                equity[i] = equity[i - 1]

        returns = (equity - self.initial_capital) / self.initial_capital
        return returns, equity

    def compare_all(self, prices, ppo_equity, output_dir=None):
        """
        Compare all baselines.

        Args:
            prices: Price series
            ppo_equity: PPO agent equity curve
            output_dir: Output directory for results

        Returns:
            Comparison results
        """
        results = {}

        # PPO
        ppo_return = (ppo_equity[-1] - self.initial_capital) / self.initial_capital
        results["PPO"] = {
            "total_return": float(ppo_return),
            "final_equity": float(ppo_equity[-1]),
        }

        # Buy and hold
        bh_returns, bh_equity = self.buy_and_hold(prices)
        results["Buy-and-Hold"] = {
            "total_return": float(bh_returns[-1]),
            "final_equity": float(bh_equity[-1]),
        }

        # Buy and hold with rebalance
        bhr_returns, bhr_equity = self.buy_and_hold_with_rebalance(prices)
        results["Buy-Hold-Rebalance"] = {
            "total_return": float(bhr_returns[-1]),
            "final_equity": float(bhr_equity[-1]),
        }

        # Momentum
        mom_returns, mom_equity = self.momentum(prices)
        results["Momentum"] = {
            "total_return": float(mom_returns[-1]),
            "final_equity": float(mom_equity[-1]),
        }

        self.results = results

        if output_dir:
            save_json(results, str(Path(output_dir) / "comparison_results.json"))

        return results


def main():
    """Run baseline comparison."""
    parser = argparse.ArgumentParser(description="Compare against baselines")
    parser.add_argument("--data", type=str, required=True, help="Test data path")
    parser.add_argument(
        "--agent-equity", type=str, required=True, help="Agent equity file"
    )
    parser.add_argument(
        "--output", type=str, default="comparison_results", help="Output directory"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Loading data and agent results...")

        # Load data
        data = pd.read_parquet(args.data)
        prices = data["close"].values

        # Load agent equity
        agent_results = pd.read_csv(args.agent_equity)
        ppo_equity = agent_results["equity"].values

        # Compare
        logger.info("Running baseline comparison...")
        comparator = BaselineComparison()
        results = comparator.compare_all(prices, ppo_equity, output_dir=str(output_dir))

        # Print results
        print("\n" + "=" * 70)
        print("BASELINE COMPARISON RESULTS")
        print("=" * 70)
        print(f"{'Strategy':<20} {'Total Return':<15} {'Final Equity':<15}")
        print("-" * 70)

        for strategy, metrics in results.items():
            ret = metrics["total_return"]
            equity = metrics["final_equity"]
            print(f"{strategy:<20} {ret:>13.2%} {equity:>13,.0f}")

        print("=" * 70 + "\n")

        # Determine winner
        best_strategy = max(results.items(), key=lambda x: x[1]["total_return"])
        logger.info(
            f"Best strategy: {best_strategy[0]} ({best_strategy[1]['total_return']:.2%})"
        )

    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        raise


if __name__ == "__main__":
    main()
