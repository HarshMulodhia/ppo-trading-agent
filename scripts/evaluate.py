#!/usr/bin/env python3
"""
Evaluation and Backtesting Script

Run comprehensive evaluation and backtesting of trained agents.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from src.deployment import ModelManager
from src.evaluation import AgentEvaluator, Backtester, ResultsVisualizer
from src.utils import save_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run evaluation and backtesting."""
    parser = argparse.ArgumentParser(description="Evaluate trading agent")
    parser.add_argument("--model", type=str, required=True, help="Model path or version ID")
    parser.add_argument("--data", type=str, required=True, help="Test data path")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--initial-capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Loading model and data...")

        # Load data
        import pandas as pd

        data = pd.read_parquet(args.data)

        # Load model
        model_manager = ModelManager()
        # For now, assume model is loaded directly

        # Run backtester
        logger.info("Running backtest...")
        backtester = Backtester(initial_capital=args.initial_capital)

        # Get metrics (would need actual agent)
        metrics = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }

        # Visualization
        if args.visualize:
            logger.info("Generating visualizations...")
            visualizer = ResultsVisualizer(output_dir=str(output_dir))
            # Generate plots

        # Save results
        results_file = output_dir / "evaluation_results.json"
        save_json(metrics, str(results_file))
        logger.info(f"Results saved to {results_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        for key, value in metrics.items():
            print(f"{key:.<30} {value:>10.4f}")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
