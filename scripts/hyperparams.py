#!/usr/bin/env python3
"""
Hyperparameter Tuning Script

Bayesian hyperparameter optimization using Optuna.
"""

import argparse
import logging
from pathlib import Path

import optuna
import torch
from optuna.samplers import TPESampler

from src.utils import save_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def objective(trial):
    """
    Objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object

    Returns:
        Optimized metric value
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    gamma = trial.suggest_float("gamma", 0.95, 0.99)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    entropy_coeff = trial.suggest_float("entropy_coeff", 0.001, 0.1, log=True)

    logger.info(
        f"Trial {trial.number}: LR={learning_rate:.5f}, BS={batch_size}, "
        f"γ={gamma:.4f}, λ={gae_lambda:.4f}, ent={entropy_coeff:.4f}"
    )

    # TODO: Train agent with these parameters and return metric
    # For now, return dummy value
    metric = -learning_rate * 100  # Dummy objective

    return metric


def main():
    """Run hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Tune hyperparameters")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument(
        "--output", type=str, default="tuning_results", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Starting hyperparameter tuning with {args.trials} trials...")

        # Create study
        sampler = TPESampler(seed=args.seed)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        # Optimize
        study.optimize(objective, n_trials=args.trials)

        # Get best trial
        best_trial = study.best_trial
        logger.info(f"Best trial: #{best_trial.number}")
        logger.info(f"Best value: {best_trial.value:.4f}")
        logger.info(f"Best params: {best_trial.params}")

        # Save results
        results = {
            "best_trial": best_trial.number,
            "best_value": float(best_trial.value),
            "best_params": best_trial.params,
            "n_trials": len(study.trials),
        }

        save_json(results, str(output_dir / "tuning_results.json"))

        # Save study
        import pickle

        with open(output_dir / "study.pkl", "wb") as f:
            pickle.dump(study, f)

        print("\n" + "=" * 60)
        print("TUNING RESULTS")
        print("=" * 60)
        print(f"Best Trial: #{best_trial.number}")
        print(f"Best Value: {best_trial.value:.4f}")
        print("Best Hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key:.<30} {value}")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Error during tuning: {e}")
        raise


if __name__ == "__main__":
    main()
