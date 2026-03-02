"""
Results Visualizer Module

Visualization and reporting of trading results.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """
    Generate visualizations and reports from trading results.

    Features:
    - Equity curve plotting
    - Drawdown visualization
    - Trade analysis
    - HTML report generation
    """

    def __init__(self, output_dir: str = "results"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_equity(
        self,
        equity_curve: np.ndarray,
        title: str = "Equity Curve",
        save_path: Optional[str] = None,
    ) -> str:
        """
        Plot equity curve.

        Args:
            equity_curve: Equity values over time
            title: Plot title
            save_path: Save path (optional)

        Returns:
            Path to saved file or plot data
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(equity_curve, linewidth=2)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Portfolio Value ($)")
            ax.grid(True, alpha=0.3)

            if save_path is None:
                save_path = self.output_dir / "equity_curve.png"
            else:
                save_path = Path(save_path)

            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close()

            logger.info(f"Equity curve saved to {save_path}")
            return str(save_path)

        except ImportError:
            logger.warning("matplotlib not available, skipping visualization")
            return ""

    def plot_drawdown(
        self,
        equity_curve: np.ndarray,
        title: str = "Drawdown",
        save_path: Optional[str] = None,
    ) -> str:
        """
        Plot drawdown.

        Args:
            equity_curve: Equity values over time
            title: Plot title
            save_path: Save path (optional)

        Returns:
            Path to saved file
        """
        try:
            import matplotlib.pyplot as plt

            # Compute drawdown
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max * 100

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5, color="red")
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Drawdown (%)")
            ax.grid(True, alpha=0.3)

            if save_path is None:
                save_path = self.output_dir / "drawdown.png"
            else:
                save_path = Path(save_path)

            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close()

            logger.info(f"Drawdown plot saved to {save_path}")
            return str(save_path)

        except ImportError:
            logger.warning("matplotlib not available, skipping visualization")
            return ""

    def plot_trades(
        self,
        equity_curve: np.ndarray,
        trades: List,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Plot equity curve with trade markers.

        Args:
            equity_curve: Equity values
            trades: List of Trade objects
            save_path: Save path (optional)

        Returns:
            Path to saved file
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(equity_curve, linewidth=2, label="Equity Curve")

            # Mark trades
            for trade in trades:
                entry_idx = trade.entry_time
                exit_idx = trade.exit_time

                # Entry point (buy)
                ax.scatter(
                    entry_idx,
                    equity_curve[entry_idx],
                    color="green",
                    marker="^",
                    s=100,
                    label="Buy" if trade == trades[0] else "",
                )
                # Exit point (sell)
                ax.scatter(
                    exit_idx,
                    equity_curve[exit_idx],
                    color="red",
                    marker="v",
                    s=100,
                    label="Sell" if trade == trades[0] else "",
                )

            ax.set_title(
                "Equity Curve with Trade Markers", fontsize=14, fontweight="bold"
            )
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Portfolio Value ($)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if save_path is None:
                save_path = self.output_dir / "trades.png"
            else:
                save_path = Path(save_path)

            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close()

            logger.info(f"Trade plot saved to {save_path}")
            return str(save_path)

        except ImportError:
            logger.warning("matplotlib not available, skipping visualization")
            return ""

    def generate_html_report(
        self,
        metrics: Dict[str, float],
        equity_curve: np.ndarray,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Generate HTML report.

        Args:
            metrics: Performance metrics
            equity_curve: Equity curve
            save_path: Save path (optional)

        Returns:
            Path to saved file
        """
        if save_path is None:
            save_path = self.output_dir / "report.html"
        else:
            save_path = Path(save_path)

        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Trading Strategy Report</h1>
            <h2>Performance Metrics</h2>
            <table>
        """

        for metric_name, value in metrics.items():
            html_content += f"<tr><td>{metric_name}</td><td>{value:.4f}</td></tr>"

        html_content += """
            </table>
            <h2>Equity Statistics</h2>
            <table>
        """

        html_content += (
            f"<tr><td>Initial Equity</td><td>${equity_curve[0]:,.2f}</td></tr>"
        )
        html_content += (
            f"<tr><td>Final Equity</td><td>${equity_curve[-1]:,.2f}</td></tr>"
        )
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
        html_content += f"<tr><td>Total Return</td><td>{total_return:.2f}%</td></tr>"

        html_content += """
            </table>
        </body>
        </html>
        """

        with open(save_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML report saved to {save_path}")
        return str(save_path)
