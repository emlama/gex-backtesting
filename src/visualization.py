"""Visualization functions for GCI meta-analysis results.

Creates publication-quality charts for:
- Correlation heatmaps
- PUT return distributions
- Signal comparison charts
- Time series of metrics
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import Config


class Visualizer:
    """Create visualizations for analysis results."""

    def __init__(self, config: Config):
        """Initialize with configuration.

        Args:
            config: Analysis configuration
        """
        self.config = config
        self.results_dir = config.ensure_results_dir()

        # Set default style
        plt.style.use("dark_background")
        sns.set_palette("husl")

    def plot_correlation_heatmap(
        self,
        df_results: pd.DataFrame,
        filename: str = "correlation_heatmap.png",
    ) -> Path:
        """Create correlation heatmap comparing metrics and time windows.

        Args:
            df_results: DataFrame with 'metric', 'window', 'spearman_r', 'lift' columns
            filename: Output filename

        Returns:
            Path to saved figure
        """
        # Handle multiple PUT selection methods - create separate heatmaps
        put_methods = df_results["put_method"].unique() if "put_method" in df_results.columns else ["default"]
        n_methods = len(put_methods)

        fig, axes = plt.subplots(n_methods, 2, figsize=(16, 6 * n_methods))

        if n_methods == 1:
            axes = [axes]

        for i, method in enumerate(put_methods):
            if "put_method" in df_results.columns:
                df_method = df_results[df_results["put_method"] == method]
            else:
                df_method = df_results

            # ----- Plot 1: Correlation Heatmap -----
            pivot_corr = df_method.pivot(
                index="metric", columns="window", values="spearman_r"
            )

            sns.heatmap(
                pivot_corr,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                center=0,
                ax=axes[i][0],
                vmin=-0.5,
                vmax=0.5,
            )
            axes[i][0].set_title(
                f"Spearman Correlation: Metric vs PUT % Gain\n({method})",
                fontsize=12,
            )
            axes[i][0].set_xlabel("Minutes after signal")
            axes[i][0].set_ylabel("Metric")

            # ----- Plot 2: Lift Comparison -----
            pivot_lift = df_method.pivot(index="metric", columns="window", values="lift")

            # Replace inf with large value for visualization
            pivot_lift = pivot_lift.replace([np.inf, -np.inf], 10.0)

            sns.heatmap(
                pivot_lift,
                annot=True,
                fmt=".1f",
                cmap="RdYlGn",
                center=1,
                ax=axes[i][1],
                vmin=0.5,
                vmax=10.0,
            )
            axes[i][1].set_title(
                f"Lift Ratio: P(gain>100%|spike) / P(gain>100%|no spike)\n({method})",
                fontsize=12,
            )
            axes[i][1].set_xlabel("Minutes after signal")
            axes[i][1].set_ylabel("Metric")

        plt.tight_layout()

        filepath = self.results_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        return filepath

    def plot_put_return_distribution(
        self,
        df: pd.DataFrame,
        signal_col: str,
        threshold: float,
        gain_col: str = "pct_gain_30m",
        filename: str = "put_return_distribution.png",
    ) -> Path:
        """Plot PUT return distributions conditional on signal.

        Args:
            df: DataFrame with gain and signal columns
            signal_col: Name of signal column
            threshold: Threshold for spike definition
            gain_col: Name of gain column
            filename: Output filename

        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ----- Plot 1: Overall Distribution -----
        gains = df[gain_col].dropna()
        gains_clipped = gains.clip(-100, 500)  # Clip for visualization

        axes[0].hist(
            gains_clipped,
            bins=50,
            alpha=0.7,
            color="steelblue",
            edgecolor="white",
        )
        axes[0].axvline(x=0, color="white", linestyle="--", linewidth=2)
        axes[0].axvline(
            x=100, color="lime", linestyle="--", linewidth=2, label="100% gain"
        )
        axes[0].set_xlabel("PUT % Gain (30 min)")
        axes[0].set_ylabel("Count")
        axes[0].set_title(
            f"Distribution of PUT Returns\n"
            f"(n={len(gains)}, mean={gains.mean():.1f}%, median={gains.median():.1f}%)"
        )
        axes[0].legend()

        # ----- Plot 2: Conditional on Signal Spike -----
        spike = df[signal_col] > threshold
        gains_spike = df.loc[spike, gain_col].dropna()
        gains_no_spike = df.loc[~spike, gain_col].dropna()

        axes[1].hist(
            gains_no_spike.clip(-100, 500),
            bins=30,
            alpha=0.5,
            color="gray",
            label=f"No spike (n={len(gains_no_spike)})",
            edgecolor="white",
        )
        axes[1].hist(
            gains_spike.clip(-100, 500),
            bins=30,
            alpha=0.7,
            color="red",
            label=f"Spike (n={len(gains_spike)})",
            edgecolor="white",
        )
        axes[1].axvline(x=100, color="lime", linestyle="--", linewidth=2)
        axes[1].set_xlabel("PUT % Gain (30 min)")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"PUT Returns: {signal_col.upper()} Spike vs No Spike")
        axes[1].legend()

        plt.tight_layout()

        filepath = self.results_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        return filepath

    def plot_permutation_null(
        self,
        observed_lift: float,
        null_distribution: np.ndarray,
        p_value: float,
        metric_name: str = "GCI",
        filename: str = "permutation_null.png",
    ) -> Path:
        """Plot permutation test null distribution.

        Args:
            observed_lift: Observed lift value
            null_distribution: Array of null lift values
            p_value: Permutation p-value
            metric_name: Name of metric being tested
            filename: Output filename

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(null_distribution, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
        ax.axvline(
            x=observed_lift,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Observed: {observed_lift:.2f}",
        )
        ax.axvline(
            x=np.mean(null_distribution),
            color="white",
            linestyle=":",
            linewidth=2,
            label=f"Null mean: {np.mean(null_distribution):.2f}",
        )

        ax.set_xlabel("Lift")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Permutation Test: {metric_name} Spike Signal\n"
            f"p-value = {p_value:.4f}"
        )
        ax.legend()

        plt.tight_layout()

        filepath = self.results_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        return filepath

    def plot_composite_comparison(
        self,
        df_composite: pd.DataFrame,
        filename: str = "composite_comparison.png",
    ) -> Path:
        """Plot comparison of composite signals.

        Args:
            df_composite: DataFrame with signal comparison results
            filename: Output filename

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Sort by lift
        df_sorted = df_composite.sort_values("lift", ascending=True)

        y_pos = np.arange(len(df_sorted))
        colors = ["green" if l > 1.5 else "orange" if l > 1.0 else "red" for l in df_sorted["lift"]]

        ax.barh(y_pos, df_sorted["lift"], color=colors, alpha=0.7, edgecolor="white")
        ax.axvline(x=1.0, color="white", linestyle="--", linewidth=2, label="No edge")
        ax.axvline(x=1.5, color="lime", linestyle=":", linewidth=2, label="Target lift")

        # Add error bars if CI available
        if "ci_low" in df_sorted.columns and "ci_high" in df_sorted.columns:
            xerr_low = df_sorted["lift"] - df_sorted["ci_low"]
            xerr_high = df_sorted["ci_high"] - df_sorted["lift"]
            ax.errorbar(
                df_sorted["lift"],
                y_pos,
                xerr=[xerr_low, xerr_high],
                fmt="none",
                color="white",
                capsize=3,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted["signal"])
        ax.set_xlabel("Lift Ratio")
        ax.set_title("Signal Comparison: Lift Ratios for PUT > 100% Gain")
        ax.legend(loc="lower right")

        plt.tight_layout()

        filepath = self.results_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        return filepath

    def plot_metric_timeseries(
        self,
        df: pd.DataFrame,
        date: str,
        metrics: Optional[list[str]] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """Plot metric values over time for a single day.

        Args:
            df: DataFrame with interval and metric columns
            date: Date string for title
            metrics: List of metrics to plot (default: gci, pgr, car_net)
            filename: Output filename

        Returns:
            Path to saved figure
        """
        if metrics is None:
            metrics = ["gci", "pgr", "car_net"]

        if filename is None:
            filename = f"metrics_timeseries_{date}.png"

        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 4 * n_metrics), sharex=True)

        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            if metric in df.columns:
                ax.plot(df["interval"], df[metric], marker="o", markersize=3)
                ax.axhline(
                    y=self.config.thresholds.__dict__.get(metric, 0),
                    color="red",
                    linestyle="--",
                    alpha=0.5,
                    label="Threshold",
                )
                ax.set_ylabel(metric.upper())
                ax.legend(loc="upper right")
                ax.grid(alpha=0.3)

        axes[-1].set_xlabel("Time (ET)")
        fig.suptitle(f"Metric Evolution: {date}", fontsize=14)

        plt.tight_layout()

        filepath = self.results_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        return filepath
