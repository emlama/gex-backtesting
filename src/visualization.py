"""Visualization functions for GCI meta-analysis results.

Creates publication-quality charts using matplotlib/seaborn with a dark
background theme.  All figures are saved to the ``results/`` directory as
PNG files at 150 DPI.

Chart types and their purpose:

- **Correlation heatmap** (``plot_correlation_heatmap``): 2-panel grid showing
  Spearman correlation and lift ratio for each metric x time-window combination.
  One row per PUT selection method. Answers: "Which metrics predict PUT gains?"
- **PUT return distribution** (``plot_put_return_distribution``): 2-panel figure
  showing overall return distribution and conditional distribution (spike vs
  no-spike).  Answers: "Do returns shift when a metric spikes?"
- **Permutation null** (``plot_permutation_null``): Histogram of null
  distribution from permutation test with observed value marked.  Answers:
  "Is the observed lift statistically significant?"
- **Composite comparison** (``plot_composite_comparison``): Horizontal bar
  chart comparing lift ratios across composite signals with confidence
  intervals.  Answers: "Which signal combination has the best edge?"
- **Metric timeseries** (``plot_metric_timeseries``): Multi-panel line chart
  of metric values over a single trading day with threshold lines.
  Answers: "What did the metrics look like on this specific day?"

The ``Visualizer`` class is instantiated by ``processor.BacktestRunner``
and uses ``Config.results_dir`` for file output.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import Config


class Visualizer:
    """Create and save publication-quality matplotlib charts for analysis results.

    Uses dark background theme (``plt.style.use("dark_background")``) and the
    ``husl`` seaborn palette.  All methods return the ``Path`` to the saved PNG.
    """

    def __init__(self, config: Config):
        """Initialize visualizer and create the results output directory.

        Args:
            config: Master ``Config`` -- uses ``results_dir`` for file output
                and ``thresholds`` for threshold reference lines.
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
        """Create a 2-column heatmap grid: Spearman correlation and lift ratio.

        Layout: One row per PUT selection method (n_strikes_otm, max_vomma).
        Left column = Spearman r (metric vs PUT % gain), right column = lift
        ratio (P(gain>100% | spike) / P(gain>100% | no spike)).

        Green cells indicate predictive metrics; red indicates inverse or
        no relationship.  Infinite lift values are capped at 10.0 for display.

        Args:
            df_results: Output from ``BacktestRunner.run_univariate_analysis()``
                with columns: metric, window, spearman_r, lift, put_method.
            filename: Output filename (saved in ``results_dir``).

        Returns:
            Path to saved figure.
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
        """Plot PUT return distributions: overall and conditional on signal spike.

        Layout: 2 panels side by side.
        - Left panel: Histogram of all PUT % gains with breakeven (0%) and
          100% gain reference lines.
        - Right panel: Overlaid histograms comparing gains when the signal
          metric exceeds the threshold ("spike") vs when it does not.

        Returns are clipped to [-100%, +500%] for visualization only.

        Args:
            df: Interval-level DataFrame with metric and gain columns.
            signal_col: Column name of the metric to split on (e.g. "gci").
            threshold: Value above which the metric counts as a "spike".
            gain_col: Column name of the PUT % gain to plot.
            filename: Output filename.

        Returns:
            Path to saved figure.
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
        """Plot the permutation test null distribution with observed lift marked.

        Shows a histogram of lift values obtained by randomly shuffling the
        signal labels (null hypothesis: signal has no predictive power).
        The observed lift is drawn as a red dashed line.  If the observed
        value falls far into the right tail, the signal is significant.

        Args:
            observed_lift: The actual lift from the un-shuffled data.
            null_distribution: Array of lift values from permutation shuffles.
            p_value: Fraction of null lifts >= observed lift.
            metric_name: Label for the metric (used in title).
            filename: Output filename.

        Returns:
            Path to saved figure.
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
        """Plot horizontal bar chart comparing lift ratios of composite signals.

        Bars are color-coded: green (lift > 1.5, strong edge), orange
        (1.0 < lift <= 1.5, marginal), red (lift <= 1.0, no edge).
        Confidence interval error bars are drawn if ``ci_low``/``ci_high``
        columns are present.  Reference lines at lift=1.0 (no edge) and
        lift=1.5 (target edge) are included.

        Args:
            df_composite: Output from ``BacktestRunner.run_composite_analysis()``
                with columns: signal, lift, and optionally ci_low, ci_high.
            filename: Output filename.

        Returns:
            Path to saved figure.
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
        """Plot metric values over time for a single trading day.

        Layout: Vertically stacked subplots, one per metric, sharing the
        x-axis (time in ET).  Each subplot shows the metric line with markers
        and a horizontal red dashed threshold reference line (from
        ``Config.thresholds``).

        Useful for inspecting what happened on a specific day -- e.g. to see
        whether a GCI spike coincided with a PUT price explosion.

        Args:
            df: Single-day interval DataFrame with ``interval`` column and
                one column per metric (output of ``DayProcessor.process()``).
            date: Date string for the figure title (e.g. "2024-06-21").
            metrics: Which metrics to plot (default: ["gci", "pgr", "car_net"]).
            filename: Output filename (default: ``metrics_timeseries_{date}.png``).

        Returns:
            Path to saved figure.
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
