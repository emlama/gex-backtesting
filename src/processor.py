"""Orchestration layer for the GCI meta-analysis backtest.

This module ties together data loading, metric calculation, PUT price tracking,
statistical analysis, and visualization into a single pipeline.

Processing flow:
    1. BacktestRunner.run() iterates over all available trading dates.
    2. For each date, DayProcessor.process() loads trade data, filters to the
       late-day window (default 2:00-3:45 PM ET), and splits into intervals
       (default 5-min buckets).
    3. For each interval, MetricCalculator computes gamma metrics (GCI, PGR, GDW,
       CAR, charm, vomma, zomma exposures).
    4. PutTracker measures forward PUT returns at multiple time horizons (15, 30,
       45, 60 min) using two strike-selection methods (N-strikes-OTM, max-vomma).
    5. Results are concatenated into a single DataFrame (one row per interval per day).
    6. BacktestRunner then runs univariate screening, composite signal analysis,
       and control experiments (placebo, time-shifted, permutation) to evaluate
       which metrics predict PUT explosions with statistical rigor.

Key design decisions:
    - PUT returns use bid for entry (conservative) and mid for exit.
    - Metrics use absolute gamma (not signed by call/put) for concentration measures
      but signed gamma for net GEX and CAR direction.
    - FDR correction (Benjamini-Hochberg) is applied across all univariate tests
      to control for multiple comparisons.

Dependencies:
    - DataLoader (data_loader.py): Parquet file loading and late-day filtering
    - MetricCalculator (metrics.py): Interval-level gamma metric computation
    - PutTracker (put_tracker.py): Forward PUT return measurement
    - StatisticalAnalyzer (statistics.py): Correlation, lift, permutation tests
    - Visualizer (visualization.py): Chart generation
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import Config, DEFAULT_CONFIG
from .data_loader import DataLoader
from .metrics import MetricCalculator
from .put_tracker import PutTracker
from .statistics import StatisticalAnalyzer
from .visualization import Visualizer


@dataclass
class DayResult:
    """Results from processing a single trading day.

    Attributes:
        date: Trading date string "YYYY-MM-DD".
        n_intervals: Number of intervals that produced valid metrics.
        df: DataFrame with one row per interval, containing all computed
            metrics (GCI, PGR, GDW, CAR, etc.) and forward PUT returns.
    """

    date: str
    n_intervals: int
    df: pd.DataFrame


class DayProcessor:
    """Process a single day of trade data into interval-level metrics.

    For each day: loads trades, filters to late-day window, splits into intervals,
    computes gamma metrics per interval, and tracks forward PUT returns.
    """

    def __init__(self, config: Config, data_loader: Optional[DataLoader] = None):
        """Initialize with config and optional shared DataLoader.

        Args:
            config: Master Config for the backtest.
            data_loader: Optional shared DataLoader instance (avoids re-creating
                per day when called from BacktestRunner).
        """
        self.config = config
        self.data_loader = data_loader or DataLoader(config)
        self.metric_calc = MetricCalculator(config)
        self.put_tracker = PutTracker(config)

    def process(self, trade_date: str | date) -> Optional[DayResult]:
        """Process a single day: calculate metrics and track PUT returns.

        Pipeline per day:
        1. Load late-day trades with interval buckets (load_and_prepare).
        2. Also load full-day trades (needed for PUT return lookups that
           extend past the signal window).
        3. For each interval: compute spot, calculate gamma metrics, measure
           forward PUT returns at configured time horizons.
        4. Merge metrics + returns into a single row per interval.

        Args:
            trade_date: Date string "YYYY-MM-DD" or date object.

        Returns:
            DayResult containing the per-interval DataFrame, or None if no
            valid intervals were produced (e.g., no data, insufficient trades).
        """
        if isinstance(trade_date, date):
            date_str = trade_date.strftime("%Y-%m-%d")
        else:
            date_str = trade_date

        # Load and prepare data (late-day window with intervals)
        df = self.data_loader.load_and_prepare(date_str)
        if len(df) == 0:
            return None

        # Also load full day data for PUT tracking
        df_full = self.data_loader.load_trades_for_date(date_str)

        results = []

        for interval, df_interval in df.groupby("interval"):
            spot = df_interval["spot"].median()
            if pd.isna(spot) or spot <= 0:
                continue

            metrics = self.metric_calc.calculate(df_interval, spot)
            if metrics is None:
                continue

            put_returns_by_method = self.put_tracker.calculate_returns(df_full, interval, spot)

            row = {
                "date": date_str,
                "interval": interval,
                "spot": spot,
                **self.metric_calc.metrics_to_dict(metrics),
            }

            if put_returns_by_method:
                row.update(self.put_tracker.all_returns_to_dict(put_returns_by_method))

            results.append(row)

        if not results:
            return None

        df_results = pd.DataFrame(results)
        return DayResult(date=date_str, n_intervals=len(df_results), df=df_results)


class BacktestRunner:
    """Run the full GCI meta-analysis backtest across all available dates.

    Orchestrates the end-to-end pipeline: day processing, univariate screening,
    composite signal analysis, control experiments, and result persistence.

    Usage:
        runner = BacktestRunner(Config())
        df = runner.run(limit=10)                   # Process 10 days
        uni = runner.run_univariate_analysis()       # Screen all metrics
        comp = runner.run_composite_analysis()       # Test composite signals
        ctrl = runner.run_control_experiments()      # Validate with controls
        files = runner.save_results()                # Persist to disk

    Attributes:
        df_all: Combined DataFrame of all interval-level metrics + PUT returns
            across all processed days. Set after run().
        univariate_results: DataFrame of univariate screening results (one row
            per metric x window x PUT method). Set after run_univariate_analysis().
        composite_results: DataFrame of composite signal results. Set after
            run_composite_analysis().
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize with optional config (defaults to DEFAULT_CONFIG).

        Args:
            config: Master Config. If None, uses DEFAULT_CONFIG from config.py.
        """
        self.config = config or DEFAULT_CONFIG
        self.data_loader = DataLoader(self.config)
        self.day_processor = DayProcessor(self.config, self.data_loader)
        self.stats = StatisticalAnalyzer(self.config)
        self.viz = Visualizer(self.config)

        self.df_all: Optional[pd.DataFrame] = None
        self.univariate_results: Optional[pd.DataFrame] = None
        self.composite_results: Optional[pd.DataFrame] = None

    def run(
        self,
        dates: Optional[list[date]] = None,
        limit: Optional[int] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Run the day-by-day backtest across all available (or specified) dates.

        Processes each date through DayProcessor.process() and concatenates
        results into self.df_all. Errors on individual days are logged but
        do not halt the overall run.

        Args:
            dates: Optional list of dates to process. If None, auto-discovers
                all available dates from parquet files in data_dir.
            limit: If set, process only the first N dates (useful for testing).
            show_progress: If True, display a tqdm progress bar.

        Returns:
            Combined DataFrame (also stored as self.df_all). Empty DataFrame
            if no days produced results.
        """
        if dates is None:
            dates = self.data_loader.get_available_dates()

        if limit is not None:
            dates = dates[:limit]

        all_results = []
        iterator = tqdm(dates, desc="Processing days") if show_progress else dates

        for trade_date in iterator:
            try:
                result = self.day_processor.process(trade_date)
                if result is not None:
                    all_results.append(result.df)
            except Exception as e:
                if show_progress:
                    tqdm.write(f"Error processing {trade_date}: {e}")

        if all_results:
            self.df_all = pd.concat(all_results, ignore_index=True)
        else:
            self.df_all = pd.DataFrame()

        return self.df_all

    def run_univariate_analysis(self, outcome_threshold: float = 100.0) -> pd.DataFrame:
        """Screen each metric individually for predictive power over PUT returns.

        For each (metric, time_horizon, PUT_method) combination:
        1. Compute Spearman rank correlation between metric and PUT % gain.
        2. Compute lift: P(PUT explosion | metric spike) / P(PUT explosion).
        3. Apply FDR correction (Benjamini-Hochberg) across all tests.

        PGR is inverted (negated) before correlation because low PGR is the
        bearish signal (unlike other metrics where high = bearish).

        Args:
            outcome_threshold: PUT % gain threshold to define "explosion" (default 100%).

        Returns:
            DataFrame sorted by lift (descending) with columns: metric, window,
            put_method, spearman_r, spearman_p, lift, ci_low, ci_high, fisher_p,
            n_signals, n_total, p_adjusted, significant.

        Raises:
            ValueError: If run() has not been called yet (no data to analyze).
        """
        if self.df_all is None or len(self.df_all) == 0:
            raise ValueError("Must run backtest first")

        metrics = ["gci", "pgr", "gdw", "car_net", "car_gross", "charm_risk", "vomma_exp", "zomma_exp"]
        windows = self.config.time_horizons

        put_method_prefixes = [""]
        if any(c.startswith("max_vomma_pct_gain_") for c in self.df_all.columns):
            put_method_prefixes.append("max_vomma_")

        results = []

        for metric in metrics:
            if metric not in self.df_all.columns:
                continue

            for window in windows:
                for prefix in put_method_prefixes:
                    outcome_col = f"{prefix}pct_gain_{window}m"
                    if outcome_col not in self.df_all.columns:
                        continue

                    put_method = "n_strikes_otm" if prefix == "" else "max_vomma"

                    df_valid = self.df_all[[metric, outcome_col]].dropna()
                    if len(df_valid) < self.config.stats.min_sample_size:
                        continue

                    corr_result = self.stats.spearman_correlation(
                        -df_valid[metric] if metric == "pgr" else df_valid[metric],
                        df_valid[outcome_col],
                    )

                    threshold = self.config.thresholds.__dict__.get(metric)
                    if threshold is None or self.config.thresholds.use_percentiles:
                        threshold = df_valid[metric].quantile(0.90)

                    spike = df_valid[metric] < threshold if metric == "pgr" else df_valid[metric] > threshold
                    outcome = df_valid[outcome_col] > outcome_threshold
                    lift_result = self.stats.calculate_lift(spike, outcome)

                    results.append({
                        "metric": metric,
                        "window": window,
                        "put_method": put_method,
                        "spearman_r": corr_result.spearman_r,
                        "spearman_p": corr_result.p_value,
                        "lift": lift_result.lift,
                        "ci_low": lift_result.ci_low,
                        "ci_high": lift_result.ci_high,
                        "fisher_p": lift_result.p_value,
                        "n_signals": lift_result.n_signals,
                        "n_total": corr_result.n_samples,
                    })

        df_results = pd.DataFrame(results)

        if len(df_results) > 0:
            reject, p_adjusted = self.stats.apply_fdr_correction(df_results["fisher_p"].values)
            df_results["p_adjusted"] = p_adjusted
            df_results["significant"] = reject
            df_results = df_results.sort_values("lift", ascending=False)

        self.univariate_results = df_results
        return df_results

    def run_composite_analysis(
        self, outcome_col: str = "pct_gain_30m", outcome_threshold: float = 100.0
    ) -> pd.DataFrame:
        """Test composite (multi-metric) signals for predictive power.

        Defines composite signals by combining metric thresholds (e.g.,
        GCI spike AND low PGR) and measures lift for each composite.

        Signals tested:
            - GCI alone (>90th percentile)
            - PGR alone (<10th percentile, i.e., low protective gamma)
            - GCI + Low PGR (both conditions simultaneously)
            - CAR + GCI (high convexity risk with concentrated gamma)

        Args:
            outcome_col: Column name for PUT returns (default "pct_gain_30m").
            outcome_threshold: PUT % gain threshold for "explosion" (default 100%).

        Returns:
            DataFrame with columns: signal, lift, ci_low, ci_high, p_value,
            n_signals, signal_rate.

        Raises:
            ValueError: If run() has not been called yet.
        """
        if self.df_all is None or len(self.df_all) == 0:
            raise ValueError("Must run backtest first")

        df = self.df_all.copy()

        if self.config.thresholds.use_percentiles:
            gci_thresh = df["gci"].quantile(0.90)
            pgr_thresh = df["pgr"].quantile(0.10)
            car_thresh = df["car_net"].quantile(0.90)
        else:
            gci_thresh = self.config.thresholds.gci
            pgr_thresh = self.config.thresholds.pgr
            car_thresh = self.config.thresholds.car_net

        df["gci_spike"] = df["gci"] > gci_thresh
        df["pgr_low"] = df["pgr"] < pgr_thresh
        df["car_spike"] = df["car_net"] > car_thresh

        gci_75th = df["gci"].quantile(0.75)
        df["composite_gci_pgr"] = df["gci_spike"] & df["pgr_low"]
        df["composite_car_gci"] = df["car_spike"] & (df["gci"] > gci_75th)

        outcome = df[outcome_col] > outcome_threshold

        signals = {
            "GCI alone": df["gci_spike"],
            "PGR alone (low)": df["pgr_low"],
            "GCI + Low PGR": df["composite_gci_pgr"],
            "CAR + GCI": df["composite_car_gci"],
        }

        results = []
        for name, signal in signals.items():
            df_valid = pd.DataFrame({"signal": signal, "outcome": outcome}).dropna()
            if len(df_valid) < self.config.stats.min_sample_size:
                continue

            lift_result = self.stats.calculate_lift(df_valid["signal"], df_valid["outcome"])
            results.append({
                "signal": name,
                "lift": lift_result.lift,
                "ci_low": lift_result.ci_low,
                "ci_high": lift_result.ci_high,
                "p_value": lift_result.p_value,
                "n_signals": int(signal.sum()),
                "signal_rate": signal.mean(),
            })

        self.composite_results = pd.DataFrame(results)
        return self.composite_results

    def run_control_experiments(
        self, outcome_col: str = "pct_gain_30m", outcome_threshold: float = 100.0, n_permutations: int = 500
    ) -> dict:
        """Run control experiments to validate that GCI signals are genuine.

        Three control tests:
        1. Placebo: Random signal with same frequency as GCI spike. Expected
           lift ~1.0 (no predictive power). Validates that the analysis framework
           itself does not create spurious results.
        2. Time-shifted: Shift GCI signals forward by 3 intervals. If GCI is
           truly predictive, the shifted version should perform worse (it is
           predicting outcomes that happen BEFORE the signal).
        3. Permutation: Shuffle GCI spike labels 500 times and compute lift
           distribution. The observed lift should exceed 95% of permuted lifts
           for statistical significance.

        Args:
            outcome_col: Column name for PUT returns (default "pct_gain_30m").
            outcome_threshold: PUT % gain threshold for "explosion" (default 100%).
            n_permutations: Number of permutation iterations (default 500).

        Returns:
            Dictionary with keys 'placebo', 'time_shifted', 'permutation', each
            containing lift values, p-values, and pass/fail indicators.

        Raises:
            ValueError: If run() has not been called yet.
        """
        if self.df_all is None or len(self.df_all) == 0:
            raise ValueError("Must run backtest first")

        df = self.df_all.copy()

        if outcome_col not in df.columns:
            gain_cols = [c for c in df.columns if c.startswith("pct_gain_")]
            if gain_cols:
                outcome_col = gain_cols[0]
            else:
                return {"error": "No PUT return data available"}

        df["outcome"] = df[outcome_col] > outcome_threshold

        if self.config.thresholds.use_percentiles:
            gci_thresh = df["gci"].quantile(0.90)
        else:
            gci_thresh = self.config.thresholds.gci
        gci_freq = (df["gci"] > gci_thresh).mean()

        results = {}

        placebo = self.stats.run_control_placebo(df, "outcome", gci_freq)
        results["placebo"] = {
            "lift": placebo.lift,
            "ci_low": placebo.ci_low,
            "ci_high": placebo.ci_high,
            "p_value": placebo.p_value,
            "expected": "~1.0 (no predictive power)",
            "pass": 0.7 < placebo.lift < 1.3,
        }

        df["gci_spike"] = df["gci"] > gci_thresh
        time_shifted = self.stats.run_control_time_shifted(df, "gci_spike", "outcome", shift_intervals=3)
        if time_shifted:
            results["time_shifted"] = {
                "lift": time_shifted.lift,
                "ci_low": time_shifted.ci_low,
                "ci_high": time_shifted.ci_high,
                "p_value": time_shifted.p_value,
                "expected": "Should NOT be better than current GCI",
            }

        perm_result = self.stats.run_permutation_test(
            df["gci_spike"], df["outcome"], n_permutations=n_permutations
        )
        results["permutation"] = {
            "observed_lift": perm_result.observed_lift,
            "p_value": perm_result.p_value,
            "null_mean": perm_result.null_mean,
            "null_std": perm_result.null_std,
            "significant": perm_result.p_value < 0.05,
        }

        return results

    def save_results(self) -> list[Path]:
        """Save all computed results to the configured results directory.

        Writes up to three files:
        - interval_metrics_with_returns.parquet: Full interval-level data (df_all)
        - univariate_screen_results.csv: Univariate analysis results
        - composite_signal_results.csv: Composite signal results

        Returns:
            List of Path objects for files that were written.
        """
        files = []
        results_dir = self.config.ensure_results_dir()

        if self.df_all is not None and len(self.df_all) > 0:
            path = results_dir / "interval_metrics_with_returns.parquet"
            self.df_all.to_parquet(path)
            files.append(path)

        if self.univariate_results is not None and len(self.univariate_results) > 0:
            path = results_dir / "univariate_screen_results.csv"
            self.univariate_results.to_csv(path, index=False)
            files.append(path)

        if self.composite_results is not None and len(self.composite_results) > 0:
            path = results_dir / "composite_signal_results.csv"
            self.composite_results.to_csv(path, index=False)
            files.append(path)

        return files

    def print_summary(self) -> None:
        """Print a human-readable summary of backtest results to stdout.

        Includes data summary (days, intervals, date range), best single metric
        (highest lift), and count of statistically significant results after
        FDR correction.
        """
        print("=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        if self.df_all is None or len(self.df_all) == 0:
            print("\nNO DATA PROCESSED")
            return

        print(f"\nData Summary:")
        print(f"  - Days processed: {self.df_all['date'].nunique()}")
        print(f"  - Total intervals: {len(self.df_all)}")
        print(f"  - Date range: {self.df_all['date'].min()} to {self.df_all['date'].max()}")

        if self.univariate_results is not None and len(self.univariate_results) > 0:
            best = self.univariate_results.iloc[0]
            print(f"\nBest Single Metric:")
            print(f"  - {best['metric']} at {best['window']}m")
            print(f"  - Lift: {best['lift']:.2f} (95% CI: {best['ci_low']:.2f} - {best['ci_high']:.2f})")
            print(f"  - FDR-adjusted p-value: {best['p_adjusted']:.4f}")

            n_sig = self.univariate_results["significant"].sum()
            print(f"\nSignificant Results: {n_sig} / {len(self.univariate_results)}")

        print("\n" + "=" * 60)
