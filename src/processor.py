"""Main processing logic for GCI meta-analysis.

Orchestrates data loading, metric calculation, PUT tracking, and analysis.
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
    """Results from processing a single day."""

    date: str
    n_intervals: int
    df: pd.DataFrame


class DayProcessor:
    """Process a single day of trade data."""

    def __init__(self, config: Config, data_loader: Optional[DataLoader] = None):
        self.config = config
        self.data_loader = data_loader or DataLoader(config)
        self.metric_calc = MetricCalculator(config)
        self.put_tracker = PutTracker(config)

    def process(self, trade_date: str | date) -> Optional[DayResult]:
        """Process a single day: calculate metrics and track PUT returns."""
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
    """Run full backtest across all available dates."""

    def __init__(self, config: Optional[Config] = None):
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
        """Run backtest across all dates."""
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
        """Run univariate analysis on all metrics."""
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
        """Run composite signal analysis."""
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
        """Run all control experiments."""
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
        """Save all results to files."""
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
        """Print analysis summary to console."""
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
