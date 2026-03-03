"""Statistical analysis functions for GCI meta-analysis.

Implements proper statistical methodology:
- Spearman correlation (for fat-tailed returns)
- Benjamini-Hochberg FDR correction
- Bootstrap confidence intervals
- Permutation tests for robustness
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr
from statsmodels.stats.multitest import fdrcorrection

from .config import Config


@dataclass
class LiftResult:
    """Result of lift calculation."""

    lift: float
    ci_low: float
    ci_high: float
    p_value: float
    n_signals: int
    n_outcomes: int
    p_outcome_given_signal: float
    p_outcome_given_no_signal: float


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""

    spearman_r: float
    p_value: float
    n_samples: int


@dataclass
class PermutationResult:
    """Result of permutation test."""

    observed_lift: float
    p_value: float
    null_mean: float
    null_std: float
    null_distribution: np.ndarray


class StatisticalAnalyzer:
    """Statistical analysis for signal evaluation."""

    def __init__(self, config: Config):
        """Initialize with configuration.

        Args:
            config: Analysis configuration
        """
        self.config = config
        self.stats_params = config.stats

    def calculate_lift(
        self,
        signal: pd.Series,
        outcome: pd.Series,
    ) -> LiftResult:
        """Calculate lift ratio with 95% confidence interval.

        Lift = P(outcome|signal) / P(outcome|no_signal)

        Args:
            signal: Boolean series indicating signal presence
            outcome: Boolean series indicating outcome occurrence

        Returns:
            LiftResult with lift, CI, and p-value
        """
        # Build contingency table
        signal = signal.astype(bool)
        outcome = outcome.astype(bool)

        # Counts
        tp = ((signal) & (outcome)).sum()  # Signal + Outcome
        fp = ((signal) & (~outcome)).sum()  # Signal + No Outcome
        fn = ((~signal) & (outcome)).sum()  # No Signal + Outcome
        tn = ((~signal) & (~outcome)).sum()  # No Signal + No Outcome

        # Rates
        p_outcome_given_signal = tp / (tp + fp) if (tp + fp) > 0 else 0
        p_outcome_given_no_signal = fn / (fn + tn) if (fn + tn) > 0 else 0

        # Lift
        if p_outcome_given_no_signal > 0:
            lift = p_outcome_given_signal / p_outcome_given_no_signal
        else:
            lift = np.inf if p_outcome_given_signal > 0 else 1.0

        # Fisher's exact test
        contingency = [[tp, fp], [fn, tn]]
        _, p_value = fisher_exact(contingency)

        # Bootstrap confidence interval for lift
        ci_low, ci_high = self._bootstrap_lift_ci(signal, outcome)

        return LiftResult(
            lift=lift,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            n_signals=int(signal.sum()),
            n_outcomes=int(outcome.sum()),
            p_outcome_given_signal=p_outcome_given_signal,
            p_outcome_given_no_signal=p_outcome_given_no_signal,
        )

    def _bootstrap_lift_ci(
        self,
        signal: pd.Series,
        outcome: pd.Series,
    ) -> tuple[float, float]:
        """Calculate bootstrap confidence interval for lift."""
        lifts = []
        n = len(signal)

        for _ in range(self.stats_params.bootstrap_iterations):
            idx = np.random.choice(n, n, replace=True)
            s_boot = signal.iloc[idx]
            o_boot = outcome.iloc[idx]

            tp_b = ((s_boot) & (o_boot)).sum()
            fp_b = ((s_boot) & (~o_boot)).sum()
            fn_b = ((~s_boot) & (o_boot)).sum()
            tn_b = ((~s_boot) & (~o_boot)).sum()

            p_s = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
            p_ns = fn_b / (fn_b + tn_b) if (fn_b + tn_b) > 0 else 0

            if p_ns > 0:
                lifts.append(p_s / p_ns)

        if lifts:
            return float(np.percentile(lifts, [2.5, 97.5])[0]), float(
                np.percentile(lifts, [2.5, 97.5])[1]
            )
        return np.nan, np.nan

    def spearman_correlation(
        self,
        x: pd.Series,
        y: pd.Series,
    ) -> CorrelationResult:
        """Calculate Spearman rank correlation.

        Spearman is preferred for fat-tailed returns (robust to outliers).

        Args:
            x: Independent variable
            y: Dependent variable

        Returns:
            CorrelationResult
        """
        # Remove NaN values
        mask = x.notna() & y.notna()
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < self.stats_params.min_sample_size:
            return CorrelationResult(
                spearman_r=np.nan,
                p_value=np.nan,
                n_samples=len(x_clean),
            )

        corr, p_value = spearmanr(x_clean, y_clean)

        return CorrelationResult(
            spearman_r=corr,
            p_value=p_value,
            n_samples=len(x_clean),
        )

    def run_permutation_test(
        self,
        signal: pd.Series,
        outcome: pd.Series,
        n_permutations: Optional[int] = None,
    ) -> PermutationResult:
        """Run permutation test to get null distribution for lift.

        Args:
            signal: Boolean signal series
            outcome: Boolean outcome series
            n_permutations: Number of permutations (default from config)

        Returns:
            PermutationResult with observed lift and null distribution
        """
        if n_permutations is None:
            n_permutations = self.stats_params.permutation_iterations

        observed_result = self.calculate_lift(signal, outcome)
        observed_lift = observed_result.lift

        null_lifts = []
        for _ in range(n_permutations):
            outcome_shuffled = outcome.sample(frac=1).reset_index(drop=True)
            null_result = self.calculate_lift(signal, outcome_shuffled)
            null_lifts.append(null_result.lift)

        null_array = np.array(null_lifts)
        p_value = (null_array >= observed_lift).mean()

        return PermutationResult(
            observed_lift=observed_lift,
            p_value=float(p_value),
            null_mean=float(np.mean(null_lifts)),
            null_std=float(np.std(null_lifts)),
            null_distribution=null_array,
        )

    def apply_fdr_correction(
        self,
        p_values: list[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply Benjamini-Hochberg FDR correction.

        Args:
            p_values: Array of p-values

        Returns:
            Tuple of (reject mask, adjusted p-values)
        """
        reject, p_adjusted = fdrcorrection(
            p_values, alpha=self.stats_params.fdr_alpha
        )
        return reject, p_adjusted

    def run_control_placebo(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        signal_frequency: float,
    ) -> LiftResult:
        """Run placebo test with random signal.

        A random signal should have lift ~1.0 (no predictive power).

        Args:
            df: DataFrame with outcome column
            outcome_col: Name of outcome column
            signal_frequency: Frequency to match for random signal

        Returns:
            LiftResult for random signal
        """
        random_signal = pd.Series(
            np.random.random(len(df)) < signal_frequency, index=df.index
        )
        outcome = df[outcome_col].astype(bool)

        return self.calculate_lift(random_signal, outcome)

    def run_control_time_shifted(
        self,
        df: pd.DataFrame,
        signal_col: str,
        outcome_col: str,
        date_col: str = "date",
        shift_intervals: int = 3,
    ) -> Optional[LiftResult]:
        """Run time-shifted control (future data shouldn't predict).

        Args:
            df: DataFrame with signal and outcome
            signal_col: Name of signal column
            outcome_col: Name of outcome column
            date_col: Column to group by for shifting
            shift_intervals: Number of intervals to shift forward

        Returns:
            LiftResult for future signal, or None if insufficient data
        """
        df_copy = df.copy()
        df_copy["future_signal"] = df_copy.groupby(date_col)[signal_col].shift(
            -shift_intervals
        )

        df_valid = df_copy[df_copy["future_signal"].notna()]

        if len(df_valid) < self.stats_params.min_sample_size:
            return None

        return self.calculate_lift(
            df_valid["future_signal"].astype(bool),
            df_valid[outcome_col].astype(bool),
        )
