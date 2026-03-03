"""Configuration for the GEX backtesting framework.

All pre-registered thresholds, time windows, and analysis parameters are
centralized here.  There are two distinct config classes:

- **Config**: Master configuration for the GCI meta-analysis backtest
  pipeline (``processor.BacktestRunner``).  Controls study period, metric
  thresholds, PUT selection, statistical parameters, and output paths.
- **AnalysisConfig**: Lightweight config for the 0DTE analysis notebook
  (GEX chart generation).  Mirrors the old ``gex-analytics`` config for
  compatibility with ``gex_calculator.calculate_gex()``.

Pre-registration principle: Thresholds are defined BEFORE running the
analysis to avoid p-hacking.  By default, percentile-based thresholds
(90th, 95th, 99th) are used instead of fixed values, since the metric
distributions vary across market regimes.

DO NOT modify thresholds after analysis begins.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import pytz

# Timezone
ET = pytz.timezone("America/New_York")

# Project root (one level up from src/)
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class TimeWindow:
    """Late-day analysis time window for signal detection.

    Defines the window of the trading day to scan for gamma-metric signals
    and the 0DTE option expiry time used to cap exit horizons.

    The default window is 2:00 PM - 3:45 PM ET, which captures the
    "gamma ramp" period where dealer hedging pressure intensifies as
    expiration approaches.  Signals detected too close to expiry (4:00 PM)
    will have their longer time horizons (45m, 60m) excluded.

    Attributes:
        start_hour: Window start hour in ET (default 14 = 2:00 PM).
        start_min: Window start minute.
        end_hour: Window end hour in ET (default 15 = 3:00 PM).
        end_min: Window end minute (default 45 -> 3:45 PM).
        expiry_hour: 0DTE expiration hour (default 16 = 4:00 PM).
        expiry_min: 0DTE expiration minute.
        cap_exit_at_expiry: If True, exclude exit horizons past expiry.
    """

    start_hour: int = 14
    start_min: int = 0
    end_hour: int = 15
    end_min: int = 45

    expiry_hour: int = 16
    expiry_min: int = 0
    cap_exit_at_expiry: bool = True

    @property
    def start_minutes(self) -> int:
        """Window start as minutes-from-midnight (e.g. 840 for 2:00 PM)."""
        return self.start_hour * 60 + self.start_min

    @property
    def end_minutes(self) -> int:
        """Window end as minutes-from-midnight."""
        return self.end_hour * 60 + self.end_min

    @property
    def expiry_minutes(self) -> int:
        """Expiry time as minutes-from-midnight (e.g. 960 for 4:00 PM)."""
        return self.expiry_hour * 60 + self.expiry_min

    def get_valid_time_horizons(self, signal_minutes: int) -> list[int]:
        """Return exit horizons (in minutes) that fit before expiry.

        For a signal at 3:30 PM (930 min), with expiry at 4:00 PM (960 min),
        only the 15-minute and 29-minute horizons fit.  This prevents tracking
        PUT returns past option expiration.

        Args:
            signal_minutes: Signal time as minutes-from-midnight.

        Returns:
            List of valid horizon values from [15, 30, 45, 60].
        """
        if not self.cap_exit_at_expiry:
            return [15, 30, 45, 60]
        max_horizon = self.expiry_minutes - signal_minutes - 1
        return [h for h in [15, 30, 45, 60] if h <= max_horizon]


@dataclass
class Thresholds:
    """Pre-registered metric thresholds for signal classification.

    Controls how "spikes" are identified for each gamma metric.  When
    ``use_percentiles`` is True (default), thresholds are computed
    dynamically from the data distribution (e.g. 90th percentile).
    When False, the fixed values below are used.

    Fixed threshold reference values (used when ``use_percentiles=False``):

    Attributes:
        use_percentiles: Use data-driven percentile thresholds (recommended).
        percentiles: Percentile levels to test (default [90, 95, 99]).
        gci: Gamma Concentration Index -- above this = concentrated gamma.
        pgr: Protective Gamma Ratio -- BELOW this = lack of hedging.
        gdw: Gamma Distance Weighted threshold (None = auto from percentile).
        car_net: Convexity Acceleration Risk (signed) threshold.
        car_gross: CAR magnitude threshold.
        car_accel: CAR acceleration threshold.
        charm_risk: Charm risk threshold (delta decay rate).
        zomma: Zomma exposure threshold (None = auto from percentile).
        vomma: Vomma exposure threshold (None = auto from percentile).
    """

    use_percentiles: bool = True
    percentiles: list[int] = field(default_factory=lambda: [90, 95, 99])

    gci: float = 0.30
    pgr: float = 0.25
    gdw: Optional[float] = None
    car_net: float = 2.5
    car_gross: float = 5.0
    car_accel: float = 2.0
    charm_risk: float = 1.0
    zomma: Optional[float] = None
    vomma: Optional[float] = None


class PutSelectionMethod(Enum):
    """PUT strike selection strategy used by ``put_tracker.PutTracker``.

    Attributes:
        N_STRIKES_OTM: Pick the Nth strike below ATM (simple, deterministic).
        MAX_VOMMA: Pick the OTM strike with highest volume-weighted vomma.
    """

    N_STRIKES_OTM = "n_strikes_otm"
    MAX_VOMMA = "max_vomma"


@dataclass
class PutSelection:
    """PUT option strike selection parameters.

    Attributes:
        methods: Which selection methods to run (both by default).
        n_strikes_otm: For N_STRIKES_OTM method -- how many strikes below ATM
            to select (default 2, i.e. the 2nd OTM put strike).
        vomma_otm_range: For MAX_VOMMA method -- (min, max) SPX points below
            spot to search for the highest-vomma strike (default 5-50 points OTM).
    """

    methods: list[PutSelectionMethod] = field(
        default_factory=lambda: [PutSelectionMethod.N_STRIKES_OTM, PutSelectionMethod.MAX_VOMMA]
    )
    n_strikes_otm: int = 2
    vomma_otm_range: tuple[int, int] = (5, 50)


@dataclass
class StatisticalParams:
    """Statistical analysis parameters used by ``statistics.StatisticalAnalyzer``.

    Attributes:
        fdr_alpha: False Discovery Rate significance level for Benjamini-Hochberg
            correction (default 0.10 = 10% FDR).
        bootstrap_iterations: Number of bootstrap resamples for confidence intervals.
        permutation_iterations: Number of permutation shuffles for null distribution.
        min_sample_size: Minimum observations required to run a statistical test.
    """

    fdr_alpha: float = 0.10
    bootstrap_iterations: int = 1000
    permutation_iterations: int = 1000
    min_sample_size: int = 30


@dataclass
class AnalysisConfig:
    """Configuration for the 0DTE analysis notebook (GEX charts).

    This is a separate, simpler config used by ``gex_calculator.calculate_gex()``
    and ``gex_calculator.calculate_greeks()`` for single-day GEX chart generation
    in Jupyter notebooks.  It mirrors the old gex-analytics AnalysisConfig for
    backwards compatibility.

    **Not used by the backtest pipeline** -- ``BacktestRunner`` uses ``Config``.

    Attributes:
        trade_date: Date to analyze (YYYY-MM-DD).
        strike_range: Points above/below spot to include in GEX calculations.
        risk_free_rate: Annual risk-free rate for Black-Scholes.
        market_close_time: Market close time in HH:MM (for TTE calculation).
        buy_sides: Trade side values classified as buyer-initiated.
        sell_sides: Trade side values classified as seller-initiated.
        exclude_complex: Whether to filter out multi-leg/complex trades.
        complex_codes: Polygon condition codes indicating complex/multi-leg trades.
    """

    trade_date: str = "2024-01-02"
    strike_range: int = 200
    risk_free_rate: float = 0.05
    market_close_time: str = "16:00"

    buy_sides: tuple = ("at_ask", "above_ask", "mid_market")
    sell_sides: tuple = ("at_bid", "below_bid")

    exclude_complex: bool = True
    complex_codes: tuple = (12, 13, 14, 15, 33, 37, 38)


@dataclass
class Config:
    """Master configuration for the GCI meta-analysis backtest pipeline.

    Used by ``processor.BacktestRunner`` and all its sub-components
    (``MetricCalculator``, ``PutTracker``, ``StatisticalAnalyzer``,
    ``Visualizer``).  Composes the sub-configs above into a single object.

    Attributes:
        study_start: First date to include in the backtest (YYYY-MM-DD).
        study_end: Last date to include.
        interval_minutes: Width of each analysis interval in minutes (default 5).
        strike_range: SPX points above/below spot to include for gamma
            calculations in ``MetricCalculator`` (default 100).
        pgr_near_spot: Points from spot to define "near ATM" for the
            Protective Gamma Ratio calculation (default 20).
        gdw_decay: Exponential decay constant for Gamma Distance Weighted
            metric -- strikes this many points from spot are weighted at
            1/e of ATM (default 20).
        risk_free_rate: Annual risk-free rate for Black-Scholes.
        time_horizons: Exit horizons in minutes for PUT return tracking.
        time_window: Late-day time window for signal detection.
        thresholds: Spike threshold definitions for each metric.
        put_selection: PUT strike selection parameters.
        stats: Statistical analysis parameters.
        data_dir: Path to input trade data (parquet files).
        results_dir: Path for output files (created on first use).
    """

    study_start: str = "2024-01-02"
    study_end: str = "2026-02-19"

    interval_minutes: int = 5

    strike_range: int = 100
    pgr_near_spot: int = 20
    gdw_decay: int = 20

    risk_free_rate: float = 0.05

    time_horizons: list[int] = field(default_factory=lambda: [15, 30, 45, 60])

    time_window: TimeWindow = field(default_factory=TimeWindow)
    thresholds: Thresholds = field(default_factory=Thresholds)
    put_selection: PutSelection = field(default_factory=PutSelection)
    stats: StatisticalParams = field(default_factory=StatisticalParams)

    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")

    def get_data_dir(self) -> Path:
        """Return the configured data directory path."""
        return self.data_dir

    def ensure_results_dir(self) -> Path:
        """Create and return the results directory (creates parents if needed)."""
        self.results_dir.mkdir(exist_ok=True, parents=True)
        return self.results_dir


# Singleton default config used by BacktestRunner when no config is provided.
DEFAULT_CONFIG = Config()
