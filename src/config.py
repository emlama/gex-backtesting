"""Configuration for GEX backtesting.

All pre-registered thresholds and parameters are defined here.
DO NOT modify thresholds after analysis begins to avoid p-hacking.
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
    """Late-day analysis time window."""

    start_hour: int = 14  # 2:00 PM ET
    start_min: int = 0
    end_hour: int = 15  # 3:45 PM ET
    end_min: int = 45

    expiry_hour: int = 16  # 4:00 PM ET
    expiry_min: int = 0
    cap_exit_at_expiry: bool = True

    @property
    def start_minutes(self) -> int:
        return self.start_hour * 60 + self.start_min

    @property
    def end_minutes(self) -> int:
        return self.end_hour * 60 + self.end_min

    @property
    def expiry_minutes(self) -> int:
        return self.expiry_hour * 60 + self.expiry_min

    def get_valid_time_horizons(self, signal_minutes: int) -> list[int]:
        if not self.cap_exit_at_expiry:
            return [15, 30, 45, 60]
        max_horizon = self.expiry_minutes - signal_minutes - 1
        return [h for h in [15, 30, 45, 60] if h <= max_horizon]


@dataclass
class Thresholds:
    """Pre-registered metric thresholds."""

    use_percentiles: bool = True
    percentiles: list[int] = field(default_factory=lambda: [90, 95, 99])

    # Fixed thresholds (used if use_percentiles=False)
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
    """PUT selection strategy."""

    N_STRIKES_OTM = "n_strikes_otm"
    MAX_VOMMA = "max_vomma"


@dataclass
class PutSelection:
    """PUT option selection parameters."""

    methods: list[PutSelectionMethod] = field(
        default_factory=lambda: [PutSelectionMethod.N_STRIKES_OTM, PutSelectionMethod.MAX_VOMMA]
    )
    n_strikes_otm: int = 2
    vomma_otm_range: tuple[int, int] = (5, 50)


@dataclass
class StatisticalParams:
    """Statistical analysis parameters."""

    fdr_alpha: float = 0.10
    bootstrap_iterations: int = 1000
    permutation_iterations: int = 1000
    min_sample_size: int = 30


@dataclass
class AnalysisConfig:
    """Configuration for the 0DTE analysis notebook (GEX charts).

    This mirrors the old gex-analytics AnalysisConfig for backwards
    compatibility with calculate_gex() and calculate_greeks().
    """

    trade_date: str = "2024-01-02"
    strike_range: int = 200
    risk_free_rate: float = 0.05
    market_close_time: str = "16:00"

    # Trade side classification
    buy_sides: tuple = ("at_ask", "above_ask", "mid_market")
    sell_sides: tuple = ("at_bid", "below_bid")

    # Complex trade filtering
    exclude_complex: bool = True
    complex_codes: tuple = (12, 13, 14, 15, 33, 37, 38)


@dataclass
class Config:
    """Master configuration for GCI meta-analysis backtests."""

    # Study period
    study_start: str = "2024-01-02"
    study_end: str = "2026-02-19"

    # Interval settings
    interval_minutes: int = 5

    # Strike range for gamma calculations
    strike_range: int = 100
    pgr_near_spot: int = 20
    gdw_decay: int = 20

    # Risk-free rate for Greeks
    risk_free_rate: float = 0.05

    # Time horizons for PUT tracking (minutes after signal)
    time_horizons: list[int] = field(default_factory=lambda: [15, 30, 45, 60])

    # Component configs
    time_window: TimeWindow = field(default_factory=TimeWindow)
    thresholds: Thresholds = field(default_factory=Thresholds)
    put_selection: PutSelection = field(default_factory=PutSelection)
    stats: StatisticalParams = field(default_factory=StatisticalParams)

    # Data paths
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")

    def get_data_dir(self) -> Path:
        return self.data_dir

    def ensure_results_dir(self) -> Path:
        self.results_dir.mkdir(exist_ok=True, parents=True)
        return self.results_dir


DEFAULT_CONFIG = Config()
