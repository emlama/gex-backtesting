"""Gamma-related metric calculations.

Computes GCI, PGR, GDW, CAR, and related metrics from trade data.
These metrics are designed to identify gamma concentration and
instability that precedes late-day PUT explosions.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config, ET
from .greeks import BlackScholesGreeks


def calculate_tte_from_timestamp(timestamp: pd.Timestamp, expiry_hour: int = 16) -> float:
    """Calculate time-to-expiry in years from a timestamp.

    For 0DTE options, TTE should be calculated as time remaining until
    market close (4 PM ET), not just DTE/365.

    Args:
        timestamp: Current timestamp (timezone-aware, ET expected)
        expiry_hour: Hour of expiry (default: 16 = 4 PM ET)

    Returns:
        Time-to-expiry in years
    """
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(ET)

    # Expiry is at expiry_hour:00 on the same day
    expiry = timestamp.replace(hour=expiry_hour, minute=0, second=0, microsecond=0)

    # If past expiry, return small positive value
    if timestamp >= expiry:
        return 0.0001  # ~52 seconds

    # Calculate time remaining
    time_remaining = (expiry - timestamp).total_seconds()
    minutes_remaining = time_remaining / 60
    hours_remaining = time_remaining / 3600

    # Convert to years (trading hours)
    # Using 6.5 trading hours/day, 252 trading days/year
    trading_hours_per_year = 6.5 * 252
    tte_years = hours_remaining / trading_hours_per_year

    return max(tte_years, 0.0001)  # Minimum 0.0001 years


@dataclass
class IntervalMetrics:
    """All metrics for a single interval."""

    # Concentration metrics
    gci: float  # Gamma Concentration Index (Herfindahl)
    pgr: float  # Protective Gamma Ratio
    gdw: float  # Gamma Distance Weighted

    # Convexity/acceleration metrics
    car_net: float   # Convexity Acceleration Risk (signed)
    car_gross: float  # CAR magnitude

    # Charm and exposure
    charm_risk: float
    vomma_exp: float
    zomma_exp: float

    # Aggregate measures
    gamma_total: float
    net_gex: float

    # Data quality
    n_trades: int
    avg_iv: float
    tte_years: float


class MetricCalculator:
    """Calculate gamma-related metrics from interval trade data."""

    def __init__(self, config: Config):
        """Initialize with configuration.

        Args:
            config: Analysis configuration
        """
        self.config = config
        self.greeks = BlackScholesGreeks(config.risk_free_rate)

        # Side sign mapping: positive = destabilizing, negative = stabilizing
        self._side_sign_map = {
            "at_ask": 1,
            "above_ask": 1,
            "at_bid": -1,
            "below_bid": -1,
            "mid_market": 0,
        }

    def _get_side_signs(self, df: pd.DataFrame) -> pd.Series:
        """Map trade sides to signs."""
        return df["side"].map(self._side_sign_map).fillna(0)

    def calculate(
        self,
        df_interval: pd.DataFrame,
        spot: float,
    ) -> Optional[IntervalMetrics]:
        """Calculate all metrics for a single interval.

        Args:
            df_interval: Trade data for one interval
            spot: Spot price for the interval

        Returns:
            IntervalMetrics or None if insufficient data
        """
        # Filter to strike range
        df = df_interval[
            (df_interval["strike"] >= spot - self.config.strike_range)
            & (df_interval["strike"] <= spot + self.config.strike_range)
        ].copy()

        if len(df) < 10:
            return None

        # Get IV (use median for stability)
        sigma = df["iv"].median() if "iv" in df.columns else 0.20

        # Get TTE - calculate from timestamp if stored value is 0 (common for 0DTE)
        tte = df["tte_years"].median() if "tte_years" in df.columns else 0.0
        if tte <= 0.0001:  # 0DTE stored as 0, need to calculate actual time remaining
            interval_time = df["timestamp"].median() if "timestamp" in df.columns else None
            if interval_time is not None:
                tte = calculate_tte_from_timestamp(
                    interval_time,
                    expiry_hour=self.config.time_window.expiry_hour
                )
            else:
                tte = 0.01  # Fallback

        # Calculate Greeks per trade
        strikes = df["strike"].values
        # Support both column names: opt_type (Analytics Bucket) and option_type (local)
        # Also support both value formats: "C"/"P" and "call"/"put"
        opt_type_col = "opt_type" if "opt_type" in df.columns else "option_type"
        opt_type_vals = df[opt_type_col].str.upper()
        is_call = opt_type_vals.isin(["C", "CALL"]).values

        greeks = self.greeks.calculate_all(spot, strikes, tte, sigma, is_call)

        df["gamma"] = greeks["gamma"]
        df["vomma"] = greeks["vomma"]
        df["zomma"] = greeks["zomma"]
        df["charm"] = greeks["charm"]

        # Side-weighted exposures
        side_sign = self._get_side_signs(df)

        df["gamma_exp"] = df["gamma"].abs() * df["size"] * 100
        df["vomma_exp"] = df["vomma"] * df["size"] * 100 * side_sign
        df["zomma_exp"] = df["zomma"] * df["size"] * 100 * side_sign
        df["charm_exp"] = df["charm"] * df["size"] * side_sign

        # Aggregate gamma by strike
        gamma_by_strike = df.groupby("strike")["gamma_exp"].sum()
        gamma_total = gamma_by_strike.sum()

        if gamma_total == 0:
            return None

        # ----- GCI: Gamma Concentration Index (Herfindahl) -----
        gamma_shares = gamma_by_strike / gamma_total
        gci = (gamma_shares**2).sum()

        # ----- PGR: Protective Gamma Ratio -----
        near_spot_mask = (
            gamma_by_strike.index >= spot - self.config.pgr_near_spot
        ) & (gamma_by_strike.index <= spot + self.config.pgr_near_spot)
        gamma_near = gamma_by_strike[near_spot_mask].sum()
        pgr = gamma_near / gamma_total

        # ----- GDW: Gamma Distance Weighted -----
        distances = np.abs(gamma_by_strike.index - spot)
        weights = np.exp(-distances / self.config.gdw_decay)
        gdw = (gamma_by_strike * weights).sum()

        # ----- CAR: Convexity Acceleration Risk -----
        # Net GEX sign determines direction
        df["gex_exp"] = df["gamma_exp"] * side_sign
        gex_by_strike = df.groupby("strike")["gex_exp"].sum()
        net_gex = gex_by_strike.sum()
        gamma_sign = -1 if net_gex < 0 else 1  # Short gamma = negative sign

        # Time amplifier
        time_amp = min(30, 1 / np.sqrt(max(tte, 0.001)))

        # CAR components
        zomma_total = df["zomma_exp"].sum()
        vomma_total = df["vomma_exp"].sum()

        car_net = (
            gamma_sign * (0.6 * zomma_total + 0.4 * vomma_total) * time_amp / 1e6
        )
        car_gross = (
            (0.6 * abs(zomma_total) + 0.4 * abs(vomma_total)) * time_amp / 1e6
        )

        # ----- Charm Risk -----
        charm_risk = df["charm_exp"].sum() / 1e6

        return IntervalMetrics(
            gci=gci,
            pgr=pgr,
            gdw=gdw,
            car_net=car_net,
            car_gross=car_gross,
            charm_risk=charm_risk,
            vomma_exp=vomma_total / 1e6,
            zomma_exp=zomma_total / 1e6,
            gamma_total=gamma_total,
            net_gex=net_gex,
            n_trades=len(df),
            avg_iv=sigma,
            tte_years=tte,
        )

    def metrics_to_dict(self, metrics: IntervalMetrics) -> dict:
        """Convert IntervalMetrics to dictionary."""
        return {
            "gci": metrics.gci,
            "pgr": metrics.pgr,
            "gdw": metrics.gdw,
            "car_net": metrics.car_net,
            "car_gross": metrics.car_gross,
            "charm_risk": metrics.charm_risk,
            "vomma_exp": metrics.vomma_exp,
            "zomma_exp": metrics.zomma_exp,
            "gamma_total": metrics.gamma_total,
            "net_gex": metrics.net_gex,
            "n_trades": metrics.n_trades,
            "avg_iv": metrics.avg_iv,
            "tte_years": metrics.tte_years,
        }
