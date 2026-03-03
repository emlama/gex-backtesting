"""Gamma-related metric calculations for 0DTE PUT explosion detection.

Computes interval-level metrics from SPX 0DTE option trade data to identify
gamma concentration and instability that precedes late-day PUT price explosions.

Metrics computed:
    GCI (Gamma Concentration Index): Herfindahl-style measure of how concentrated
        gamma exposure is across strikes. High GCI = gamma piled into few strikes,
        creating fragile positioning vulnerable to rapid unwinds.

    PGR (Protective Gamma Ratio): Fraction of total gamma near ATM (within
        config.pgr_near_spot points of spot). Low PGR = gamma has migrated away
        from spot, reducing the stabilizing "gamma cushion" around the current price.

    GDW (Gamma Distance Weighted): Exponentially-weighted gamma sum that gives
        more weight to gamma near the spot price. Uses exp(-distance / decay) weighting.
        High GDW = large gamma exposure close to spot (potential for pin risk or
        explosive moves if gamma flips).

    CAR (Convexity Acceleration Risk): Composite risk measure combining zomma
        (d(gamma)/d(vol)) and vomma (d(vega)/d(vol)), amplified by a time decay
        factor. Captures the "feedback loop" risk where a vol spike increases gamma,
        which forces more hedging, which moves spot further, which increases vol.
        CAR_net is signed (negative = short gamma regime), CAR_gross is magnitude only.

    Charm Risk: Aggregate delta decay rate across all positions. High charm risk
        means delta hedging requirements change rapidly with time, creating forced
        flows as expiry approaches.

These metrics are computed per time interval (default 5-min) during the late-day
window (default 2:00-3:45 PM ET) and paired with forward PUT returns to evaluate
predictive power.

Dependencies:
    - BlackScholesGreeks (greeks.py): Analytical gamma, vomma, zomma, charm
    - Config (config.py): Strike range, decay parameters, time windows
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
    """All computed metrics for a single time interval.

    Each field represents a different lens on the gamma landscape during
    one interval (e.g., 5-min bucket). These are later joined with forward
    PUT returns to evaluate predictive power.

    Attributes:
        gci: Gamma Concentration Index. Herfindahl-Hirschman Index applied to
            gamma shares by strike. Range [0, 1]. Value of 1/N if perfectly
            spread across N strikes; approaches 1.0 if all gamma is at one strike.
        pgr: Protective Gamma Ratio. Fraction of total gamma within pgr_near_spot
            points of the current spot. Range [0, 1]. High PGR = stabilizing cushion
            around ATM; low PGR = gamma has shifted away, reducing pin effect.
        gdw: Gamma Distance Weighted. Exponentially-weighted gamma sum favoring
            strikes near spot. Uses exp(-|strike - spot| / gdw_decay) weights.
            Units: same as raw gamma exposure (contracts * 100 * gamma).
        car_net: Convexity Acceleration Risk (signed). Composite of zomma (60%)
            and vomma (40%) exposures, amplified by a time decay factor and signed
            by the net GEX direction. Positive = long gamma amplification;
            negative = short gamma amplification (more dangerous). Units: millions.
        car_gross: CAR magnitude (unsigned). Same formula as car_net but uses
            absolute values of zomma/vomma. Captures total convexity risk
            regardless of direction. Units: millions.
        charm_risk: Aggregate charm (delta decay) exposure across all trades.
            High values mean delta hedging requirements are changing rapidly,
            creating forced directional flows. Units: millions.
        vomma_exp: Total vomma exposure (side-weighted). Measures aggregate
            sensitivity of vega to vol changes. Units: millions.
        zomma_exp: Total zomma exposure (side-weighted). Measures aggregate
            sensitivity of gamma to vol changes. Units: millions.
        gamma_total: Sum of absolute gamma exposure across all strikes.
            Units: contracts * 100 * gamma.
        net_gex: Side-weighted net gamma exposure. Positive = dealer long gamma
            (stabilizing); negative = dealer short gamma (destabilizing).
        n_trades: Number of trades in the interval (after strike-range filtering).
        avg_iv: Median implied volatility across trades in the interval.
        tte_years: Time-to-expiry in years at the interval midpoint.
    """

    # Concentration metrics
    gci: float
    pgr: float
    gdw: float

    # Convexity/acceleration metrics
    car_net: float
    car_gross: float

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
    """Calculate gamma-related metrics from interval trade data.

    Computes GCI, PGR, GDW, CAR, charm risk, and aggregate exposures for a
    single time interval of option trades. Designed to be called once per
    interval by DayProcessor.

    The calculator uses Black-Scholes Greeks (gamma, vomma, zomma, charm)
    computed analytically, and weights exposures by trade side to distinguish
    buying pressure (destabilizing for dealers) from selling pressure (stabilizing).
    """

    def __init__(self, config: Config):
        """Initialize with configuration.

        Args:
            config: Analysis configuration containing strike_range, pgr_near_spot,
                gdw_decay, risk_free_rate, and time_window parameters.
        """
        self.config = config
        self.greeks = BlackScholesGreeks(config.risk_free_rate)

        # Side sign mapping for trade-direction weighting:
        #   +1 = customer buying (at/above ask) -> dealer is short -> destabilizing
        #   -1 = customer selling (at/below bid) -> dealer is long -> stabilizing
        #    0 = mid-market trades -> ambiguous direction, treated as neutral
        self._side_sign_map = {
            "at_ask": 1,
            "above_ask": 1,
            "at_bid": -1,
            "below_bid": -1,
            "mid_market": 0,
        }

    def _get_side_signs(self, df: pd.DataFrame) -> pd.Series:
        """Map trade side labels to numeric signs (+1, -1, or 0).

        Args:
            df: DataFrame with a 'side' column containing Polygon trade-side
                classifications (at_ask, above_ask, at_bid, below_bid, mid_market).

        Returns:
            Series of side signs aligned with df index. Unknown sides default to 0.
        """
        return df["side"].map(self._side_sign_map).fillna(0)

    def calculate(
        self,
        df_interval: pd.DataFrame,
        spot: float,
    ) -> Optional[IntervalMetrics]:
        """Calculate all metrics for a single time interval.

        Pipeline: filter to strike range -> compute Greeks -> weight by trade side
        -> aggregate by strike -> derive concentration/risk metrics.

        Args:
            df_interval: Trade data for one interval. Expected columns: strike,
                size, side, iv (optional), tte_years (optional), timestamp,
                opt_type or option_type.
            spot: SPX spot price for the interval (usually median of trades).

        Returns:
            IntervalMetrics dataclass with all computed metrics, or None if the
            interval has fewer than 10 trades in the strike range or zero total gamma.
        """
        # Filter to strikes within config.strike_range points of spot (e.g., +/- 100 pts).
        # This excludes deep OTM/ITM options that add noise without meaningful gamma.
        df = df_interval[
            (df_interval["strike"] >= spot - self.config.strike_range)
            & (df_interval["strike"] <= spot + self.config.strike_range)
        ].copy()

        # Minimum trade threshold: 10 trades required for meaningful metrics.
        if len(df) < 10:
            return None

        # Use median IV across trades for stability (avoids outlier skew from
        # wide bid-ask spreads on illiquid strikes). Default 20% if IV not present.
        sigma = df["iv"].median() if "iv" in df.columns else 0.20

        # Get TTE - for 0DTE options, the parquet files often store tte_years as 0.
        # In that case, calculate actual time remaining from the interval's median
        # timestamp to market close (4 PM ET).
        tte = df["tte_years"].median() if "tte_years" in df.columns else 0.0
        if tte <= 0.0001:  # 0DTE stored as 0, need to calculate actual time remaining
            interval_time = df["timestamp"].median() if "timestamp" in df.columns else None
            if interval_time is not None:
                tte = calculate_tte_from_timestamp(
                    interval_time,
                    expiry_hour=self.config.time_window.expiry_hour
                )
            else:
                tte = 0.01  # Fallback: ~2.5 trading days

        # --- Greeks Calculation ---
        # Compute analytical Black-Scholes Greeks for each trade using a single
        # IV and TTE (interval-level) rather than per-trade values, for speed
        # and stability. Gamma is same for calls/puts; charm differs by type.
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

        # --- Side-Weighted Exposure Calculation ---
        # Multiply Greek values by trade size (contracts) and contract multiplier (100)
        # to get dollar-equivalent exposures.
        # gamma_exp uses absolute gamma (unsigned) for concentration metrics.
        # vomma/zomma/charm are signed by trade direction to capture net positioning.
        side_sign = self._get_side_signs(df)

        df["gamma_exp"] = df["gamma"].abs() * df["size"] * 100  # Unsigned: for GCI/PGR/GDW
        df["vomma_exp"] = df["vomma"] * df["size"] * 100 * side_sign  # Signed: for CAR
        df["zomma_exp"] = df["zomma"] * df["size"] * 100 * side_sign  # Signed: for CAR
        df["charm_exp"] = df["charm"] * df["size"] * side_sign         # Signed: for charm risk

        # --- Aggregate gamma by strike for concentration metrics ---
        gamma_by_strike = df.groupby("strike")["gamma_exp"].sum()
        gamma_total = gamma_by_strike.sum()

        if gamma_total == 0:
            return None

        # ----- GCI: Gamma Concentration Index (Herfindahl-Hirschman) -----
        # Step 1: Compute each strike's share of total gamma (sums to 1.0)
        gamma_shares = gamma_by_strike / gamma_total
        # Step 2: Square shares and sum. This is the HHI formula.
        # Perfectly uniform distribution across N strikes gives GCI = 1/N.
        # All gamma at one strike gives GCI = 1.0.
        # Typical values: 0.05-0.15 (normal), >0.25 (concentrated/dangerous).
        gci = (gamma_shares**2).sum()

        # ----- PGR: Protective Gamma Ratio -----
        # Measures what fraction of total gamma is "protecting" the current spot
        # by being within pgr_near_spot points (default 20 SPX points).
        # When PGR is high, ATM gamma acts as a cushion (dealers hedge into moves,
        # dampening them). When PGR drops, the cushion is gone.
        near_spot_mask = (
            gamma_by_strike.index >= spot - self.config.pgr_near_spot
        ) & (gamma_by_strike.index <= spot + self.config.pgr_near_spot)
        gamma_near = gamma_by_strike[near_spot_mask].sum()
        pgr = gamma_near / gamma_total

        # ----- GDW: Gamma Distance Weighted -----
        # Exponentially decaying weight by distance from spot.
        # Decay parameter (gdw_decay, default 20) controls how quickly weight
        # drops off. At distance = gdw_decay, weight = 1/e (~37%).
        # GDW captures "effective gamma" near spot better than simple sums.
        distances = np.abs(gamma_by_strike.index - spot)
        weights = np.exp(-distances / self.config.gdw_decay)
        gdw = (gamma_by_strike * weights).sum()

        # ----- CAR: Convexity Acceleration Risk -----
        # CAR captures the "feedback loop" risk: a vol spike changes gamma (zomma)
        # and vega (vomma), which forces more hedging, which moves spot, which
        # increases vol further.

        # Step 1: Compute net GEX to determine if dealers are net long or short gamma.
        # gex_exp is gamma_exp * side_sign (positive = dealer long gamma).
        df["gex_exp"] = df["gamma_exp"] * side_sign
        gex_by_strike = df.groupby("strike")["gex_exp"].sum()
        net_gex = gex_by_strike.sum()
        # gamma_sign: -1 when net short gamma (destabilizing), +1 when long gamma.
        gamma_sign = -1 if net_gex < 0 else 1

        # Step 2: Time amplifier. As TTE shrinks, convexity effects become more
        # extreme. Using 1/sqrt(TTE) with a cap of 30x to prevent blow-up.
        # At 1 hour to expiry (TTE ~ 0.0006), time_amp ~ 30 (capped).
        # At 2 hours (TTE ~ 0.0012), time_amp ~ 29.
        time_amp = min(30, 1 / np.sqrt(max(tte, 0.001)))

        # Step 3: Composite score. Zomma gets 60% weight (gamma feedback is primary
        # driver of PUT explosions), vomma gets 40% (vol sensitivity amplifies).
        # Divided by 1e6 to get millions for readable output.
        zomma_total = df["zomma_exp"].sum()
        vomma_total = df["vomma_exp"].sum()

        # CAR_net: signed by gamma regime. Negative = short gamma amplification risk.
        car_net = (
            gamma_sign * (0.6 * zomma_total + 0.4 * vomma_total) * time_amp / 1e6
        )
        # CAR_gross: unsigned magnitude. Total convexity risk regardless of direction.
        car_gross = (
            (0.6 * abs(zomma_total) + 0.4 * abs(vomma_total)) * time_amp / 1e6
        )

        # ----- Charm Risk -----
        # Aggregate signed charm exposure, divided by 1e6 for readability.
        # High charm risk = rapid delta decay forcing large hedging flows.
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
        """Convert IntervalMetrics dataclass to a flat dictionary.

        Used by DayProcessor to merge metric values into the per-interval
        results row before creating the output DataFrame.

        Args:
            metrics: Computed metrics for one interval.

        Returns:
            Dictionary with metric name keys and float/int values.
        """
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
