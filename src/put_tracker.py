"""PUT option price tracking for measuring actual returns after signals.

This module answers the core research question: "When a gamma metric spikes,
how much do OTM puts actually gain?"  It tracks PUT prices from signal entry
to multiple exit horizons, providing the outcome variable (% gain) that
``metrics.py`` signals are tested against in ``processor.py``.

Two strike selection methods are supported:
1. **N-strikes-OTM**: Select the Nth put strike below ATM (simple, robust).
2. **Max-vomma**: Select the strike with highest vomma exposure, weighted
   by volume.  Vomma-rich strikes see the largest price explosions when
   volatility spikes (calculated via ``greeks.BlackScholesGreeks``).

Pricing conventions:
- **Entry**: Use bid price (conservative -- assumes you'd have to sell
  to enter, or that you'd pay the bid to buy in a fast market).
- **Exit**: Use mid price (fair value estimate at exit time).
- **Time window**: Trades within +/- ``window_seconds`` of the target
  time are aggregated to estimate prices.

The ``PutTracker`` is instantiated by ``processor.DayProcessor`` and called
once per interval to produce return columns that feed into statistical
analysis (``statistics.py``) and visualization (``visualization.py``).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config, PutSelectionMethod
from .greeks import BlackScholesGreeks


@dataclass
class PutPrices:
    """Snapshot of PUT prices at a specific time, derived from trade-level data.

    Bid and ask are estimated from trade side classification (at_bid, at_ask, etc.)
    rather than from a direct quote feed.  When no bid-side or ask-side trades exist
    in the window, the min/max of all prices is used as a fallback.

    Attributes:
        strike: The option strike price.
        bid_price: Estimated bid (median of bid-side trades, or price min).
        mid_price: Median of all trade prices in the window.
        ask_price: Estimated ask (median of ask-side trades, or price max).
        n_trades: Number of trades in the time window.
        volume: Total contracts traded (sum of ``size`` column).
    """

    strike: float
    bid_price: float
    mid_price: float
    ask_price: float
    n_trades: int
    volume: int


@dataclass
class PutReturns:
    """PUT return profile from entry (signal time) to multiple exit horizons.

    Each time horizon (15m, 30m, 45m, 60m) has an exit price, percentage gain,
    and corresponding spot price change for reference.  Horizons that extend
    past 4:00 PM ET expiry are excluded (see ``TimeWindow.get_valid_time_horizons``).

    Attributes:
        entry_price: Bid price at signal time (conservative entry).
        strike: The selected put strike.
        entry_volume: Total volume at the strike around signal time.
        exit_prices: Map of minutes-after-signal -> mid exit price (None if unavailable).
        pct_gains: Map of minutes-after-signal -> percentage gain vs entry price.
        spot_changes: Map of minutes-after-signal -> SPX point change from signal.
        selection_method: Which method chose this strike ("n_strikes_otm" or "max_vomma").
    """

    entry_price: float
    strike: float
    entry_volume: int
    exit_prices: dict[int, Optional[float]]
    pct_gains: dict[int, Optional[float]]
    spot_changes: dict[int, Optional[float]]
    selection_method: str


class PutTracker:
    """Track PUT option prices and calculate returns after gamma-metric signals.

    Orchestrates strike selection, price lookup, and return calculation for
    each signal interval.  Used by ``processor.DayProcessor.process()`` to
    attach PUT return columns alongside the gamma metrics computed by
    ``metrics.MetricCalculator``.
    """

    def __init__(self, config: Config):
        """Initialize with configuration.

        Args:
            config: Master ``Config`` object; uses ``put_selection`` for strike
                selection parameters and ``time_window`` for expiry-aware horizon
                capping.
        """
        self.config = config
        self.put_selection = config.put_selection
        self.time_window = config.time_window
        self.greeks_calc = BlackScholesGreeks(config.risk_free_rate)

    def _get_opt_type_col(self, df: pd.DataFrame) -> str:
        """Get the option type column name (handles both ``opt_type`` and ``option_type``)."""
        return "opt_type" if "opt_type" in df.columns else "option_type"

    def _is_put(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean mask identifying PUT options (handles 'P' and 'PUT' values)."""
        opt_type_col = self._get_opt_type_col(df)
        opt_type_vals = df[opt_type_col].str.upper()
        return opt_type_vals.isin(["P", "PUT"])

    def select_strike_n_otm(
        self,
        df: pd.DataFrame,
        spot: float,
        n_strikes: int = 2,
    ) -> Optional[float]:
        """Select PUT strike that is N strikes below ATM.

        Args:
            df: Trade data with PUT options
            spot: Current spot price
            n_strikes: Number of strikes below ATM (default 2)

        Returns:
            Selected strike or None if not enough strikes
        """
        # Filter to PUTs
        df_puts = df[self._is_put(df)]
        if len(df_puts) == 0:
            return None

        # Get unique strikes
        available_strikes = sorted(df_puts["strike"].unique())

        # Find ATM strike (closest to spot)
        atm_idx = np.argmin(np.abs(np.array(available_strikes) - spot))
        atm_strike = available_strikes[atm_idx]

        # Get strikes below ATM (OTM for puts)
        otm_strikes = [s for s in available_strikes if s < atm_strike]
        otm_strikes = sorted(otm_strikes, reverse=True)  # Highest first (closest to ATM)

        # Select Nth strike OTM
        if len(otm_strikes) >= n_strikes:
            return otm_strikes[n_strikes - 1]
        elif otm_strikes:
            return otm_strikes[-1]  # Furthest available
        return None

    def select_strike_max_vomma(
        self,
        df: pd.DataFrame,
        spot: float,
        iv: float = 0.20,
        tte: float = 0.01,
    ) -> Optional[float]:
        """Select PUT strike with highest vomma exposure.

        Vomma (vol-of-vol sensitivity) indicates where option prices
        will explode most during volatility spikes.

        Args:
            df: Trade data with PUT options
            spot: Current spot price
            iv: Implied volatility
            tte: Time to expiry in years

        Returns:
            Strike with maximum vomma or None
        """
        # Filter to PUTs in the search range
        otm_min, otm_max = self.put_selection.vomma_otm_range
        df_puts = df[
            self._is_put(df)
            & (df["strike"] >= spot - otm_max)
            & (df["strike"] <= spot - otm_min)
        ].copy()

        if len(df_puts) == 0:
            return None

        # Get unique strikes
        strikes = df_puts["strike"].unique()

        # Calculate vomma for each strike
        vomma_by_strike = {}
        for strike in strikes:
            # Vomma calculation using greeks module
            greeks = self.greeks_calc.calculate_all(
                S=spot,
                K=np.array([strike]),
                T=tte,
                sigma=iv,
                is_call=np.array([False]),  # PUT
            )
            vomma = greeks["vomma"][0]

            # Weight by volume at this strike
            strike_volume = df_puts[df_puts["strike"] == strike]["size"].sum()
            vomma_by_strike[strike] = abs(vomma) * max(strike_volume, 1)

        if not vomma_by_strike:
            return None

        # Return strike with maximum vomma
        return max(vomma_by_strike, key=vomma_by_strike.get)

    def get_prices_at_time(
        self,
        df: pd.DataFrame,
        strike: float,
        target_time: pd.Timestamp,
        window_seconds: int = 60,
    ) -> Optional[PutPrices]:
        """Get PUT prices for a specific strike around a target time.

        Uses conservative pricing: bid for entry assumption.

        Args:
            df: Trade data
            strike: Strike price to look up
            target_time: Target timestamp
            window_seconds: Time window for price lookup

        Returns:
            PutPrices or None if no data
        """
        # Filter to PUTs at the specified strike
        df_puts = df[self._is_put(df) & (df["strike"] == strike)].copy()

        if len(df_puts) == 0:
            return None

        # Filter to time window
        time_min = target_time - pd.Timedelta(seconds=window_seconds)
        time_max = target_time + pd.Timedelta(seconds=window_seconds)

        df_window = df_puts[
            (df_puts["timestamp"] >= time_min) & (df_puts["timestamp"] <= time_max)
        ]

        if len(df_window) == 0:
            return None

        # Separate by side for bid/ask estimation
        bid_trades = df_window[df_window["side"].isin(["at_bid", "below_bid"])]
        ask_trades = df_window[df_window["side"].isin(["at_ask", "above_ask"])]

        # Calculate prices
        bid_price = (
            bid_trades["price"].median()
            if len(bid_trades) > 0
            else df_window["price"].min()
        )
        ask_price = (
            ask_trades["price"].median()
            if len(ask_trades) > 0
            else df_window["price"].max()
        )
        mid_price = df_window["price"].median()

        return PutPrices(
            strike=strike,
            bid_price=bid_price,
            mid_price=mid_price,
            ask_price=ask_price,
            n_trades=len(df_window),
            volume=int(df_window["size"].sum()),
        )

    def calculate_returns_for_strike(
        self,
        df: pd.DataFrame,
        strike: float,
        signal_time: pd.Timestamp,
        spot_at_signal: float,
        selection_method: str,
    ) -> Optional[PutReturns]:
        """Calculate PUT returns for a specific strike.

        Uses bid for entry (conservative) and mid for exit.

        Args:
            df: Trade data
            strike: Selected strike
            signal_time: When the signal occurred
            spot_at_signal: Spot price at signal time
            selection_method: "n_strikes_otm" or "max_vomma"

        Returns:
            PutReturns or None if no entry price available
        """
        # Get entry price (bid)
        entry = self.get_prices_at_time(df, strike, signal_time)

        if entry is None or entry.bid_price is None or entry.bid_price <= 0:
            return None

        entry_price = entry.bid_price

        exit_prices: dict[int, Optional[float]] = {}
        pct_gains: dict[int, Optional[float]] = {}
        spot_changes: dict[int, Optional[float]] = {}

        # Get signal time in minutes for expiry-aware horizons
        signal_minutes = signal_time.hour * 60 + signal_time.minute
        valid_horizons = self.time_window.get_valid_time_horizons(signal_minutes)

        # Get prices at each valid time horizon
        for minutes in valid_horizons:
            exit_time = signal_time + pd.Timedelta(minutes=minutes)

            # Get spot at exit time for reference
            df_exit = df[
                df["timestamp"].between(
                    exit_time - pd.Timedelta(seconds=30),
                    exit_time + pd.Timedelta(seconds=30),
                )
            ]
            spot_at_exit = (
                df_exit["spot"].median() if len(df_exit) > 0 else spot_at_signal
            )

            # Get PUT price at exit (use mid for exit)
            exit_price_data = self.get_prices_at_time(df, strike, exit_time)

            if exit_price_data is not None and exit_price_data.mid_price is not None:
                exit_price = exit_price_data.mid_price
                pct_gain = (exit_price - entry_price) / entry_price * 100
            else:
                exit_price = None
                pct_gain = None

            exit_prices[minutes] = exit_price
            pct_gains[minutes] = pct_gain
            spot_changes[minutes] = (
                spot_at_exit - spot_at_signal if spot_at_exit else None
            )

        return PutReturns(
            entry_price=entry_price,
            strike=strike,
            entry_volume=entry.volume,
            exit_prices=exit_prices,
            pct_gains=pct_gains,
            spot_changes=spot_changes,
            selection_method=selection_method,
        )

    def calculate_returns(
        self,
        df: pd.DataFrame,
        signal_time: pd.Timestamp,
        spot_at_signal: float,
    ) -> dict[str, Optional[PutReturns]]:
        """Calculate PUT returns using all configured selection methods.

        Args:
            df: Trade data
            signal_time: When the signal occurred
            spot_at_signal: Spot price at signal time

        Returns:
            Dictionary mapping method name to PutReturns (or None)
        """
        results = {}

        # Get IV and TTE from data
        iv = df["iv"].median() if "iv" in df.columns else 0.20
        tte = df["tte_years"].median() if "tte_years" in df.columns else 0.01

        for method in self.put_selection.methods:
            if method == PutSelectionMethod.N_STRIKES_OTM:
                strike = self.select_strike_n_otm(
                    df, spot_at_signal, self.put_selection.n_strikes_otm
                )
                method_name = "n_strikes_otm"
            elif method == PutSelectionMethod.MAX_VOMMA:
                strike = self.select_strike_max_vomma(
                    df, spot_at_signal, iv=iv, tte=tte
                )
                method_name = "max_vomma"
            else:
                continue

            if strike is not None:
                returns = self.calculate_returns_for_strike(
                    df, strike, signal_time, spot_at_signal, method_name
                )
                results[method_name] = returns
            else:
                results[method_name] = None

        return results

    def returns_to_dict(self, returns: PutReturns, prefix: str = "") -> dict:
        """Convert PutReturns to flat dictionary for DataFrame.

        Args:
            returns: PutReturns object
            prefix: Prefix for column names (e.g., "vomma_" for max_vomma method)

        Returns:
            Dictionary of column name -> value
        """
        result = {
            f"{prefix}entry_price": returns.entry_price,
            f"{prefix}strike": returns.strike,
            f"{prefix}entry_volume": returns.entry_volume,
        }

        for minutes in [15, 30, 45, 60]:
            result[f"{prefix}exit_price_{minutes}m"] = returns.exit_prices.get(minutes)
            result[f"{prefix}pct_gain_{minutes}m"] = returns.pct_gains.get(minutes)
            result[f"{prefix}spot_change_{minutes}m"] = returns.spot_changes.get(minutes)

        return result

    def all_returns_to_dict(
        self, returns_by_method: dict[str, Optional[PutReturns]]
    ) -> dict:
        """Convert all method returns to a single flat dictionary.

        Args:
            returns_by_method: Dictionary mapping method name to PutReturns

        Returns:
            Flat dictionary with prefixed columns for each method
        """
        result = {}

        for method_name, returns in returns_by_method.items():
            if returns is not None:
                prefix = f"{method_name}_" if method_name != "n_strikes_otm" else ""
                result.update(self.returns_to_dict(returns, prefix))

        return result
