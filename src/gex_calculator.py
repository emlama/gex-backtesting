"""GEX Calculator - Side-Weighted Gamma Exposure Analysis.

Calculates Traditional and Side-Weighted GEX from SPX 0DTE options trade data.
Used by the analysis notebooks to produce GEX-by-strike bar charts.

GEX Formula:
    GEX_per_trade = Size * Gamma * Spot^2 * 0.01

    The 0.01 factor converts the dollar-gamma to a "per 1% move" basis:
    GEX represents the dollar value of delta-hedging flows triggered by
    a 1% move in the underlying.

Two GEX flavors are computed:

    Traditional GEX:
        Assumes all trades are customer buys (the academic standard).
        Calls contribute positive GEX (dealer is short calls -> long gamma).
        Puts contribute negative GEX (dealer is short puts -> short gamma).
        Formula: trad_gex = +gex_raw for calls, -gex_raw for puts.

    Side-Weighted GEX:
        Uses actual trade direction (BUY/SELL) inferred from trade-side
        classification (at_ask = buy, at_bid = sell).
        - Customer BUY call  -> dealer short call -> +GEX (positive / stabilizing)
        - Customer SELL call -> dealer long call  -> -GEX (negative / destabilizing)
        - Customer BUY put   -> dealer short put  -> -GEX (negative / destabilizing)
        - Customer SELL put  -> dealer long put   -> +GEX (positive / stabilizing)

    Positive net GEX = market makers are net long gamma (stabilizing).
    Negative net GEX = market makers are net short gamma (destabilizing).

Dependencies:
    - calculate_greeks (black_scholes.py): Vectorized IV and gamma computation
    - AnalysisConfig (config.py): Strike range, risk-free rate, trade-side mapping
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .black_scholes import calculate_greeks
from .config import AnalysisConfig


@dataclass
class GEXResult:
    """Results from a single-day GEX calculation.

    Contains both the per-strike breakdown DataFrame and aggregate totals
    for traditional and side-weighted GEX. Used by notebooks to produce
    GEX bar charts and summary statistics.

    Attributes:
        by_strike: DataFrame with per-strike GEX values. Columns include
            trad_gex, sw_gex, volume, trade_count, and call/put breakdowns.
        spot: SPX spot price (from most recent trade in the dataset).
        trade_count: Total number of valid trades (after Greek filtering).
        contract_count: Total contract volume (sum of trade sizes).
        trad_net: Net traditional GEX across all strikes (calls positive, puts negative).
        trad_call: Traditional GEX from call options only.
        trad_put: Traditional GEX from put options only (negative value).
        sw_net: Net side-weighted GEX. Positive = stabilizing, negative = destabilizing.
        sw_call: Side-weighted GEX from calls only.
        sw_put: Side-weighted GEX from puts only.
        call_buy_pct: Percentage of call volume classified as customer buys.
        put_buy_pct: Percentage of put volume classified as customer buys.
    """

    by_strike: pd.DataFrame
    spot: float
    trade_count: int
    contract_count: int

    # Traditional GEX
    trad_net: float
    trad_call: float
    trad_put: float

    # Side-Weighted GEX
    sw_net: float
    sw_call: float
    sw_put: float

    # Buy percentages
    call_buy_pct: float
    put_buy_pct: float


def calculate_gex(
    df: pd.DataFrame,
    config: AnalysisConfig,
    verbose: bool = True,
) -> GEXResult:
    """Calculate Traditional and Side-Weighted GEX from trade data.

    This is the main entry point for GEX analysis. It computes Greeks from
    market prices (IV implied from trade prices), calculates per-trade GEX,
    and aggregates by strike for charting.

    Args:
        df: Trade DataFrame with required columns:
            - spot: SPX price at trade time
            - strike: Option strike price
            - tte_years: Time to expiry in years
            - price: Trade price (used to imply IV)
            - opt_type: "C" or "P"
            - size: Number of contracts
            - trade_dir: "BUY" or "SELL" (derived from Polygon trade side)
            - timestamp: Trade timestamp (used to determine most recent spot)
        config: AnalysisConfig with strike_range and risk_free_rate.
        verbose: If True, print progress messages to stdout.

    Returns:
        GEXResult with per-strike breakdown and aggregate GEX totals.

    Raises:
        ValueError: If df is empty.
    """
    if df.empty:
        raise ValueError("No trade data provided")

    if verbose:
        print(f"Calculating GEX for {len(df):,} trades...")

    spot = df["spot"].values
    strike = df["strike"].values
    tte = df["tte_years"].values
    price = df["price"].values
    is_call = (df["opt_type"] == "C").values
    trade_dir = df["trade_dir"].values

    if verbose:
        print("  Calculating Greeks (vectorized)...")

    # Compute IV (implied from trade price) and Greeks for all trades at once.
    # This uses the vectorized Black-Scholes implementation (~100x faster than iterrows).
    greeks = calculate_greeks(spot=spot, strike=strike, tte=tte, rate=config.risk_free_rate, price=price, is_call=is_call)

    df = df.copy()
    df["iv"] = greeks["iv"]
    df["gamma"] = greeks["gamma"]
    df["delta"] = greeks["delta"]

    # Filter trades with invalid Greeks: gamma outside [0, 1] or IV outside [1%, 300%].
    # This removes trades where the IV solver failed (e.g., deep ITM with no time value)
    # or produced nonsensical results.
    valid_mask = (df["gamma"] > 0) & (df["gamma"] < 1) & (df["iv"] > 0.01) & (df["iv"] < 3.0)
    df = df[valid_mask].copy()

    if verbose:
        print(f"  Valid Greeks: {len(df):,} trades")

    # --- Per-trade GEX calculation ---
    # GEX = Size * Gamma * Spot^2 * 0.01
    # The 0.01 factor normalizes to "per 1% move in the underlying".
    df["gex_raw"] = df["size"] * df["gamma"] * df["spot"] ** 2 * 0.01

    # Traditional GEX: assumes all options were customer-bought.
    # Calls -> dealer short calls -> positive gamma (stabilizing) -> +GEX
    # Puts  -> dealer short puts  -> negative gamma (destabilizing) -> -GEX
    df["trad_gex"] = np.where(df["opt_type"] == "C", df["gex_raw"], -df["gex_raw"])

    # Side-Weighted GEX: uses actual trade direction to determine dealer positioning.
    # BUY call  -> dealer short call -> +GEX | SELL call -> dealer long call -> -GEX
    # BUY put   -> dealer short put  -> -GEX | SELL put  -> dealer long put  -> +GEX
    is_buy = df["trade_dir"] == "BUY"
    is_call_mask = df["opt_type"] == "C"
    df["sw_gex"] = np.where(
        is_buy,
        np.where(is_call_mask, df["gex_raw"], -df["gex_raw"]),
        np.where(is_call_mask, -df["gex_raw"], df["gex_raw"]),
    )

    if verbose:
        print("  Aggregating by strike...")

    # Use spot from the most recent trade (closest to "current" market state).
    df_with_spot = df[df["spot"].notna()]
    if len(df_with_spot) > 0:
        most_recent = df_with_spot.loc[df_with_spot["timestamp"].idxmax()]
        calculated_spot = most_recent["spot"]
    else:
        calculated_spot = df["spot"].median()

    if verbose:
        print(f"  Spot: ${calculated_spot:,.2f}")

    # Filter to strikes within config.strike_range of spot for charting.
    df_filtered = df[
        (df["strike"] >= calculated_spot - config.strike_range)
        & (df["strike"] <= calculated_spot + config.strike_range)
    ]

    # --- Aggregate by strike ---
    # Main aggregation: total GEX (both flavors), volume, and trade count per strike.
    by_strike = (
        df_filtered.groupby("strike")
        .agg(trad_gex=("trad_gex", "sum"), sw_gex=("sw_gex", "sum"), volume=("size", "sum"), trade_count=("size", "count"))
        .reset_index()
    )

    # Separate call/put breakdowns for the stacked bar chart.
    calls = df_filtered[df_filtered["opt_type"] == "C"]
    puts = df_filtered[df_filtered["opt_type"] == "P"]

    call_by_strike = calls.groupby("strike").agg(
        trad_call_gex=("trad_gex", "sum"), sw_call_gex=("sw_gex", "sum"), call_vol=("size", "sum")
    ).reset_index()

    put_by_strike = puts.groupby("strike").agg(
        trad_put_gex=("trad_gex", "sum"), sw_put_gex=("sw_gex", "sum"), put_vol=("size", "sum")
    ).reset_index()

    # Merge call/put breakdowns onto the main by-strike DataFrame.
    # Outer join to keep strikes that only have calls or only have puts.
    by_strike = by_strike.merge(call_by_strike, on="strike", how="outer")
    by_strike = by_strike.merge(put_by_strike, on="strike", how="outer")
    by_strike = by_strike.fillna(0).sort_values("strike")

    # Compute buy percentages for the summary (what fraction of volume is customer buys).
    total_call_vol = calls["size"].sum()
    total_put_vol = puts["size"].sum()
    call_buy_vol = calls[calls["trade_dir"] == "BUY"]["size"].sum()
    put_buy_vol = puts[puts["trade_dir"] == "BUY"]["size"].sum()

    result = GEXResult(
        by_strike=by_strike,
        spot=calculated_spot,
        trade_count=len(df),
        contract_count=int(df["size"].sum()),
        trad_net=by_strike["trad_gex"].sum(),
        trad_call=by_strike["trad_call_gex"].sum(),
        trad_put=by_strike["trad_put_gex"].sum(),
        sw_net=by_strike["sw_gex"].sum(),
        sw_call=by_strike["sw_call_gex"].sum(),
        sw_put=by_strike["sw_put_gex"].sum(),
        call_buy_pct=call_buy_vol / total_call_vol * 100 if total_call_vol > 0 else 0,
        put_buy_pct=put_buy_vol / total_put_vol * 100 if total_put_vol > 0 else 0,
    )

    if verbose:
        print()
        print("=" * 60)
        print("GEX ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Spot: ${result.spot:,.0f}")
        print(f"Trades: {result.trade_count:,} | Contracts: {result.contract_count:,}")
        print()
        print("Side-Weighted GEX (actual buy/sell):")
        print(f"  Net:   ${result.sw_net / 1e6:,.2f}M", end="")
        print(" (STABILIZING)" if result.sw_net > 0 else " (DESTABILIZING)")
        print(f"  Calls: ${result.sw_call / 1e6:,.2f}M | Puts: ${result.sw_put / 1e6:,.2f}M")
        print("=" * 60)

    return result
