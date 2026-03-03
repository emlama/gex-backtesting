"""GEX Calculator - Side-Weighted Gamma Exposure Analysis.

Calculates Traditional and Side-Weighted GEX from 0DTE options trade data.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .black_scholes import calculate_greeks
from .config import AnalysisConfig


@dataclass
class GEXResult:
    """Results from GEX calculation."""

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

    Args:
        df: Trade DataFrame with columns: spot, strike, tte_years, price, opt_type, size, trade_dir
        config: Analysis configuration
        verbose: Print progress
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

    greeks = calculate_greeks(spot=spot, strike=strike, tte=tte, rate=config.risk_free_rate, price=price, is_call=is_call)

    df = df.copy()
    df["iv"] = greeks["iv"]
    df["gamma"] = greeks["gamma"]
    df["delta"] = greeks["delta"]

    # Filter invalid Greeks
    valid_mask = (df["gamma"] > 0) & (df["gamma"] < 1) & (df["iv"] > 0.01) & (df["iv"] < 3.0)
    df = df[valid_mask].copy()

    if verbose:
        print(f"  Valid Greeks: {len(df):,} trades")

    # GEX per trade
    df["gex_raw"] = df["size"] * df["gamma"] * df["spot"] ** 2 * 0.01

    # Traditional GEX (assumes all customer buys)
    df["trad_gex"] = np.where(df["opt_type"] == "C", df["gex_raw"], -df["gex_raw"])

    # Side-Weighted GEX (uses actual trade direction)
    is_buy = df["trade_dir"] == "BUY"
    is_call_mask = df["opt_type"] == "C"
    df["sw_gex"] = np.where(
        is_buy,
        np.where(is_call_mask, df["gex_raw"], -df["gex_raw"]),
        np.where(is_call_mask, -df["gex_raw"], df["gex_raw"]),
    )

    if verbose:
        print("  Aggregating by strike...")

    # Spot from most recent trade
    df_with_spot = df[df["spot"].notna()]
    if len(df_with_spot) > 0:
        most_recent = df_with_spot.loc[df_with_spot["timestamp"].idxmax()]
        calculated_spot = most_recent["spot"]
    else:
        calculated_spot = df["spot"].median()

    if verbose:
        print(f"  Spot: ${calculated_spot:,.2f}")

    # Filter to strike range
    df_filtered = df[
        (df["strike"] >= calculated_spot - config.strike_range)
        & (df["strike"] <= calculated_spot + config.strike_range)
    ]

    # Aggregate by strike
    by_strike = (
        df_filtered.groupby("strike")
        .agg(trad_gex=("trad_gex", "sum"), sw_gex=("sw_gex", "sum"), volume=("size", "sum"), trade_count=("size", "count"))
        .reset_index()
    )

    calls = df_filtered[df_filtered["opt_type"] == "C"]
    puts = df_filtered[df_filtered["opt_type"] == "P"]

    call_by_strike = calls.groupby("strike").agg(
        trad_call_gex=("trad_gex", "sum"), sw_call_gex=("sw_gex", "sum"), call_vol=("size", "sum")
    ).reset_index()

    put_by_strike = puts.groupby("strike").agg(
        trad_put_gex=("trad_gex", "sum"), sw_put_gex=("sw_gex", "sum"), put_vol=("size", "sum")
    ).reset_index()

    by_strike = by_strike.merge(call_by_strike, on="strike", how="outer")
    by_strike = by_strike.merge(put_by_strike, on="strike", how="outer")
    by_strike = by_strike.fillna(0).sort_values("strike")

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
