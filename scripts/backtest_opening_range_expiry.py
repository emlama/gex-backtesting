"""Price-only proxy for an opening-range SPXW short-option expiry test.

Up breakout -> short put at the first 30-minute low.
Down breakout -> short call at the first 30-minute high.

A breakout is confirmed by a one-minute close after 10:00 ET. The hypothetical
entry is the following minute. This script tests whether the option would have
zero intrinsic value at Cboe's SPXW expiration settlement value. It does not use option
quotes or establish that the contract/entry was available to trade.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def run(minute_file: Path, settlement_file: Path, include_incomplete: bool = False) -> pd.DataFrame:
    bars = pd.read_parquet(minute_file)
    closes = pd.read_csv(settlement_file).set_index("date")["settlement"]
    rows = []

    for date, day in bars.groupby("date", sort=True):
        complete = len(day) == 390
        opening = day[day["time"] < "10:00"]
        if len(opening) != 30 or (not complete and not include_incomplete):
            continue
        # Standard monthly Fridays may have no PM-settled SPXW contract.
        if date not in closes.index:
            continue

        high = float(opening["high"].max())
        low = float(opening["low"].min())
        settlement = float(closes.loc[date])
        # A signal on the 15:59 bar is known at 16:00, too late to enter.
        later = day[(day["time"] >= "10:00") & (day["time"] < "15:59")]
        signals = (
            ("up", "put", later[later["close"] > high], low, np.floor(low / 5) * 5),
            ("down", "call", later[later["close"] < low], high, np.ceil(high / 5) * 5),
        )
        for direction, option_type, hits, strike, rounded_strike in signals:
            if hits.empty:
                continue
            first = hits.iloc[0]
            if option_type == "put":
                zero = settlement >= strike
                zero_rounded = settlement >= rounded_strike
            else:
                zero = settlement <= strike
                zero_rounded = settlement <= rounded_strike
            rows.append(
                {
                    "date": date,
                    "complete_day": complete,
                    "direction": direction,
                    "option_type": option_type,
                    "signal_bar_start_et": first["bar_start_et"],
                    "earliest_entry_et": first["known_at_et"],
                    "opening_high": high,
                    "opening_low": low,
                    "signal_close": float(first["close"]),
                    "strike_exact": strike,
                    "strike_5pt_assumption": rounded_strike,
                    "spxw_settlement": settlement,
                    "expires_zero_exact": bool(zero),
                    "expires_zero_5pt_assumption": bool(zero_rounded),
                }
            )

    result = pd.DataFrame(rows).sort_values(["date", "signal_bar_start_et", "direction"])
    result["first_breakout_of_day"] = ~result["date"].duplicated()
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minute-file", type=Path, default=Path("data/spx_1m_indicators.parquet"))
    parser.add_argument("--settlement-file", type=Path, default=Path("data/spxw_settlements_2026.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/opening_range_expiry_proxy.csv"))
    parser.add_argument("--include-incomplete", action="store_true")
    args = parser.parse_args()

    result = run(args.minute_file, args.settlement_file, args.include_incomplete)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    for label, subset in (
        ("Up breakout -> short put", result[result["direction"] == "up"]),
        ("Down breakout -> short call", result[result["direction"] == "down"]),
        ("Both sides, one signal each", result),
        ("First breakout only", result[result["first_breakout_of_day"]]),
    ):
        wins = int(subset["expires_zero_exact"].sum())
        print(f"{label}: {wins}/{len(subset)} = {wins / len(subset):.1%} zero at expiry")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
