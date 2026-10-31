"""Test a 10:30 ET SPXW short strangle after both opening-range breakouts.

The 09:30–09:59 ET range sets the breakout levels. Both an up and a down
breakout must be confirmed by one-minute closes during 10:00–10:29. At 10:30,
the theoretical short call is at the 09:30–10:29 high and the short put is at
the corresponding low. Fixed-width protective legs illustrate credit-spread
expiry debits. Option entry credits and actual listed strikes are unavailable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def run(minute_file: Path, settlement_file: Path) -> pd.DataFrame:
    bars = pd.read_parquet(minute_file)
    settlements = pd.read_csv(settlement_file).set_index("date")["settlement"]
    rows = []

    for date, day in bars.groupby("date", sort=True):
        if len(day) != 390 or date not in settlements.index:
            continue

        opening = day[day["time"] < "10:00"]
        second_half = day[(day["time"] >= "10:00") & (day["time"] < "10:30")]
        first_hour = day[day["time"] < "10:30"]
        if len(opening) != 30 or len(second_half) != 30 or len(first_hour) != 60:
            continue

        opening_high = float(opening["high"].max())
        opening_low = float(opening["low"].min())
        up = second_half[second_half["close"] > opening_high]
        down = second_half[second_half["close"] < opening_low]
        if up.empty or down.empty:
            continue

        call_strike = float(first_hour["high"].max())
        put_strike = float(first_hour["low"].min())
        settlement = float(settlements.loc[date])
        call_debit = round(max(settlement - call_strike, 0.0), 2)
        put_debit = round(max(put_strike - settlement, 0.0), 2)
        rounded_call = float(np.ceil(call_strike / 5) * 5)
        rounded_put = float(np.floor(put_strike / 5) * 5)

        row = {
            "date": date,
            "entry_et": first_hour.iloc[-1]["known_at_et"],
            "up_break_bar_start_et": up.iloc[0]["bar_start_et"],
            "down_break_bar_start_et": down.iloc[0]["bar_start_et"],
            "opening_30m_high": opening_high,
            "opening_30m_low": opening_low,
            "short_call_strike_exact": call_strike,
            "short_put_strike_exact": put_strike,
            "short_call_strike_5pt_assumption": rounded_call,
            "short_put_strike_5pt_assumption": rounded_put,
            "spxw_settlement": settlement,
            "call_zero_exact": call_debit == 0,
            "put_zero_exact": put_debit == 0,
            "both_zero_exact": call_debit == 0 and put_debit == 0,
            "both_zero_5pt_assumption": rounded_put <= settlement <= rounded_call,
            "call_intrinsic_exact": call_debit,
            "put_intrinsic_exact": put_debit,
        }
        for width in (5, 10, 20):
            row[f"both_spreads_expiry_debit_{width}pt"] = min(call_debit, width) + min(put_debit, width)
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minute-file", type=Path, default=Path("data/spx_1m_indicators.parquet"))
    parser.add_argument("--settlement-file", type=Path, default=Path("data/spxw_settlements_2026.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/double_breakout_first_hour.csv"))
    args = parser.parse_args()

    result = run(args.minute_file, args.settlement_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    for column in ("call_zero_exact", "put_zero_exact", "both_zero_exact", "both_zero_5pt_assumption"):
        print(f"{column}: {int(result[column].sum())}/{len(result)}")
    for width in (5, 10, 20):
        column = f"both_spreads_expiry_debit_{width}pt"
        print(f"{width}-point wings: mean combined expiry debit {result[column].mean():.2f} SPX points")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
