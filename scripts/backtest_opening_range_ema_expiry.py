"""Compare 0DTE and next-trading-day SPXW expiry with an EMA30 ORB filter.

The first 30 minutes set the high/low. After 10:00 ET, a one-minute close above
the high is an up breakout; below the low is a down breakout. Keep the first
breakout of each side only when the close is also above/below its one-minute
EMA30. Up sells a put at the opening low; down sells a call at the opening high.

This tests zero intrinsic value at official SPXW settlement, not actual fills
or P&L. 1DTE means expiry on the next trading day, including Monday after Friday.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(label: str, data: pd.DataFrame) -> None:
    print(label)
    for dte in (0, 1):
        eligible = data[data[f"expiry_{dte}dte"].notna()]
        wins = int(eligible[f"zero_{dte}dte"].astype(int).sum())
        rate = wins / len(eligible) if len(eligible) else float("nan")
        print(f"  {dte}DTE: {wins}/{len(eligible)} = {rate:.1%}")
    paired = data[data["expiry_0dte"].notna() & data["expiry_1dte"].notna()]
    for dte in (0, 1):
        wins = int(paired[f"zero_{dte}dte"].astype(int).sum())
        rate = wins / len(paired) if len(paired) else float("nan")
        print(f"  paired {dte}DTE: {wins}/{len(paired)} = {rate:.1%}")


def run(minute_file: Path, settlement_file: Path, calendar_file: Path | None = None) -> pd.DataFrame:
    bars = pd.read_parquet(minute_file)
    if "ema_30" not in bars.columns:
        raise ValueError("Minute file has no ema_30; rebuild it with build_spx_minute_indicators.py")
    settlements = pd.read_csv(settlement_file).set_index("date")["settlement"]
    calendar = (
        pd.read_csv(calendar_file)["date"].tolist()
        if calendar_file is not None
        else sorted(bars["date"].unique())
    )
    next_day = dict(zip(calendar, calendar[1:]))
    rows = []

    for date, day in bars.groupby("date", sort=True):
        opening = day[day["time"] < "10:00"]
        if len(day) != 390 or len(opening) != 30:
            continue
        high, low = float(opening["high"].max()), float(opening["low"].min())
        later = day[(day["time"] >= "10:00") & (day["time"] < "15:59")]
        signals = (
            ("up", "put", later[later["close"] > high], low, np.floor(low / 5) * 5),
            ("down", "call", later[later["close"] < low], high, np.ceil(high / 5) * 5),
        )
        for direction, option_type, hits, strike, strike_5pt in signals:
            if hits.empty:
                continue
            first = hits.iloc[0]
            aligned = first["close"] > first["ema_30"] if direction == "up" else first["close"] < first["ema_30"]
            row = {
                "date": date,
                "direction": direction,
                "option_type": option_type,
                "signal_bar_start_et": first["bar_start_et"],
                "earliest_entry_et": first["known_at_et"],
                "signal_close": float(first["close"]),
                "ema_30_at_signal": float(first["ema_30"]),
                "ema_aligned": bool(aligned),
                "opening_high": high,
                "opening_low": low,
                "strike_exact": strike,
                "strike_5pt_assumption": strike_5pt,
            }
            for dte, expiry in ((0, date), (1, next_day.get(date))):
                available = expiry in settlements.index if expiry else False
                row[f"expiry_{dte}dte"] = expiry if available else None
                if available:
                    settlement = float(settlements.loc[expiry])
                    row[f"settlement_{dte}dte"] = settlement
                    row[f"zero_{dte}dte"] = settlement >= strike if option_type == "put" else settlement <= strike
                    row[f"zero_{dte}dte_5pt"] = (
                        settlement >= strike_5pt if option_type == "put" else settlement <= strike_5pt
                    )
                else:
                    row[f"settlement_{dte}dte"] = None
                    row[f"zero_{dte}dte"] = None
                    row[f"zero_{dte}dte_5pt"] = None
            rows.append(row)

    result = pd.DataFrame(rows).sort_values(["date", "signal_bar_start_et", "direction"])
    result["first_aligned_breakout_of_day"] = False
    aligned_indices = result[result["ema_aligned"]].drop_duplicates("date").index
    result.loc[aligned_indices, "first_aligned_breakout_of_day"] = True
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minute-file", type=Path, default=Path("data/spx_1m_indicators.parquet"))
    parser.add_argument("--settlement-file", type=Path, default=Path("data/spxw_settlements_2026.csv"))
    parser.add_argument(
        "--calendar-file", type=Path, help="Optional index.csv from spx-0dte-archive; defaults to dates in minute data"
    )
    parser.add_argument("--output", type=Path, default=Path("results/opening_range_ema30_expiry.csv"))
    args = parser.parse_args()

    result = run(args.minute_file, args.settlement_file, args.calendar_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    aligned = result[result["ema_aligned"]]
    print(f"Breakouts: {len(result)}, EMA30-aligned: {len(aligned)}")
    summarize("One breakout per side", aligned)
    summarize("First aligned breakout per day", aligned[aligned["first_aligned_breakout_of_day"]])
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
