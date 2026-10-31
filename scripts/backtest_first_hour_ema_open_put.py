"""Test a short put at the SPX open after EMA30 exceeds EMA50 all first hour.

The 09:30–10:29 ET one-minute bars must each have EMA30 > EMA50. This is
observable at 10:30 ET, the earliest hypothetical entry. The theoretical put
strike is the 09:30 SPX open. Test zero intrinsic value at official SPXW PM
settlement on the signal date (0DTE) and next trading date (1DTE).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def run(minute_file: Path, settlement_file: Path, calendar_file: Path | None = None) -> pd.DataFrame:
    bars = pd.read_parquet(minute_file)
    required = {"date", "time", "open", "ema_30", "ema_50", "known_at_et"}
    if missing := required - set(bars.columns):
        raise ValueError(f"Minute file is missing columns: {sorted(missing)}")

    settlements = pd.read_csv(settlement_file).set_index("date")["settlement"]
    if settlements.index.has_duplicates:
        raise ValueError("Multiple settlements found on one date")
    calendar = (
        sorted(pd.read_csv(calendar_file)["date"].unique())
        if calendar_file is not None
        else sorted(bars["date"].unique())
    )
    next_day = dict(zip(calendar, calendar[1:]))
    expected_day = pd.date_range("09:30", "15:59", freq="min").strftime("%H:%M").tolist()
    first_hour_times = expected_day[:60]
    rows = []

    for date, day in bars.groupby("date", sort=True):
        day = day.sort_values("time")
        if day["time"].tolist() != expected_day:
            continue
        first_hour = day.iloc[:60]
        if first_hour["time"].tolist() != first_hour_times:
            continue

        open_price = float(day.iloc[0]["open"])
        ema_available = bool(first_hour[["ema_30", "ema_50"]].notna().all().all())
        aligned_all_hour = bool(ema_available and (first_hour["ema_30"] > first_hour["ema_50"]).all())
        aligned_at_end = bool(first_hour.iloc[-1]["ema_30"] > first_hour.iloc[-1]["ema_50"])
        call_aligned_all_hour = bool(ema_available and (first_hour["ema_50"] > first_hour["ema_30"]).all())
        call_aligned_at_end = bool(first_hour.iloc[-1]["ema_50"] > first_hour.iloc[-1]["ema_30"])
        row = {
            "date": date,
            "earliest_entry_et": first_hour.iloc[-1]["known_at_et"],
            "opening_price_strike_exact": open_price,
            "strike_5pt_assumption": math.floor(open_price / 5) * 5,
            "call_strike_5pt_assumption": math.ceil(open_price / 5) * 5,
            "ema30_above_ema50_all_first_hour": aligned_all_hour,
            "ema30_above_ema50_at_1029": aligned_at_end,
            "ema50_above_ema30_all_first_hour": call_aligned_all_hour,
            "ema50_above_ema30_at_1029": call_aligned_at_end,
            "ema_available_all_first_hour": ema_available,
            "spx_1030_open": float(day.iloc[60]["open"]),
        }
        row["put_in_money_at_entry"] = row["spx_1030_open"] < open_price
        row["call_in_money_at_entry"] = row["spx_1030_open"] > open_price

        for dte, expiry in ((0, date), (1, next_day.get(date))):
            available = expiry in settlements.index if expiry else False
            row[f"expiry_{dte}dte"] = expiry if available else None
            row[f"settlement_{dte}dte"] = float(settlements.loc[expiry]) if available else None
            row[f"zero_{dte}dte"] = row[f"settlement_{dte}dte"] >= open_price if available else None
            row[f"zero_{dte}dte_5pt"] = (
                row[f"settlement_{dte}dte"] >= row["strike_5pt_assumption"] if available else None
            )
            row[f"zero_call_{dte}dte"] = row[f"settlement_{dte}dte"] <= open_price if available else None
            row[f"zero_call_{dte}dte_5pt"] = (
                row[f"settlement_{dte}dte"] <= row["call_strike_5pt_assumption"] if available else None
            )
        rows.append(row)

    return pd.DataFrame(rows)


def summarize(label: str, data: pd.DataFrame, option_type: str = "put") -> None:
    zero_prefix = "zero" if option_type == "put" else "zero_call"
    print(f"{label}: {len(data)} signal days")
    for dte in (0, 1):
        eligible = data[data[f"expiry_{dte}dte"].notna()]
        wins = int(eligible[f"{zero_prefix}_{dte}dte"].astype(int).sum())
        rate = wins / len(eligible) if len(eligible) else float("nan")
        print(f"  {dte}DTE: {wins}/{len(eligible)} = {rate:.1%}")
    paired = data[data["expiry_0dte"].notna() & data["expiry_1dte"].notna()]
    print(
        "  Paired: "
        f"0DTE {int(paired[f'{zero_prefix}_0dte'].astype(int).sum())}/{len(paired)}, "
        f"1DTE {int(paired[f'{zero_prefix}_1dte'].astype(int).sum())}/{len(paired)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minute-file", type=Path, default=Path("data/spx_1m_indicators.parquet"))
    parser.add_argument("--settlement-file", type=Path, default=Path("data/spxw_settlements_2026.csv"))
    parser.add_argument("--calendar-file", type=Path, help="Optional complete trading calendar")
    parser.add_argument("--output", type=Path, default=Path("results/first_hour_ema_open_put.csv"))
    args = parser.parse_args()

    result = run(args.minute_file, args.settlement_file, args.calendar_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Complete SPX sessions: {len(result)}")
    summarize("EMA30 > EMA50 throughout 09:30–10:29", result[result["ema30_above_ema50_all_first_hour"]])
    summarize("EMA30 > EMA50 at 10:29 only", result[result["ema30_above_ema50_at_1029"]])
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
