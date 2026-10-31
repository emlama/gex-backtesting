"""Test a short call at the SPX open after EMA50 exceeds EMA30 all first hour.

The 09:30–10:29 ET one-minute bars must each have EMA50 > EMA30. This is
observable at 10:30 ET, the earliest hypothetical entry. The theoretical call
strike is the 09:30 SPX open. Test zero intrinsic value at official SPXW PM
settlement on the signal date (0DTE) and next trading date (1DTE).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backtest_first_hour_ema_open_put import run, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minute-file", type=Path, default=Path("data/spx_1m_indicators.parquet"))
    parser.add_argument("--settlement-file", type=Path, default=Path("data/spxw_settlements_2026.csv"))
    parser.add_argument("--calendar-file", type=Path, help="Optional complete trading calendar")
    parser.add_argument("--output", type=Path, default=Path("results/first_hour_ema_open_call.csv"))
    args = parser.parse_args()

    result = run(args.minute_file, args.settlement_file, args.calendar_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Complete SPX sessions: {len(result)}")
    summarize(
        "EMA50 > EMA30 throughout 09:30–10:29",
        result[result["ema50_above_ema30_all_first_hour"]],
        option_type="call",
    )
    summarize(
        "EMA50 > EMA30 at 10:29 only",
        result[result["ema50_above_ema30_at_1029"]],
        option_type="call",
    )
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
