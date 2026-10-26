"""Cache UW full-tape SPXW data for a range of locally known SPX sessions.

Each day is resumable; completed Parquet days are skipped. The initial UW ZIP
downloads can be large. This command runs locally without LLM involvement.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from probe_uw_history import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="First SPX session YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Last SPX session YYYY-MM-DD")
    parser.add_argument("--include-next-expiry", action="store_true", help="Cache both 0DTE and next-session expiry")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("start must be on or before end")
    bars = pd.read_parquet(ROOT / "data/spx_1m_indicators.parquet", columns=["date"])
    all_dates = sorted(bars["date"].unique())
    dates = [day for day in all_dates if args.start <= day <= args.end]
    if not dates:
        parser.error("No locally known SPX sessions in the requested range")
    if args.include_next_expiry and dates[-1] == all_dates[-1]:
        parser.error("Last selected session has no known next session; choose an earlier --end")
    script = Path(__file__).with_name("cache_uw_spxw_0dte.py")
    print(f"Caching {len(dates)} SPX sessions from {dates[0]} through {dates[-1]}", flush=True)
    for index, day in enumerate(dates, start=1):
        print(f"[{index}/{len(dates)}] {day}", flush=True)
        command = [sys.executable, str(script), day]
        if args.include_next_expiry:
            command.append("--include-next-expiry")
        if args.keep_zip:
            command.append("--keep-zip")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
