"""Check historical SPXW chain and minute-trade coverage for one 0DTE condor.

This probe does not calculate fill prices: UW contract intraday bars aggregate
executed trades, whereas a reproducible credit-spread backtest needs historical
simultaneous executable quotes for all legs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://api.unusualwhales.com"
OPTION_SYMBOL = re.compile(r"^(?P<root>[A-Z0-9]+)(?P<expiry>\d{6})(?P<side>[PC])(?P<strike>\d{8})$")
ET = ZoneInfo("America/New_York")


def api_key() -> str:
    key = os.environ.get("UW_API_KEY", "").strip()
    if key:
        return key
    dotenv = ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if line.startswith("UW_API_KEY="):
                return line.partition("=")[2].strip().strip('"\'')
    raise SystemExit("UW_API_KEY is missing. Run: .venv/bin/python scripts/setup_uw_key.py")


def get_json(path: str, key: str, query: dict[str, str] | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    if query:
        url += "?" + urlencode(query)
    request = Request(url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            detail = json.load(exc)
            reason = detail.get("reason") or detail.get("code") or "unknown"
        except (ValueError, AttributeError):
            reason = "unknown"
        raise RuntimeError(f"UW API returned HTTP {exc.code} ({reason}) for {path}") from None
    except URLError as exc:
        raise RuntimeError(f"UW API network error for {path}: {exc.reason}") from None


def signal_row(path: Path, date: str | None) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    selected = [row for row in rows if row["date"] == date] if date else rows
    if not selected:
        raise ValueError(f"No double-breakout signal for {date or 'the available dates'}")
    return selected[-1]


def parse_chain(symbols: list[str], date: str) -> dict[str, list[tuple[float, str]]]:
    expiry = datetime.strptime(date, "%Y-%m-%d").strftime("%y%m%d")
    by_side: dict[str, list[tuple[float, str]]] = {"C": [], "P": []}
    for symbol in symbols:
        match = OPTION_SYMBOL.fullmatch(symbol)
        if match and match["root"] == "SPXW" and match["expiry"] == expiry:
            by_side[match["side"]].append((int(match["strike"]) / 1000, symbol))
    for contracts in by_side.values():
        contracts.sort()
    return by_side


def select_legs(
    chain: dict[str, list[tuple[float, str]]], call_target: float, put_target: float, width: float
) -> dict[str, tuple[float, str]]:
    calls, puts = chain["C"], chain["P"]
    short_call = next(((strike, symbol) for strike, symbol in calls if strike >= call_target), None)
    short_put = next(((strike, symbol) for strike, symbol in reversed(puts) if strike <= put_target), None)
    if short_call is None or short_put is None:
        raise ValueError("The historical chain has no matching outward SPXW short strikes")
    if short_call[0] - call_target > 5 or put_target - short_put[0] > 5:
        raise ValueError("The closest listed short strike is over 5 points from its theoretical target")
    long_call = next(((strike, symbol) for strike, symbol in calls if strike >= short_call[0] + width), None)
    long_put = next(((strike, symbol) for strike, symbol in reversed(puts) if strike <= short_put[0] - width), None)
    if long_call is None or long_put is None:
        raise ValueError("The historical chain has no matching protective wings")
    return {"short_call": short_call, "long_call": long_call, "short_put": short_put, "long_put": long_put}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Qualifying date in YYYY-MM-DD; defaults to the latest")
    parser.add_argument("--width", type=float, default=5.0, help="Minimum distance to each long wing in SPX points")
    parser.add_argument("--signals", type=Path, default=ROOT / "results/double_breakout_first_hour.csv")
    args = parser.parse_args()
    if args.width <= 0:
        parser.error("--width must be positive")

    signal = signal_row(args.signals, args.date)
    date = signal["date"]
    entry = datetime.fromisoformat(signal["entry_et"]).astimezone(ET)
    key = api_key()
    chain_result = get_json("/api/stock/SPX/option-chains", key, {"date": date})
    symbols = chain_result.get("data")
    if not isinstance(symbols, list) or not all(isinstance(symbol, str) for symbol in symbols):
        raise RuntimeError("Unexpected option-chain response shape")
    chain = parse_chain(symbols, date)
    print(f"{date}: {len(chain['C'])} SPXW calls and {len(chain['P'])} SPXW puts for same-day PM expiry")
    legs = select_legs(
        chain, float(signal["short_call_strike_exact"]), float(signal["short_put_strike_exact"]), args.width
    )

    available_by_leg = {}
    for label, (strike, symbol) in legs.items():
        data = get_json(f"/api/option-contract/{quote(symbol, safe='')}/intraday", key, {"date": date}).get("data")
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected intraday response shape for {label}")
        near = []
        for bar in data:
            start = datetime.fromisoformat(bar["start_time"].replace("Z", "+00:00")).astimezone(ET)
            if entry <= start < entry + timedelta(minutes=10):
                near.append((start, bar))
        near.sort(key=lambda item: item[0])
        available_by_leg[label] = {start for start, _ in near}
        first = near[0][0].strftime("%H:%M ET") if near else "none in 10 minutes"
        print(f"  {label}: {symbol} (K={strike:g}), {len(data)} trade minutes; first after entry: {first}")

    shared = set.intersection(*available_by_leg.values())
    shared = sorted(t for t in shared if t < entry + timedelta(minutes=5))
    print(f"All four legs have a trade bar in the same 5-minute entry window: {'yes' if shared else 'no'}")
    if shared:
        print(f"First shared minute: {shared[0].strftime('%H:%M ET')}")
    print("Trade-minute OHLC is not an executable bid/ask quote; no spread P&L was inferred.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"Probe stopped: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
