"""Run one 0DTE first-hour condor through OSBT using a cached UW tape day.

This is an integration pilot: the entry uses recent trade-time NBBO snapshots
with OSBT's slippage model, and expiry uses the official SPXW settlement in a
small Backtester subclass. It does not simulate an executable complex-order fill.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
from probe_uw_history import ROOT, signal_row

try:
    from osbt.backtest import Backtester
    from osbt.data_loader import OptionChain
    from osbt.strategies.base import Leg, OptionType, Position, PositionSide, Strategy
except ImportError as exc:
    raise SystemExit('Install OSBT with: .venv/bin/python -m pip install -e ".[osbt]"') from exc


class FixedCondor(Strategy):
    """OSBT strategy using the project's known first-hour breakout strikes."""

    def __init__(self, short_put: float, short_call: float, width: int):
        super().__init__(min_dte=0, max_dte=0)
        self.short_put = short_put
        self.short_call = short_call
        self.width = width

    def describe(self) -> str:
        return f"First-hour {self.width}-point condor: {self.short_put}P / {self.short_call}C"

    def build_position(self, chain: OptionChain, quantity: int = 1) -> Position | None:
        expiry = chain.quote_time.date().isoformat()
        by_strike = {quote.strike: quote for quote in chain.quotes if quote.expire_date == expiry}
        contracts = (
            (self.short_put - self.width, OptionType.PUT, PositionSide.LONG),
            (self.short_put, OptionType.PUT, PositionSide.SHORT),
            (self.short_call, OptionType.CALL, PositionSide.SHORT),
            (self.short_call + self.width, OptionType.CALL, PositionSide.LONG),
        )
        legs = []
        for strike, option_type, side in contracts:
            quote = by_strike.get(strike)
            if quote is None:
                return None
            bid, ask = (quote.p_bid, quote.p_ask) if option_type == OptionType.PUT else (quote.c_bid, quote.c_ask)
            if bid <= 0 or ask < bid:
                return None
            legs.append(Leg(option_type, side, strike, quantity, (bid + ask) / 2, expiry, 0))
        return Position(legs, expiry, chain.underlying, expiry)


class SettlementBacktester(Backtester):
    """Supply an official settlement chain at the expiry exit timestamp."""

    def __init__(self, *args, settlements: dict[str, float], **kwargs):
        super().__init__(*args, **kwargs)
        self.settlements = settlements

    def _load_chain(self, date_str, target_time, *args, **kwargs):
        if target_time == self.exit_time and date_str in self.settlements:
            return OptionChain(
                quote_time=datetime.combine(date.fromisoformat(date_str), self.exit_time),
                underlying=self.settlements[date_str],
                quotes=[],
            )
        return super()._load_chain(date_str, target_time, *args, **kwargs)


def underlying_at(entry: datetime) -> float:
    bars = pd.read_parquet(ROOT / "data/spx_1m_indicators.parquet", columns=["known_at_et", "close"])
    match = bars.loc[bars["known_at_et"] == entry, "close"]
    if len(match) != 1:
        raise ValueError(f"Expected one completed SPX minute at {entry}; found {len(match)}")
    return float(match.iloc[0])


def export_entry_chain(day: str, entry: datetime, cache: Path, output: Path, age_seconds: int) -> int:
    cutoff = entry.astimezone(timezone.utc) + timedelta(seconds=age_seconds)
    source = ds.dataset(cache, format="parquet")
    rows = source.to_table(
        columns=[
            "option_symbol", "executed_at", "expiry", "option_type", "strike", "nbbo_bid", "nbbo_ask",
            "nbbo_bid_time", "nbbo_ask_time", "canceled", "delta", "implied_volatility",
        ],
        filter=(ds.field("executed_at") >= cutoff - timedelta(seconds=age_seconds))
        & (ds.field("executed_at") <= cutoff),
    ).to_pandas()
    rows = rows[(rows["nbbo_bid"] > 0) & (rows["nbbo_ask"] >= rows["nbbo_bid"])]
    rows = rows[~rows["canceled"].fillna(False)]
    for field in ("nbbo_bid_time", "nbbo_ask_time"):
        fresh = rows[field].notna() & (rows[field] <= cutoff)
        fresh &= rows[field] >= cutoff - timedelta(seconds=age_seconds)
        rows = rows[fresh]
    rows = rows.sort_values("executed_at").drop_duplicates("option_symbol", keep="last")
    if rows.empty:
        raise ValueError(f"No fresh UW quotes at {entry}")

    spot = underlying_at(entry)
    chains = {}
    for row in rows.itertuples(index=False):
        key = (row.expiry.isoformat(), float(row.strike))
        record = chains.setdefault(key, {
            "quote_date": day,
            "quote_readtime": entry.strftime("%Y-%m-%d %H:%M"),
            "underlying_last": spot,
            "expire_date": row.expiry.isoformat(),
            "dte": 0.01,
            "strike": float(row.strike),
            "c_bid": 0.0, "c_ask": 0.0, "c_delta": 0.0, "c_iv": 0.0,
            "p_bid": 0.0, "p_ask": 0.0, "p_delta": 0.0, "p_iv": 0.0,
        })
        prefix = "c" if row.option_type == "call" else "p"
        record[f"{prefix}_bid"] = float(row.nbbo_bid)
        record[f"{prefix}_ask"] = float(row.nbbo_ask)
        record[f"{prefix}_delta"] = float(row.delta) if pd.notna(row.delta) else 0.0
        record[f"{prefix}_iv"] = float(row.implied_volatility) if pd.notna(row.implied_volatility) else 0.0

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(chains.values()).to_parquet(output, index=False)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-10-22")
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--quote-age-seconds", type=int, default=3)
    parser.add_argument("--commission-per-contract", type=float, default=0.65)
    args = parser.parse_args()
    if args.width <= 0 or args.quote_age_seconds <= 0 or args.commission_per_contract < 0:
        parser.error("width and quote age must be positive; commission cannot be negative")

    signal = signal_row(ROOT / "results/double_breakout_first_hour.csv", args.date)
    entry = datetime.fromisoformat(signal["entry_et"])
    if entry.time() != time(10, 30):
        raise ValueError("Pilot currently supports a 10:30 ET entry only")
    cache = ROOT / "data/uw/spxw_0dte" / f"{args.date}.parquet"
    if not cache.exists():
        raise FileNotFoundError(f"Cache this date first: {cache}")
    data_dir = ROOT / "data/uw/osbt_pilot" / args.date
    output = data_dir / "parquet" / f"spx_15x_{entry.year}" / f"spx_15x_{entry:%Y%m}.parquet"
    quote_count = export_entry_chain(args.date, entry, cache, output, args.quote_age_seconds)

    strategy = FixedCondor(
        float(signal["short_put_strike_5pt_assumption"]),
        float(signal["short_call_strike_5pt_assumption"]),
        args.width,
    )
    backtester = SettlementBacktester(
        data_dir=str(data_dir), entry_time=entry.time(), exit_time=time(15, 45),
        settlements={args.date: float(signal["spxw_settlement"])},
    )
    result = backtester.run(
        strategy, args.date, args.date, quantity=1, slippage_pct=100,
        commission_per_contract=args.commission_per_contract, verbose=False,
    )
    report = {
        "date": args.date,
        "source": "UW full-tape trade-time NBBO plus official SPXW settlement",
        "quote_cutoff_et": (entry + timedelta(seconds=args.quote_age_seconds)).isoformat(),
        "max_quote_age_seconds": args.quote_age_seconds,
        "valid_option_quotes": quote_count,
        "short_put": strategy.short_put,
        "short_call": strategy.short_call,
        "width": strategy.width,
        "settlement": float(signal["spxw_settlement"]),
        "commission_per_contract": args.commission_per_contract,
        "trades": len(result.trades),
    }
    if result.trades:
        trade = result.trades[0]
        report.update({
            "entry_credit_dollars": -trade.entry_cost,
            "expiry_debit_dollars": -trade.exit_value,
            "pnl_dollars": trade.pnl,
            "exit_reason": trade.exit_reason,
        })
    report_path = ROOT / "results" / f"osbt_uw_pilot_{args.date}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
