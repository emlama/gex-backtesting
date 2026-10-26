"""Checks for capped multi-leg payoff, strict quote coverage, streak stats, and the dashboard."""

import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bulk_dashboard import write_dashboard  # noqa: E402
from run_bulk_spxw import evaluate, performance, quote_snapshot, summarize  # noqa: E402

CONFIG = {
    "widths": [5],
    "put_offsets": [0],
    "call_offsets": [0],
    "commission_per_contract_per_side": 0.65,
}
EVENT = {
    "signal": "test", "date": "2026-10-22", "entry_et": "2026-10-22 10:30:00-04:00",
    "structure": "iron_condor", "put_anchor": 7655.0, "call_anchor": 7705.0,
}


def quotes():
    prices = {
        ("put", 7655.0): (15.8, 15.9),
        ("put", 7650.0): (13.7, 13.8),
        ("call", 7705.0): (3.4, 3.5),
        ("call", 7710.0): (2.8, 2.85),
    }
    return {
        key: {"bid": bid, "ask": ask, "symbol": f"{key}", "trade_age_s": 1,
              "bid_age_s": 1, "ask_age_s": 1}
        for key, (bid, ask) in prices.items()
    }


def test_otm_condor_matches_independent_bid_ask_calculation():
    row = evaluate(EVENT, 0, "2026-10-22", 7663.43, quotes(), CONFIG)[0]
    assert row["status"] == "accepted"
    assert row["credit_points"] == 2.55
    assert row["expiry_debit_points"] == 0
    assert row["fees_dollars"] == 5.2
    assert row["pnl_dollars"] == 249.8
    assert row["max_loss_dollars"] == 250.2
    assert row["all_short_options_zero"] is True


def test_expiry_loss_is_capped_and_stale_leg_rejected():
    row = evaluate(EVENT, 0, "2026-10-22", 7800, quotes(), CONFIG)[0]
    assert row["expiry_debit_points"] == 5
    assert row["pnl_dollars"] == -250.2
    assert row["pnl_dollars"] == -row["max_loss_dollars"]
    without_long_call = quotes()
    del without_long_call[("call", 7710.0)]
    rejected = evaluate(EVENT, 0, "2026-10-22", 7663.43, without_long_call, CONFIG)[0]
    assert rejected["status"] == "rejected"
    assert rejected["reason"] == "missing_fresh_quote"


def test_snapshot_excludes_future_stale_canceled_and_wrong_expiry(tmp_path):
    entry = datetime(2026, 10, 22, 14, 30, tzinfo=timezone.utc)
    cutoff = entry + timedelta(seconds=3)

    def row(symbol, when, bid_time=None, canceled=False, expiry=date(2026, 10, 22), bid=1, ask=2):
        return {
            "option_symbol": symbol, "executed_at": when, "option_type": "put", "strike": 7655.0,
            "nbbo_bid": bid, "nbbo_ask": ask, "nbbo_bid_time": bid_time or when,
            "nbbo_ask_time": bid_time or when, "canceled": canceled, "expiry": expiry,
        }

    source = pd.DataFrame([
        row("valid", cutoff - timedelta(seconds=1)),
        row("future", cutoff + timedelta(microseconds=1)),
        row("stale", cutoff - timedelta(seconds=1), cutoff - timedelta(seconds=10)),
        row("canceled", cutoff - timedelta(seconds=1), canceled=True),
        row("wrong_expiry", cutoff - timedelta(seconds=1), expiry=date(2026, 10, 23)),
        row("crossed", cutoff - timedelta(seconds=1), bid=3, ask=2),
    ])
    path = tmp_path / "tape.parquet"
    source.to_parquet(path, index=False)
    snapshot = quote_snapshot(path, "2026-10-22", entry, delay=3, max_age=3)
    assert len(snapshot) == 1
    assert snapshot[("put", 7655.0)]["symbol"] == "valid"


def test_performance_tracks_drawdown_and_streaks():
    pnl = [100, -50, -30, 200, 0, -10]
    dates = [f"2026-10-0{day}" for day in range(1, 7)]
    stats = performance(pnl, dates)
    assert stats["max_drawdown_dollars"] == -80
    assert (stats["drawdown_peak_date"], stats["drawdown_trough_date"]) == ("2026-10-01", "2026-10-03")
    assert (stats["wins"], stats["losses"]) == (2, 3)
    assert (stats["max_consecutive_wins"], stats["max_consecutive_losses"]) == (1, 2)
    assert stats["largest_winning_streak_dollars"] == 200
    assert stats["largest_losing_streak_dollars"] == -80
    assert (stats["largest_win_dollars"], stats["largest_loss_dollars"]) == (200, -50)
    assert stats["profit_factor"] == 3.3333


def test_performance_without_losses_has_no_drawdown_or_profit_factor():
    stats = performance([10, 20], ["2026-10-01", "2026-10-02"])
    assert stats["max_drawdown_dollars"] == 0
    assert stats["drawdown_peak_date"] is None
    assert stats["profit_factor"] is None


def test_dashboard_embeds_parseable_escaped_data(tmp_path):
    rows = pd.DataFrame(evaluate(EVENT, 0, "2026-10-22", 7663.43, quotes(), CONFIG))
    rows.loc[0, "signal"] = "</script><b>x"
    coverage = pd.DataFrame([{"signal": "test", "date": "2026-10-22", "dte": 0, "expiry": None, "status": "ready"}])
    destination = tmp_path / "dashboard.html"
    write_dashboard(summarize(rows), rows, coverage, {"run": "test", "dtes": [0]}, destination)
    page = destination.read_text(encoding="utf-8")
    assert "</script><b>" not in page
    data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1))
    assert data["meta"]["accepted"] == 1
    assert data["trades"][0]["signal"] == "</script><b>x"
    assert data["trades"][0]["legs"][0] == {"side": "short", "type": "put", "strike": 7655.0, "bid": 15.8, "ask": 15.9}
    assert data["variants"][0]["max_consecutive_losses"] == 0


def test_dashboard_handles_run_without_trades(tmp_path):
    destination = tmp_path / "dashboard.html"
    write_dashboard(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"run": "empty", "dtes": [0]}, destination)
    assert '"trades": []' in destination.read_text(encoding="utf-8")
