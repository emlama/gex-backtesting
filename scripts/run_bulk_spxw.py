"""Run local, repeatable SPXW multi-leg 0DTE/1DTE parameter sweeps.

Reads saved price signals, UW Parquet tape caches, and Cboe settlements.
Makes no network requests and no LLM calls. Outputs every evaluated candidate,
summary statistics, coverage counts, and an offline HTML dashboard.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
from bulk_dashboard import write_dashboard
from probe_uw_history import ROOT


def config_from(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"double_breakout", "ema_first_hour"}
    if not config["signals"] or set(config["signals"]) - allowed:
        raise ValueError(f"signals must be chosen from {sorted(allowed)}")
    if not config["widths"] or any(width <= 0 for width in config["widths"]):
        raise ValueError("widths must be positive")
    if not config["dtes"] or set(config["dtes"]) - {0, 1}:
        raise ValueError("dtes must contain only 0 and/or 1")
    if config["quote_delay_seconds"] < 0 or config["max_quote_age_seconds"] <= 0:
        raise ValueError("quote delay must be nonnegative and max quote age positive")
    if config["commission_per_contract_per_side"] < 0:
        raise ValueError("commission must be nonnegative")
    start = config.get("start_date")
    end = config.get("end_date")
    if start:
        date.fromisoformat(start)
    if end:
        date.fromisoformat(end)
    if start and end and start > end:
        raise ValueError("start_date must be on or before end_date")
    if "ema_first_hour" in config["signals"]:
        available = {9, 20, 30, 50, 200}
        pairs = config.get("ema_pairs", [[30, 50]])
        if not pairs or any(len(pair) != 2 or set(pair) - available or pair[0] >= pair[1] for pair in pairs):
            raise ValueError("ema_pairs must contain ascending pairs from 9, 20, 30, 50, 200")
    return config


def signal_events(config: dict) -> list[dict]:
    events = []
    pairs = config.get("ema_pairs", [[30, 50]]) if "ema_first_hour" in config["signals"] else []
    columns = ["date", "time", "open", "high", "low", "close", "known_at_et"]
    columns += sorted({f"ema_{n}" for pair in pairs for n in pair})
    bars = pd.read_parquet(ROOT / "data/spx_1m_indicators.parquet", columns=columns)
    expected_day = pd.date_range("09:30", "15:59", freq="min").strftime("%H:%M").tolist()
    expected = expected_day[:60]
    for day, group in bars.groupby("date", sort=True):
        group = group.sort_values("time")
        if group["time"].tolist() != expected_day:
            continue
        first_hour = group.iloc[:60]
        if first_hour["time"].tolist() != expected:
            continue
        opening = float(first_hour.iloc[0]["open"])
        entry_et = str(first_hour.iloc[-1]["known_at_et"])
        if "double_breakout" in config["signals"]:
            range_high = float(first_hour.iloc[:30]["high"].max())
            range_low = float(first_hour.iloc[:30]["low"].min())
            second_half = first_hour.iloc[30:60]
            if (second_half["close"] > range_high).any() and (second_half["close"] < range_low).any():
                events.append({
                    "signal": "double_breakout", "date": day, "entry_et": entry_et,
                    "structure": "iron_condor",
                    "put_anchor": float(math.floor(float(first_hour["low"].min()) / 5) * 5),
                    "call_anchor": float(math.ceil(float(first_hour["high"].max()) / 5) * 5),
                })
        if "ema_first_hour" in config["signals"]:
            for fast, slow in pairs:
                fast_values = first_hour[f"ema_{fast}"]
                slow_values = first_hour[f"ema_{slow}"]
                if fast_values.isna().any() or slow_values.isna().any():
                    continue
                if (fast_values > slow_values).all():
                    events.append({
                        "signal": f"ema_{fast}_{slow}_put", "date": day, "entry_et": entry_et,
                        "structure": "put_vertical", "put_anchor": float(math.floor(opening / 5) * 5),
                        "call_anchor": None,
                    })
                elif (slow_values > fast_values).all():
                    events.append({
                        "signal": f"ema_{fast}_{slow}_call", "date": day, "entry_et": entry_et,
                        "structure": "call_vertical", "put_anchor": None,
                        "call_anchor": float(math.ceil(opening / 5) * 5),
                    })
    return sorted(events, key=lambda row: (row["date"], row["signal"]))


def expiry_lookup() -> tuple[dict[str, str], dict[str, float]]:
    bars = pd.read_parquet(ROOT / "data/spx_1m_indicators.parquet", columns=["date"])
    days = sorted(bars["date"].unique())
    next_session = dict(zip(days, days[1:]))
    settlement_files = sorted((ROOT / "data").glob("spxw_settlements_*.csv"))
    if not settlement_files:
        raise FileNotFoundError("No Cboe SPXW settlement CSV found under data/")
    settlements_frame = pd.concat((pd.read_csv(path) for path in settlement_files), ignore_index=True)
    if settlements_frame["date"].duplicated().any():
        raise ValueError("Duplicate SPXW settlement dates across source files")
    settlements = dict(zip(settlements_frame["date"], settlements_frame["settlement"]))
    return next_session, settlements


def cache_path(day: str, dte: int) -> Path | None:
    combined = ROOT / "data/uw/spxw_0_1dte" / f"{day}.parquet"
    if combined.exists():
        return combined
    if dte == 0:
        single = ROOT / "data/uw/spxw_0dte" / f"{day}.parquet"
        if single.exists():
            return single
    return None


def quote_snapshot(path: Path, expiry: str, entry: datetime, delay: int, max_age: int) -> dict:
    cutoff = entry.astimezone(timezone.utc) + timedelta(seconds=delay)
    start = cutoff - timedelta(seconds=max_age)
    source = ds.dataset(path, format="parquet")
    rows = source.to_table(
        columns=[
            "option_symbol", "executed_at", "option_type", "strike", "nbbo_bid", "nbbo_ask",
            "nbbo_bid_time", "nbbo_ask_time", "canceled",
        ],
        filter=(ds.field("expiry") == date.fromisoformat(expiry))
        & (ds.field("executed_at") >= start)
        & (ds.field("executed_at") <= cutoff),
    ).to_pandas()
    if rows.empty:
        return {}
    rows = rows[(rows["nbbo_bid"] > 0) & (rows["nbbo_ask"] >= rows["nbbo_bid"])]
    rows = rows[~rows["canceled"].fillna(False)]
    for field in ("nbbo_bid_time", "nbbo_ask_time"):
        rows = rows[rows[field].notna() & (rows[field] >= start) & (rows[field] <= cutoff)]
    rows = rows.sort_values("executed_at").drop_duplicates("option_symbol", keep="last")
    return {
        (row.option_type, float(row.strike)): {
            "symbol": row.option_symbol,
            "bid": float(row.nbbo_bid),
            "ask": float(row.nbbo_ask),
            "trade_age_s": (cutoff - row.executed_at).total_seconds(),
            "bid_age_s": (cutoff - row.nbbo_bid_time).total_seconds(),
            "ask_age_s": (cutoff - row.nbbo_ask_time).total_seconds(),
        }
        for row in rows.itertuples(index=False)
    }


def candidate_specs(event: dict, config: dict):
    widths = config["widths"]
    if event["structure"] == "put_vertical":
        for width, put_offset in itertools.product(widths, config["put_offsets"]):
            yield width, put_offset, None
    elif event["structure"] == "call_vertical":
        for width, call_offset in itertools.product(widths, config["call_offsets"]):
            yield width, None, call_offset
    else:
        for width, put_offset, call_offset in itertools.product(
            widths, config["put_offsets"], config["call_offsets"]
        ):
            yield width, put_offset, call_offset


def intrinsic(option_type: str, strike: float, settlement: float) -> float:
    return max(strike - settlement, 0.0) if option_type == "put" else max(settlement - strike, 0.0)


def evaluate(event: dict, dte: int, expiry: str, settlement: float, quotes: dict, config: dict) -> list[dict]:
    output = []
    fee = config["commission_per_contract_per_side"]
    for width, put_offset, call_offset in candidate_specs(event, config):
        put_strike = event["put_anchor"] + put_offset if put_offset is not None else None
        call_strike = event["call_anchor"] + call_offset if call_offset is not None else None
        result = {
            "signal": event["signal"], "date": event["date"], "entry_et": event["entry_et"],
            "structure": event["structure"], "dte": dte, "expiry": expiry, "settlement": settlement,
            "width": width, "put_offset": put_offset, "call_offset": call_offset,
            "short_put": put_strike, "short_call": call_strike,
        }
        if put_strike is not None and call_strike is not None and put_strike >= call_strike:
            result.update(status="rejected", reason="overlapping_short_strikes")
            output.append(result)
            continue
        legs = []
        if put_strike is not None:
            legs.extend([("put", put_strike, "short"), ("put", put_strike - width, "long")])
        if call_strike is not None:
            legs.extend([("call", call_strike, "short"), ("call", call_strike + width, "long")])
        missing = [f"{kind}:{strike:g}" for kind, strike, _ in legs if (kind, strike) not in quotes]
        if missing:
            result.update(status="rejected", reason="missing_fresh_quote", missing_legs=";".join(missing))
            output.append(result)
            continue
        credit = sum(
            quotes[(kind, strike)]["bid"] if side == "short" else -quotes[(kind, strike)]["ask"]
            for kind, strike, side in legs
        )
        if credit <= 0 or credit >= width:
            result.update(status="rejected", reason="invalid_credit", credit_points=round(credit, 4))
            output.append(result)
            continue
        debit = sum(
            intrinsic(kind, strike, settlement) * (1 if side == "short" else -1)
            for kind, strike, side in legs
        )
        fees = len(legs) * 2 * fee
        pnl = (credit - debit) * 100 - fees
        max_loss = (width - credit) * 100 + fees
        result.update({
            "status": "accepted", "reason": "", "credit_points": round(credit, 4),
            "expiry_debit_points": round(debit, 4), "fees_dollars": round(fees, 2),
            "pnl_dollars": round(pnl, 2), "max_loss_dollars": round(max_loss, 2),
            "return_on_risk": round(pnl / max_loss, 6), "profitable": pnl > 0,
            "all_short_options_zero": all(
                intrinsic(kind, strike, settlement) == 0
                for kind, strike, side in legs if side == "short"
            ),
            "max_trade_age_s": round(max(quotes[(kind, strike)]["trade_age_s"] for kind, strike, _ in legs), 3),
            "max_quote_age_s": round(max(
                max(quotes[(kind, strike)]["bid_age_s"], quotes[(kind, strike)]["ask_age_s"])
                for kind, strike, _ in legs
            ), 3),
            "legs": ";".join(
                f"{side}:{kind}:{strike:g}:{quotes[(kind, strike)]['symbol']}" for kind, strike, side in legs
            ),
            "leg_quotes_json": json.dumps([
                {"side": side, "type": kind, "strike": strike, **quotes[(kind, strike)]}
                for kind, strike, side in legs
            ]),
        })
        output.append(result)
    return output


def performance(pnl: list[float], dates: list[str]) -> dict:
    """Drawdown and streak statistics for one variant's date-ordered trades.

    Drawdown is measured from the running equity peak, starting at zero.
    A zero-P&L trade ends both winning and losing streaks.
    """
    equity = peak = 0.0
    peak_date = None
    drawdown = 0.0
    drawdown_peak = drawdown_trough = None
    wins = losses = 0
    win_streak = loss_streak = max_wins = max_losses = 0
    win_run = loss_run = best_run = worst_run = 0.0
    for value, day in zip(pnl, dates):
        equity += value
        if equity > peak:
            peak, peak_date = equity, day
        if equity - peak < drawdown:
            drawdown, drawdown_peak, drawdown_trough = equity - peak, peak_date, day
        if value > 0:
            wins += 1
            win_streak, loss_streak = win_streak + 1, 0
            win_run, loss_run = win_run + value, 0.0
        elif value < 0:
            losses += 1
            win_streak, loss_streak = 0, loss_streak + 1
            win_run, loss_run = 0.0, loss_run + value
        else:
            win_streak = loss_streak = 0
            win_run = loss_run = 0.0
        max_wins, max_losses = max(max_wins, win_streak), max(max_losses, loss_streak)
        best_run, worst_run = max(best_run, win_run), min(worst_run, loss_run)
    gross_win = sum(value for value in pnl if value > 0)
    gross_loss = -sum(value for value in pnl if value < 0)
    return {
        "wins": wins,
        "losses": losses,
        "max_drawdown_dollars": round(drawdown, 2),
        "drawdown_peak_date": drawdown_peak,
        "drawdown_trough_date": drawdown_trough,
        "max_consecutive_wins": max_wins,
        "max_consecutive_losses": max_losses,
        "largest_winning_streak_dollars": round(best_run, 2),
        "largest_losing_streak_dollars": round(worst_run, 2),
        "largest_win_dollars": round(max([value for value in pnl if value > 0], default=0.0), 2),
        "largest_loss_dollars": round(min([value for value in pnl if value < 0], default=0.0), 2),
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss else None,
    }


def summarize(all_rows: pd.DataFrame) -> pd.DataFrame:
    prepared = all_rows.copy()
    prepared["put_offset"] = prepared["put_offset"].fillna(0)
    prepared["call_offset"] = prepared["call_offset"].fillna(0)
    accepted = prepared[prepared["status"] == "accepted"].copy()
    if accepted.empty:
        return pd.DataFrame()
    keys = ["signal", "structure", "dte", "width", "put_offset", "call_offset"]
    attempted = prepared.groupby(keys, sort=True).size()
    summaries = []
    for key, group in accepted.groupby(keys, sort=True):
        group = group.sort_values("date")
        summaries.append(dict(zip(keys, key)) | {
            "trades": len(group),
            "attempted_days_with_tape": int(attempted.loc[key]),
            "quote_coverage_rate": round(len(group) / int(attempted.loc[key]), 4),
            "profitable_rate": round(float(group["profitable"].mean()), 4),
            "shorts_zero_rate": round(float(group["all_short_options_zero"].mean()), 4),
            "average_pnl_dollars": round(float(group["pnl_dollars"].mean()), 2),
            "total_pnl_dollars": round(float(group["pnl_dollars"].sum()), 2),
            "average_credit_points": round(float(group["credit_points"].mean()), 4),
            "average_max_loss_dollars": round(float(group["max_loss_dollars"].mean()), 2),
        } | performance(group["pnl_dollars"].tolist(), group["date"].tolist()))
    return pd.DataFrame(summaries).sort_values(["trades", "total_pnl_dollars"], ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/spxw_bulk.json")
    args = parser.parse_args()
    config = config_from(args.config)
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    events = [
        event for event in signal_events(config)
        if (not config.get("start_date") or event["date"] >= config["start_date"])
        and (not config.get("end_date") or event["date"] <= config["end_date"])
    ]
    next_session, settlements = expiry_lookup()
    results = []
    coverage = []
    for event in events:
        entry = datetime.fromisoformat(event["entry_et"])
        for dte in config["dtes"]:
            expiry = event["date"] if dte == 0 else next_session.get(event["date"])
            source = cache_path(event["date"], dte)
            status = "ready"
            if not expiry or expiry not in settlements:
                status = "no_settlement"
            elif source is None:
                status = "missing_uw_cache"
            coverage.append({
                "signal": event["signal"], "date": event["date"], "dte": dte,
                "expiry": expiry, "status": status,
            })
            if status != "ready":
                continue
            quotes = quote_snapshot(
                source, expiry, entry, config["quote_delay_seconds"], config["max_quote_age_seconds"]
            )
            results.extend(evaluate(event, dte, expiry, float(settlements[expiry]), quotes, config))
    rows = pd.DataFrame(results)
    coverage_frame = pd.DataFrame(coverage, columns=["signal", "date", "dte", "expiry", "status"])
    summary = summarize(rows) if not rows.empty else pd.DataFrame()
    candidates_path = output_dir / "candidates.parquet"
    if not rows.empty:
        rows.to_parquet(candidates_path, index=False)
    else:
        candidates_path.unlink(missing_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    coverage_frame.to_csv(output_dir / "coverage.csv", index=False)
    generated_utc = datetime.now(timezone.utc).isoformat()
    write_dashboard(summary, rows, coverage_frame, {
        "run": output_dir.name, "generated_utc": generated_utc, "dtes": config["dtes"],
        "quote_delay_seconds": config["quote_delay_seconds"],
        "max_quote_age_seconds": config["max_quote_age_seconds"],
        "commission_per_contract_per_side": config["commission_per_contract_per_side"],
    }, output_dir / "dashboard.html")
    accepted = 0 if rows.empty else int((rows["status"] == "accepted").sum())
    manifest = {
        "generated_utc": generated_utc,
        "config": config,
        "signal_events": len(events),
        "covered_signal_expiries": int((coverage_frame["status"] == "ready").sum()),
        "candidate_rows": len(rows),
        "accepted_candidates": accepted,
        "variants_with_trades": len(summary),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Signals: {len(events)}; candidate rows: {len(rows)}; accepted: {accepted}; variants: {len(summary)}")
    print(f"Coverage: {coverage_frame['status'].value_counts().to_dict()}")
    print(f"Dashboard: {output_dir / 'dashboard.html'}")


if __name__ == "__main__":
    main()
