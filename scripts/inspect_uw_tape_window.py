"""Stream the beginning of a UW daily tape to inspect SPXW entry quotes.

The daily ZIP is a single CSV, but rows are not strictly sorted by executed_at.
This script scans until all four legs have post-entry prints or reaches a byte
cap, retaining only prints close to the entry time.
The resulting bid/ask values are trade-time NBBO snapshots, not guaranteed
simultaneous executable four-leg fills.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pyarrow.compute as pc
import pyarrow.dataset as ds
from probe_uw_history import ROOT, api_key, get_json, parse_chain, select_legs, signal_row


def timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00").replace(" ", "T")).astimezone(timezone.utc)


def cached_window(path: Path, symbols: set[str], entry: datetime) -> tuple[dict, dict]:
    entry_utc = entry.astimezone(timezone.utc)
    source = ds.dataset(path, format="parquet")
    selected = (
        ds.field("option_symbol").isin(list(symbols))
        & (ds.field("executed_at") >= entry_utc - timedelta(minutes=1))
        & (ds.field("executed_at") < entry_utc + timedelta(minutes=2))
    )
    table = source.to_table(
        columns=[
            "option_symbol", "executed_at", "nbbo_bid", "nbbo_ask", "nbbo_bid_time", "nbbo_ask_time", "canceled"
        ],
        filter=selected,
    )
    records = {symbol: [] for symbol in symbols}
    for row in table.to_pylist():
        records[row["option_symbol"]].append(row)
    return records, {"date": path.stem, "source": "local SPXW Parquet", "matched_rows": table.num_rows}


def tape_window(key: str, date: str, symbols: set[str], entry: datetime, max_bytes: int) -> tuple[dict, dict]:
    url = f"https://api.unusualwhales.com/api/option-trades/full-tape/{date}"
    request = Request(url, headers={"Authorization": f"Bearer {key}", "Range": f"bytes=0-{max_bytes - 1}"})
    try:
        response = urlopen(request, timeout=60)
    except HTTPError as exc:
        raise RuntimeError(f"Full-tape request returned HTTP {exc.code}") from None

    with response:
        if response.status != 206:
            raise RuntimeError("The server did not honor the partial-download request")
        first = response.read(4096)
        if first[:4] != b"PK\x03\x04":
            raise RuntimeError("The response did not start with a ZIP member")
        method = struct.unpack_from("<H", first, 8)[0]
        name_length, extra_length = struct.unpack_from("<HH", first, 26)
        if method != 8:
            raise RuntimeError(f"Expected a deflated ZIP member; got method {method}")
        start = 30 + name_length + extra_length
        inflater = zlib.decompressobj(-15)
        pending = b""
        columns = None
        found: dict[str, list[dict]] = {symbol: [] for symbol in symbols}
        found_after: set[str] = set()
        symbol_bytes = {symbol.encode("ascii"): symbol for symbol in symbols}
        entry_utc = entry.astimezone(timezone.utc)
        start_window = entry_utc - timedelta(minutes=1)
        cutoff = entry_utc + timedelta(minutes=2)
        start_label = start_window.strftime("%Y-%m-%d %H:%M:%S").encode()
        cutoff_label = cutoff.strftime("%Y-%m-%d %H:%M:%S").encode()
        compressed_bytes = 0
        scanned_rows = 0
        spxw_rows_in_window = 0
        sample_spxw_symbols: list[str] = []
        reached_cutoff = False
        last_time = ""

        def consume(block: bytes) -> None:
            nonlocal pending, columns, scanned_rows, spxw_rows_in_window, reached_cutoff, last_time
            plain = inflater.decompress(block)
            lines = (pending + plain).split(b"\n")
            pending = lines.pop()
            for line in lines:
                if not line:
                    continue
                if columns is None:
                    columns = next(csv.reader([line.decode("utf-8-sig")]))
                    continue
                scanned_rows += 1
                parts = line.split(b",", 3)
                if len(parts) < 4:
                    continue
                when = parts[2][:19]
                last_time = when.decode("ascii", errors="replace")
                if when >= cutoff_label:
                    reached_cutoff = True
                    continue
                if when < start_label:
                    continue
                if b",SPXW," in line:
                    spxw_rows_in_window += 1
                    if len(sample_spxw_symbols) < 5:
                        fields = line.split(b",", 8)
                        if len(fields) > 7:
                            sample_spxw_symbols.append(fields[7].decode("ascii", errors="replace"))
                for needle, symbol in symbol_bytes.items():
                    if needle in line:
                        values = next(csv.reader([line.decode("utf-8")]))
                        row = dict(zip(columns, values))
                        if row.get("option_chain_id") == symbol:
                            found[symbol].append(row)
                            if timestamp(row["executed_at"]) >= entry_utc:
                                found_after.add(symbol)
                        break

        first_block = first[start:]
        compressed_bytes += len(first_block)
        consume(first_block)
        while found_after != symbols:
            block = response.read(1 << 20)
            if not block:
                break
            compressed_bytes += len(block)
            consume(block)
            if compressed_bytes // (64 << 20) > (compressed_bytes - len(block)) // (64 << 20):
                print(
                    f"Scanned {compressed_bytes / (1 << 20):.0f} MiB of compressed tape; latest {last_time} UTC",
                    flush=True,
                )

    meta = {
        "date": date,
        "entry_utc": entry_utc.isoformat(),
        "compressed_bytes_read": compressed_bytes,
        "scanned_rows": scanned_rows,
        "spxw_rows_in_window": spxw_rows_in_window,
        "legs_with_post_entry_trades": len(found_after),
        "sample_spxw_symbols": sample_spxw_symbols,
        "reached_cutoff": reached_cutoff,
        "last_time_utc": last_time,
    }
    return found, meta


def summarize_quotes(records: dict[str, list[dict]], legs: dict, entry: datetime) -> tuple[dict, float | None]:
    entry_utc = entry.astimezone(timezone.utc)
    first_after = {}
    for label, (_, symbol) in legs.items():
        matches = sorted(
            ((timestamp(row["executed_at"]), row) for row in records[symbol] if row.get("executed_at")),
            key=lambda item: item[0],
        )
        first_after[label] = next((when for when, _ in matches if when >= entry_utc), None)
    if any(when is None for when in first_after.values()):
        return {"first_after_utc": {label: str(value) for label, value in first_after.items()}}, None

    asof = max(first_after.values())
    snapshots = {}
    credit = 0.0
    for label, (strike, symbol) in legs.items():
        side = "bid" if label.startswith("short") else "ask"
        price_field = f"nbbo_{side}"
        time_field = f"nbbo_{side}_time"
        candidates = []
        for row in records[symbol]:
            when = timestamp(row["executed_at"])
            canceled_value = row.get("canceled", False)
            if when > asof or canceled_value in (True, "t", "true", "1"):
                continue
            try:
                price = float(row[price_field])
            except (KeyError, TypeError, ValueError):
                continue
            if price <= 0:
                continue
            candidates.append((when, row, price))
        if not candidates:
            snapshots[label] = {"symbol": symbol, "strike": strike, "error": "no usable trade-time NBBO snapshot"}
            continue
        when, row, price = max(candidates, key=lambda item: item[0])
        quote_time = timestamp(row[time_field]) if row.get(time_field) else None
        quote_age = (asof - quote_time).total_seconds() if quote_time else None
        snapshots[label] = {
            "symbol": symbol,
            "strike": strike,
            "side": side,
            "nbbo_price": price,
            "trade_time_utc": when.isoformat(),
            "quote_time_utc": quote_time.isoformat() if quote_time else None,
            "trade_age_seconds": round((asof - when).total_seconds(), 3),
            "quote_age_seconds": round(quote_age, 3) if quote_age is not None else None,
        }
        credit += price if side == "bid" else -price

    report = {"asof_utc": asof.isoformat(), "legs": snapshots}
    complete = all("nbbo_price" in snapshot for snapshot in snapshots.values())
    report["observed_nbbo_credit"] = round(credit, 2) if complete else None
    fresh = all(
        "nbbo_price" in snapshot
        and snapshot["trade_age_seconds"] <= 3
        and snapshot["quote_age_seconds"] is not None
        and 0 <= snapshot["quote_age_seconds"] <= 3
        for snapshot in snapshots.values()
    )
    report["all_quotes_within_3s"] = fresh
    return report, round(credit, 2) if fresh and complete else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-10-22")
    parser.add_argument("--width", type=float, default=5.0)
    parser.add_argument("--max-compressed-mib", type=int, default=768)
    parser.add_argument("--force-api", action="store_true", help="Ignore a cached SPXW Parquet day")
    args = parser.parse_args()
    if args.width <= 0 or args.max_compressed_mib <= 0:
        parser.error("width and max-compressed-mib must be positive")

    signal = signal_row(ROOT / "results/double_breakout_first_hour.csv", args.date)
    entry = datetime.fromisoformat(signal["entry_et"])
    cache = ROOT / "data/uw/spxw_0dte" / f"{args.date}.parquet"
    if cache.exists() and not args.force_api:
        source = ds.dataset(cache, format="parquet")
        symbols = pc.unique(source.to_table(columns=["option_symbol"])["option_symbol"]).to_pylist()
        chain = parse_chain(symbols, args.date)
    else:
        key = api_key()
        chain_result = get_json("/api/stock/SPX/option-chains", key, {"date": args.date})
        chain = parse_chain(chain_result["data"], args.date)
    legs = select_legs(
        chain, float(signal["short_call_strike_exact"]), float(signal["short_put_strike_exact"]), args.width
    )
    leg_symbols = {symbol for _, symbol in legs.values()}
    if cache.exists() and not args.force_api:
        records, meta = cached_window(cache, leg_symbols, entry)
    else:
        records, meta = tape_window(key, args.date, leg_symbols, entry, args.max_compressed_mib << 20)
    quote_report, indicative_credit = summarize_quotes(records, legs, entry)
    report = {"metadata": meta, "quote_report": quote_report, "indicative_credit": indicative_credit}
    output = ROOT / "data/uw" / f"{args.date}_first_hour_{args.width:g}pt_snapshot.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    match_counts = {k: len(v) for k, v in records.items()}
    print(f"Data source: {meta.get('source', 'partial UW full tape')}; matching records: {match_counts}")
    if "reached_cutoff" in meta:
        print(f"Reached 10:32 ET: {meta['reached_cutoff']}; SPXW rows: {meta['spxw_rows_in_window']}")
    print(f"Indicative four-leg credit from snapshots: {indicative_credit}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, KeyError, ValueError) as exc:
        print(f"Tape inspection stopped: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
