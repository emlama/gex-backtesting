"""Download a UW full-tape ZIP once and cache SPXW 0DTE or 0DTE+1DTE trades.

The source ZIP is transient. The resulting typed, date-partitioned Parquet file
is reusable by many backtests without another API request or ZIP scan.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq
from probe_uw_history import ROOT, api_key

SCHEMA = pa.schema(
    [
        ("trade_date", pa.date32()),
        ("option_symbol", pa.string()),
        ("executed_at", pa.timestamp("us", tz="UTC")),
        ("created_at", pa.timestamp("us", tz="UTC")),
        ("expiry", pa.date32()),
        ("option_type", pa.string()),
        ("strike", pa.float64()),
        ("price", pa.float64()),
        ("size", pa.int64()),
        ("nbbo_bid", pa.float64()),
        ("nbbo_ask", pa.float64()),
        ("nbbo_bid_time", pa.timestamp("us", tz="UTC")),
        ("nbbo_ask_time", pa.timestamp("us", tz="UTC")),
        ("nbbo_bid_size", pa.int64()),
        ("nbbo_ask_size", pa.int64()),
        ("canceled", pa.bool_()),
        ("report_flags", pa.string()),
        ("tags", pa.string()),
        ("trading_hours", pa.string()),
        ("exchange", pa.string()),
        ("implied_volatility", pa.float64()),
        ("delta", pa.float64()),
        ("open_interest", pa.int64()),
        ("volume", pa.int64()),
    ]
)


def number(value: str, kind: type = float) -> float | int | None:
    return kind(value) if value else None


def utc_time(value: str) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00").replace(" ", "T")).astimezone(timezone.utc)


def canceled(value: str) -> bool | None:
    value = value.lower()
    if value in ("f", "false", "0"):
        return False
    if value in ("t", "true", "1"):
        return True
    return None


def download(date_text: str, destination: Path, key: str) -> int:
    if destination.exists():
        print(f"Using existing ZIP: {destination}")
        return destination.stat().st_size
    partial = destination.with_suffix(".zip.part")
    prior = partial.stat().st_size if partial.exists() else 0
    headers = {"Authorization": f"Bearer {key}"}
    if prior:
        headers["Range"] = f"bytes={prior}-"
    request = Request(f"https://api.unusualwhales.com/api/option-trades/full-tape/{date_text}", headers=headers)
    try:
        response = urlopen(request, timeout=60)
    except HTTPError as exc:
        raise RuntimeError(f"UW full-tape download returned HTTP {exc.code} for {date_text}") from None
    except URLError as exc:
        raise RuntimeError(f"UW full-tape download failed: {exc.reason}") from None
    with response:
        if prior and response.status != 206:
            raise RuntimeError("UW did not honor the resume request; remove the partial ZIP and retry")
        if not prior and response.status != 200:
            raise RuntimeError(f"Unexpected UW response status {response.status}")
        expected = None
        if response.status == 206:
            match = re.search(r"/(\d+)$", response.headers.get("Content-Range", ""))
            expected = int(match.group(1)) if match else None
        elif response.headers.get("Content-Length"):
            expected = int(response.headers["Content-Length"])
        mode = "ab" if prior else "wb"
        written = prior
        last_report = prior
        with partial.open(mode) as file:
            while block := response.read(8 << 20):
                file.write(block)
                written += len(block)
                if written - last_report >= 256 << 20:
                    print(f"Downloaded {written / (1 << 20):.0f} MiB of {date_text} tape", flush=True)
                    last_report = written
    if expected is not None and written != expected:
        raise RuntimeError(f"Incomplete ZIP: {written} of {expected} bytes; partial download retained")
    os.replace(partial, destination)
    return written


def make_record(row: list[str], index: dict[str, int], day: date) -> dict:
    def value(name: str) -> str:
        column = index.get(name)
        return row[column] if column is not None and column < len(row) else ""

    return {
        "trade_date": day,
        "option_symbol": value("option_chain_id"),
        "executed_at": utc_time(value("executed_at")),
        "created_at": utc_time(value("created_at")),
        "expiry": date.fromisoformat(value("expiry")),
        "option_type": value("option_type"),
        "strike": number(value("strike")),
        "price": number(value("price")),
        "size": number(value("size"), int),
        "nbbo_bid": number(value("nbbo_bid")),
        "nbbo_ask": number(value("nbbo_ask")),
        "nbbo_bid_time": utc_time(value("nbbo_bid_time")),
        "nbbo_ask_time": utc_time(value("nbbo_ask_time")),
        "nbbo_bid_size": number(value("nbbo_bid_size"), int),
        "nbbo_ask_size": number(value("nbbo_ask_size"), int),
        "canceled": canceled(value("canceled")),
        "report_flags": value("report_flags"),
        "tags": value("tags"),
        "trading_hours": value("trading_hours"),
        "exchange": value("exchange"),
        "implied_volatility": number(value("implied_volatility")),
        "delta": number(value("delta")),
        "open_interest": number(value("open_interest"), int),
        "volume": number(value("volume"), int),
    }


def convert(zip_path: Path, output: Path, day: date, expiries: set[date]) -> tuple[int, int]:
    temp = output.with_suffix(".parquet.part")
    total = 0
    selected = 0
    batch: list[dict] = []
    writer = None
    prefixes = tuple(f"SPXW{expiry:%y%m%d}" for expiry in expiries)
    expiry_texts = {expiry.isoformat() for expiry in expiries}
    try:
        with zipfile.ZipFile(zip_path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(names) != 1:
                raise RuntimeError(f"Expected one CSV member; got {len(names)}")
            with archive.open(names[0]) as binary:
                reader = csv.reader(io.TextIOWrapper(binary, encoding="utf-8", newline=""))
                columns = next(reader)
                index = {name: position for position, name in enumerate(columns)}
                required = {"underlying_symbol", "option_chain_id", "expiry", "executed_at", "nbbo_bid", "nbbo_ask"}
                if not required.issubset(index):
                    raise RuntimeError(f"UW CSV lacks columns: {sorted(required - index.keys())}")
                for row in reader:
                    total += 1
                    if total % 2_000_000 == 0:
                        print(f"Parsed {total:,} market trades; kept {selected:,} SPXW rows", flush=True)
                    if row[index["underlying_symbol"]] != "SPXW":
                        continue
                    if row[index["expiry"]] not in expiry_texts:
                        continue
                    if not row[index["option_chain_id"]].startswith(prefixes):
                        continue
                    batch.append(make_record(row, index, day))
                    selected += 1
                    if len(batch) >= 50_000:
                        if writer is None:
                            writer = pq.ParquetWriter(temp, SCHEMA, compression="zstd", use_dictionary=True)
                        writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                        batch.clear()
        if batch:
            if writer is None:
                writer = pq.ParquetWriter(temp, SCHEMA, compression="zstd", use_dictionary=True)
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
        if writer is None:
            raise RuntimeError("No SPXW 0DTE trades found")
    finally:
        if writer is not None:
            writer.close()
    if pq.read_metadata(temp).num_rows != selected:
        raise RuntimeError("Parquet row count did not match selected CSV rows")
    os.replace(temp, output)
    return total, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", help="Trading date in YYYY-MM-DD")
    parser.add_argument("--keep-zip", action="store_true", help="Keep the full-market ZIP after conversion")
    parser.add_argument("--force", action="store_true", help="Rebuild an existing Parquet day")
    parser.add_argument("--include-next-expiry", action="store_true", help="Include next SPX trading-session expiry")
    parser.add_argument("--next-expiry", help="Explicit next expiry YYYY-MM-DD (implies --include-next-expiry)")
    args = parser.parse_args()
    day = date.fromisoformat(args.date)
    next_expiry = date.fromisoformat(args.next_expiry) if args.next_expiry else None
    if args.include_next_expiry and next_expiry is None:
        import pandas as pd

        sessions = pd.read_parquet(ROOT / "data/spx_1m_indicators.parquet", columns=["date"])
        session_dates = sorted(sessions["date"].unique())
        later = [date.fromisoformat(value) for value in session_dates if value > day.isoformat()]
        if not later:
            raise ValueError(f"No next SPX session after {day}; use --next-expiry explicitly")
        next_expiry = later[0]
    if next_expiry is not None and next_expiry <= day:
        raise ValueError("Next expiry must be after the tape date")
    expiries = {day} | ({next_expiry} if next_expiry else set())
    lake = ROOT / ("data/uw/spxw_0_1dte" if next_expiry else "data/uw/spxw_0dte")
    lake.mkdir(parents=True, exist_ok=True)
    output = lake / f"{day.isoformat()}.parquet"
    if output.exists() and not args.force:
        print(f"Already cached: {output}")
        return
    zip_path = ROOT / "data/uw/full_tape" / f"{day.isoformat()}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    compressed_bytes = download(day.isoformat(), zip_path, api_key())
    total, selected = convert(zip_path, output, day, expiries)
    if not args.keep_zip:
        zip_path.unlink()
    meta = {
        "date": day.isoformat(),
        "source": "Unusual Whales full-tape ZIP",
        "expiries": sorted(expiry.isoformat() for expiry in expiries),
        "market_trade_rows": total,
        "spxw_0dte_rows": selected,
        "source_zip_bytes": compressed_bytes,
        "parquet_bytes": output.stat().st_size,
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Cached {selected:,} SPXW trades for {len(expiries)} expiry date(s) from {total:,} market trades")
    print(f"Parquet: {output} ({output.stat().st_size / (1 << 20):.1f} MiB)")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Cache build stopped: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
