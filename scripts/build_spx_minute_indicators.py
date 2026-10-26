"""Build SPX minute indicators from the public spx-0dte-archive OHLC files.

Usage:
    python scripts/build_spx_minute_indicators.py \
        --source-dir /path/to/spx-0dte-archive/spx-1m \
        --output data/spx_1m_indicators.parquet

Input times are New York local bar-start times. The resulting indicator values
are available only after the one-minute bar closes (``known_at_et``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def sma_seeded_ema(values: pd.Series, period: int) -> pd.Series:
    """EMA seeded with the first period's simple average."""
    data = values.to_numpy(dtype=float)
    result = np.full(len(data), np.nan)
    if len(data) >= period:
        result[period - 1] = data[:period].mean()
        alpha = 2 / (period + 1)
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return pd.Series(result, index=values.index)


def load_bars(source_dir: Path) -> pd.DataFrame:
    files = sorted(source_dir.glob("????-??-??.csv"))
    if not files:
        raise ValueError(f"No dated CSV files found in {source_dir}")

    frames = []
    required = {"time", "open", "high", "low", "close"}
    for file in files:
        frame = pd.read_csv(file)
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing OHLC columns in {file}")
        frame = frame.loc[(frame["time"] >= "09:30") & (frame["time"] < "16:00"), list(required)].copy()
        frame["date"] = file.stem
        frames.append(frame)

    bars = pd.concat(frames, ignore_index=True)
    for column in ("open", "high", "low", "close"):
        bars[column] = pd.to_numeric(bars[column], errors="raise")
    if bars[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("OHLC data has nulls")
    if (bars[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC data has nonpositive prices")
    if (bars["high"] < bars[["open", "low", "close"]].max(axis=1)).any() or (
        bars["low"] > bars[["open", "high", "close"]].min(axis=1)
    ).any():
        raise ValueError("OHLC prices are inconsistent")

    bars["bar_start_et"] = pd.to_datetime(bars["date"] + " " + bars["time"]).dt.tz_localize(
        "America/New_York"
    )
    bars = bars.sort_values("bar_start_et").reset_index(drop=True)
    if bars["bar_start_et"].duplicated().any():
        raise ValueError("Duplicate minute bars found")
    bars["known_at_et"] = bars["bar_start_et"] + pd.Timedelta(minutes=1)
    return bars[["bar_start_et", "known_at_et", "date", "time", "open", "high", "low", "close"]]


def add_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()
    close = bars["close"]
    for period in (9, 20, 30, 50, 200):
        bars[f"ema_{period}"] = sma_seeded_ema(close, period)

    mean = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    bars["bb_mid_20"] = mean
    bars["bb_upper_20_2"] = mean + 2 * std
    bars["bb_lower_20_2"] = mean - 2 * std
    bars["bb_width_pct_20_2"] = (bars["bb_upper_20_2"] - bars["bb_lower_20_2"]) / mean * 100

    macd = sma_seeded_ema(close, 12) - sma_seeded_ema(close, 26)
    bars["macd_12_26"] = macd
    valid_macd = macd.dropna()
    bars["macd_signal_9"] = sma_seeded_ema(valid_macd, 9).reindex(bars.index)
    bars["macd_hist_12_26_9"] = bars["macd_12_26"] - bars["macd_signal_9"]
    return bars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="spx-1m/ directory from spx-0dte-archive")
    parser.add_argument("--output", type=Path, default=Path("data/spx_1m_indicators.parquet"))
    parser.add_argument("--csv-output", type=Path, default=Path("data/spx_1m_indicators.csv"))
    args = parser.parse_args()

    bars = add_indicators(load_bars(args.source_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(args.output, index=False)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(args.csv_output, index=False)

    expected = pd.date_range("09:30", "15:59", freq="min").strftime("%H:%M")
    missing = sum(len(set(expected) - set(group["time"])) for _, group in bars.groupby("date"))
    print(f"Wrote {len(bars):,} bars from {bars['date'].min()} to {bars['date'].max()} ({bars['date'].nunique()} days)")
    print(f"Missing regular-session minutes: {missing}; no missing prices were invented")
    print(f"Output: {args.output.resolve()}")
    print(f"CSV: {args.csv_output.resolve()}")


if __name__ == "__main__":
    main()
