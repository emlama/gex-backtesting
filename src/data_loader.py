"""Data loading and enrichment for SPX 0DTE option trade parquet files.

Loads trade data from the data/ directory (one file per day: trades_YYYY-MM-DD.parquet).
These parquet files originate from Polygon.io flat-file downloads, processed through
the backfill pipeline in the Hermes repo.

Two loader classes are provided:

    DataLoader: Used by the GCI meta-analysis backtest (processor.py / BacktestRunner).
        Loads raw trades, filters to the late-day time window, and creates interval
        buckets for metric calculation.

    GEXDataLoader: Used by the GEX chart analysis notebooks (gex_calculator.py).
        Performs heavier enrichment: derives opt_type and strike from ticker symbols,
        maps trade sides to BUY/SELL direction, and filters complex/multi-leg trades.

Data loading gotchas:
    - Columns like 'strike_price', 'open_interest', 'price', and 'size' may arrive as
      strings (Arrow/Parquet type mismatch). Numeric coercion is applied on load.
    - Timestamp column may be 'sip_timestamp' (nanoseconds since epoch) in backfill
      data or 'timestamp' (datetime) in other formats. Both are handled.
    - Spot price (SPX level) is NOT included in Polygon flat files. When missing, it
      is estimated from the highest-volume strike per minute as an ATM proxy.
    - Time-to-expiry (tte_years) may be stored as 0 for 0DTE. When missing or zero,
      it is calculated from the trade timestamp to market close (4 PM ET).
    - All timestamps are converted to Eastern Time (ET) for consistency.

Dependencies:
    - Config / AnalysisConfig (config.py): Data paths, time windows, trade-side mapping
"""

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AnalysisConfig, Config, ET


class DataLoader:
    """Load and filter trade data from local parquet files.

    Used by the GCI meta-analysis backtest pipeline (processor.py). Handles
    column normalization, timestamp conversion, spot estimation, and
    late-day time filtering.

    The typical call path is load_and_prepare() which chains:
        load_trades_for_date() -> filter_late_day() -> create_intervals()
    """

    def __init__(self, config: Config):
        """Initialize with configuration.

        Args:
            config: Master Config with data_dir path and time_window settings.
        """
        self.config = config
        self.data_dir = config.get_data_dir()

    def get_available_dates(self) -> list[date]:
        """Find all available trading dates by scanning parquet filenames.

        Returns:
            Sorted list of dates for which trades_YYYY-MM-DD.parquet files exist.
        """
        files = list(self.data_dir.glob("trades_*.parquet"))
        dates = []
        for f in files:
            date_str = f.stem.replace("trades_", "")
            try:
                dates.append(pd.to_datetime(date_str).date())
            except (ValueError, TypeError):
                pass
        return sorted(dates)

    def load_trades_for_date(self, trade_date: str | date) -> pd.DataFrame:
        """Load trades for a single date from parquet file.

        Performs the following enrichment steps:
        1. Coerces 'price' and 'size' columns to numeric (may be strings from Arrow).
        2. Normalizes timestamp column: 'sip_timestamp' (nanoseconds since epoch)
           is converted to datetime; all timestamps are converted to ET.
        3. Estimates spot price from ATM strike proximity if 'spot' column is missing
           (common in Polygon flat-file data which only has option trades, not SPX quotes).
        4. Computes tte_years from timestamp to market close if not present.

        Args:
            trade_date: Date string "YYYY-MM-DD" or date object.

        Returns:
            DataFrame of enriched trades, or empty DataFrame if file not found.

        Raises:
            ValueError: If no recognized timestamp column exists in the parquet file.
        """
        if isinstance(trade_date, date):
            trade_date = trade_date.strftime("%Y-%m-%d")

        file_path = self.data_dir / f"trades_{trade_date}.parquet"
        if not file_path.exists():
            return pd.DataFrame()

        df = pd.read_parquet(file_path)

        # Convert numeric columns that may be stored as strings (Arrow or Python)
        for col in ["price", "size"]:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Handle timestamp column name
        if "sip_timestamp" in df.columns and "timestamp" not in df.columns:
            df["timestamp"] = pd.to_datetime(df["sip_timestamp"], unit="ns", utc=True)
        elif "timestamp" not in df.columns:
            raise ValueError(f"No timestamp column found in {file_path}")

        # Ensure timestamp is timezone-aware ET
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        df["timestamp"] = df["timestamp"].dt.tz_convert(ET)

        # Derive spot price if not present
        # Use the most-traded strike as ATM proxy
        if "spot" not in df.columns:
            df["spot"] = self._estimate_spot(df, trade_date)

        # Add time-to-expiry
        if "tte_years" not in df.columns:
            market_close = pd.Timestamp(f"{trade_date} 16:00", tz="America/New_York")
            tte_hours = (market_close - df["timestamp"]).dt.total_seconds() / 3600
            df["tte_years"] = tte_hours / (365 * 24)
            df = df[df["tte_years"] > 0].copy()

        return df

    def _estimate_spot(self, df: pd.DataFrame, trade_date: str) -> pd.Series:
        """Estimate SPX spot price from option trade data.

        Uses a time-bucketed approach: for each minute, the ATM strike
        (highest volume) approximates the spot price.
        """
        if "strike" not in df.columns:
            return pd.Series(0.0, index=df.index)

        df_copy = df.copy()
        df_copy["minute"] = df_copy["timestamp"].dt.floor("1min")

        # For each minute, find the strike with the most volume
        volume_by_strike = df_copy.groupby(["minute", "strike"])["size"].sum().reset_index()
        atm_by_minute = volume_by_strike.loc[volume_by_strike.groupby("minute")["size"].idxmax()]
        atm_lookup = atm_by_minute.set_index("minute")["strike"].to_dict()

        # Map back to original DataFrame
        spot = df_copy["minute"].map(atm_lookup)
        return spot.ffill().bfill()

    def filter_late_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter trades to the late-day analysis window.

        Uses config.time_window (default 2:00 PM - 3:45 PM ET) to select
        only trades that fall within the study period. This is when gamma
        effects are most pronounced for 0DTE options.

        Args:
            df: Trade DataFrame with a 'timestamp' column in ET.

        Returns:
            Filtered DataFrame containing only late-day trades.
        """
        if len(df) == 0:
            return df
        df = df.copy()
        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute
        df["time_minutes"] = df["hour"] * 60 + df["minute"]
        window = self.config.time_window
        mask = (df["time_minutes"] >= window.start_minutes) & (
            df["time_minutes"] <= window.end_minutes
        )
        return df[mask]

    def create_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add an 'interval' column for time-bucket grouping.

        Floors each timestamp to the nearest interval boundary (default 5 min).
        The DayProcessor then groups by this column to compute per-interval metrics.

        Args:
            df: Trade DataFrame with a 'timestamp' column.

        Returns:
            DataFrame with an added 'interval' column (Timestamp floored to interval).
        """
        df = df.copy()
        df["interval"] = df["timestamp"].dt.floor(f"{self.config.interval_minutes}min")
        return df

    def load_and_prepare(self, trade_date: str | date) -> pd.DataFrame:
        """Load trades, filter to late-day window, and add interval column.

        Convenience method that chains load_trades_for_date(), filter_late_day(),
        and create_intervals(). This is the primary entry point used by
        DayProcessor.process().

        Args:
            trade_date: Date string "YYYY-MM-DD" or date object.

        Returns:
            DataFrame ready for per-interval metric calculation, or empty DataFrame
            if no data is available for the date or the late-day window is empty.
        """
        df = self.load_trades_for_date(trade_date)
        if len(df) == 0:
            return pd.DataFrame()
        df = self.filter_late_day(df)
        if len(df) == 0:
            return pd.DataFrame()
        df = self.create_intervals(df)
        return df


class GEXDataLoader:
    """Load and enrich trade data for GEX chart analysis.

    This loader performs heavier enrichment than DataLoader, deriving fields
    that calculate_gex() requires but that may not exist in raw parquet files:
        - opt_type: Extracted from ticker symbol (e.g., "O:SPXW241220C05900000" -> "C")
        - strike: Extracted from ticker symbol (e.g., "...C05900000" -> 5900.0)
        - trade_dir: Mapped from Polygon 'side' field (at_ask -> BUY, at_bid -> SELL)
        - tte_years: Calculated from timestamp to market close

    Also filters out complex/multi-leg trades using Polygon condition codes.
    """

    def __init__(self, config: AnalysisConfig, data_dir: Path):
        """Initialize with analysis config and data directory.

        Args:
            config: AnalysisConfig with trade_date, buy/sell side mappings,
                complex trade codes, and market close time.
            data_dir: Path to directory containing trades_YYYY-MM-DD.parquet files.
        """
        self.config = config
        self.data_dir = data_dir

    def load(self, verbose: bool = True) -> pd.DataFrame:
        """Load and enrich trade data for the configured trade_date.

        Enrichment pipeline:
        1. Read parquet and coerce numeric columns (price, size may be strings).
        2. Normalize timestamps to ET.
        3. Derive opt_type ("C"/"P") from ticker if not present.
        4. Derive strike from ticker symbol's numeric suffix if not present.
        5. Map trade side to BUY/SELL direction using config buy/sell side lists.
        6. Estimate spot from highest-volume strike per minute if not present.
        7. Compute tte_years from timestamp to market close.
        8. Filter out complex/multi-leg trades (condition codes 12-15, 33, 37, 38).

        Args:
            verbose: If True, print row counts before and after enrichment.

        Returns:
            Enriched DataFrame ready for calculate_gex(), or empty DataFrame
            if the file is not found or required columns cannot be derived.
        """
        file_path = self.data_dir / f"trades_{self.config.trade_date}.parquet"
        if not file_path.exists():
            if verbose:
                print(f"File not found: {file_path}")
            return pd.DataFrame()

        df = pd.read_parquet(file_path)
        if verbose:
            print(f"Loaded {len(df):,} raw trades from {file_path.name}")

        # Convert numeric columns (Arrow string or Python object)
        for col in ["price", "size"]:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Handle timestamp
        if "sip_timestamp" in df.columns and "timestamp" not in df.columns:
            df["timestamp"] = pd.to_datetime(df["sip_timestamp"], unit="ns", utc=True)
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")

        # Derive opt_type from ticker if needed
        if "opt_type" not in df.columns:
            if "ticker" in df.columns:
                df["opt_type"] = df["ticker"].apply(
                    lambda s: "C" if "C0" in str(s) else ("P" if "P0" in str(s) else None)
                )
            else:
                return pd.DataFrame()
            df = df[df["opt_type"].notna()].copy()

        # Derive strike from ticker if needed
        if "strike" not in df.columns or df["strike"].isna().all():
            if "ticker" in df.columns:
                def extract_strike(symbol: str) -> float | None:
                    match = re.search(r"[CP](\d+)$", str(symbol))
                    return float(match.group(1)) / 1000 if match else None
                df["strike"] = df["ticker"].apply(extract_strike)
            df = df[df["strike"].notna()].copy()

        # Derive trade_dir from side
        if "trade_dir" not in df.columns:
            if "side" in df.columns:
                df["trade_dir"] = df["side"].apply(
                    lambda s: "BUY" if s in self.config.buy_sides
                    else ("SELL" if s in self.config.sell_sides else None)
                    if s else None
                )
                df = df[df["trade_dir"].notna()].copy()
            else:
                return pd.DataFrame()

        # Estimate spot from ATM strikes if not present
        if "spot" not in df.columns or df["spot"].isna().all():
            df["spot"] = self._estimate_spot(df)

        # Time to expiry
        if "tte_years" not in df.columns:
            market_close = pd.Timestamp(
                f"{self.config.trade_date} {self.config.market_close_time}",
                tz="America/New_York",
            )
            tte_hours = (market_close - df["timestamp"]).dt.total_seconds() / 3600
            df["tte_years"] = tte_hours / (365 * 24)
            df = df[df["tte_years"] > 0].copy()

        # Filter complex trades
        if self.config.exclude_complex and "conditions" in df.columns:
            complex_set = set(self.config.complex_codes)
            df = df[
                ~df["conditions"].apply(
                    lambda x: bool(set(x or []) & complex_set) if isinstance(x, list) else False
                )
            ]

        if verbose:
            print(f"  After enrichment: {len(df):,} trades")

        return df

    def _estimate_spot(self, df: pd.DataFrame) -> pd.Series:
        """Estimate SPX spot from highest-volume strike per minute.

        Since Polygon flat files contain only option trades (no SPX quotes),
        spot is approximated by finding the strike with the most volume in each
        1-minute bucket (assumed to be the ATM strike). Forward/backward fill
        handles minutes with no trades.

        Args:
            df: Trade DataFrame with 'timestamp', 'strike', and 'size' columns.

        Returns:
            Series of estimated spot prices aligned with df index.
        """
        df_copy = df.copy()
        df_copy["minute"] = df_copy["timestamp"].dt.floor("1min")
        vol = df_copy.groupby(["minute", "strike"])["size"].sum().reset_index()
        atm = vol.loc[vol.groupby("minute")["size"].idxmax()]
        lookup = atm.set_index("minute")["strike"].to_dict()
        spot = df_copy["minute"].map(lookup)
        return spot.ffill().bfill()
