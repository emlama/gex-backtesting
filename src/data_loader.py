"""Data loading from local parquet files.

Loads SPX 0DTE trade data from the data/ directory.
Each file is named trades_YYYY-MM-DD.parquet.
"""

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AnalysisConfig, Config, ET


class DataLoader:
    """Load and filter trade data from local parquet files."""

    def __init__(self, config: Config):
        self.config = config
        self.data_dir = config.get_data_dir()

    def get_available_dates(self) -> list[date]:
        """Find all available trading dates from parquet files."""
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

        Handles column name differences between backfill formats:
        - sip_timestamp (nanoseconds) vs timestamp (datetime)
        - Derives spot from ATM strike proximity if not present
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
        """Filter to late-day window only."""
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
        """Add interval column for grouping."""
        df = df.copy()
        df["interval"] = df["timestamp"].dt.floor(f"{self.config.interval_minutes}min")
        return df

    def load_and_prepare(self, trade_date: str | date) -> pd.DataFrame:
        """Load, filter to late-day, and add intervals."""
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

    This loader adds derived fields needed by calculate_gex():
    opt_type, strike, trade_dir, tte_years.
    """

    def __init__(self, config: AnalysisConfig, data_dir: Path):
        self.config = config
        self.data_dir = data_dir

    def load(self, verbose: bool = True) -> pd.DataFrame:
        """Load and enrich trade data for a single date."""
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
        """Estimate spot from highest-volume strike per minute."""
        df_copy = df.copy()
        df_copy["minute"] = df_copy["timestamp"].dt.floor("1min")
        vol = df_copy.groupby(["minute", "strike"])["size"].sum().reset_index()
        atm = vol.loc[vol.groupby("minute")["size"].idxmax()]
        lookup = atm.set_index("minute")["strike"].to_dict()
        spot = df_copy["minute"].map(lookup)
        return spot.ffill().bfill()
