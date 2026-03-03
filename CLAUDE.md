# GEX Backtesting Toolkit

## Overview

SPX 0DTE Gamma Exposure (GEX) backtesting toolkit for analyzing whether gamma-related metrics
predict late-day PUT price explosions. Built on 513 days of enriched options trade data
(2024-01-02 through 2026-02-19) from Polygon.io flat files with quote-matched trade side classification.

### Research Question

> Can dealer gamma concentration and convexity metrics (GCI, PGR, GDW, CAR, Charm, Vomma, Zomma)
> measured during the 2:00-3:45 PM ET window predict 0DTE PUT option price explosions
> (>100% gain within 15-60 minutes)?

## Directory Structure

```
gex-backtesting/
├── CLAUDE.md              # This file
├── README.md              # GitHub landing page
├── pyproject.toml         # uv/pip project config (Python >=3.10)
├── download_data.sh       # Download 513-day dataset (2.3 GB) from data server
├── setup_data.sh          # Symlink parquets from Hermes repo instead
├── .gitignore
├── data/                  # Trade parquets (NOT in git)
│   └── trades_YYYY-MM-DD.parquet
├── src/                   # Analysis library (see src/CLAUDE.md)
│   ├── __init__.py
│   ├── config.py          # All thresholds and parameters (dataclasses)
│   ├── data_loader.py     # DataLoader (backtest) + GEXDataLoader (charts)
│   ├── processor.py       # BacktestRunner orchestration
│   ├── metrics.py         # GCI, PGR, GDW, CAR metric calculations
│   ├── gex_calculator.py  # Side-weighted GEX (traditional + directional)
│   ├── greeks.py          # Higher-order Greeks class (gamma, vomma, zomma, charm)
│   ├── black_scholes.py   # Vectorized BS functions (IV, delta, gamma) ~100x faster
│   ├── put_tracker.py     # PUT selection + return measurement
│   ├── statistics.py      # Spearman, Fisher's exact, FDR, permutation tests
│   └── visualization.py   # matplotlib/seaborn chart generation
├── notebooks/
│   ├── 0dte_gex_charts.ipynb          # Daily GEX visualization (single-date explorer)
│   ├── 01_gci_spike_analysis.ipynb    # GCI concentration study (Fisher's exact)
│   └── 02_multi_metric_backtest.ipynb # Multi-metric pre-registered hypothesis test
└── results/               # Generated output (not in git)
```

## Quick Start

```bash
# 1. Install with uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -e ".[jupyter]"

# 2. Download data (513 days, 2.3 GB)
./download_data.sh

# Or symlink from Hermes repo
./setup_data.sh /path/to/parquets

# 3. Run notebooks
jupyter lab notebooks/
```

Start with `0dte_gex_charts.ipynb` -- set `TRADE_DATE` to any date and run all cells.

## Architecture: Two Pipelines

### Pipeline 1: GEX Chart Generation (single-date exploration)

Used by `0dte_gex_charts.ipynb`. Produces daily GEX visualizations: Net Drift, Net Flow,
GEX by strike, DEX, Vomma/Zomma surfaces, CAR, Volatility Skew, Trade Distribution.

```
AnalysisConfig(trade_date="2024-06-14")
    -> GEXDataLoader.load()        # Enriches raw trades with opt_type, strike, trade_dir, tte_years
    -> calculate_gex(df, config)   # Side-weighted + traditional GEX via black_scholes.py
    -> GEXResult.by_strike         # DataFrame for charting
```

Key classes: `AnalysisConfig`, `GEXDataLoader`, `calculate_gex()`, `GEXResult`

### Pipeline 2: Multi-Metric Backtest (cross-date statistical analysis)

Used by `02_multi_metric_backtest.ipynb`. Runs across all 513 days to test whether
gamma metrics predict PUT explosions.

```
Config()
    -> BacktestRunner.run()                    # Loop over all dates
        -> DayProcessor.process(date)          # Per-day orchestration
            -> DataLoader.load_and_prepare()   # Load + filter late-day + create intervals
            -> MetricCalculator.calculate()    # GCI, PGR, GDW, CAR per interval (uses greeks.py)
            -> PutTracker.calculate_returns()  # Track PUT prices at 15/30/45/60 min horizons
    -> BacktestRunner.run_univariate_analysis()   # Spearman + lift per metric
    -> BacktestRunner.run_composite_analysis()    # Combined signals
    -> BacktestRunner.run_control_experiments()   # Placebo, time-shifted, permutation
```

Key classes: `Config`, `BacktestRunner`, `DayProcessor`, `DataLoader`, `MetricCalculator`,
`PutTracker`, `StatisticalAnalyzer`

## Key Domain Concepts

| Metric | Full Name | What It Measures |
|--------|-----------|------------------|
| **GEX** | Gamma Exposure | Dealer gamma position by strike. Positive = stabilizing, negative = destabilizing |
| **Side-Weighted GEX** | -- | GEX using actual trade direction (buy/sell) instead of assuming all customer buys |
| **GCI** | Gamma Concentration Index | Herfindahl index of gamma by strike. High = gamma at few strikes = fragile |
| **PGR** | Protective Gamma Ratio | Fraction of gamma within +/-$20 of spot. Low = gamma far from spot = less protection |
| **GDW** | Gamma Distance Weighted | Exponentially-weighted gamma by distance from spot (decay=20) |
| **CAR** | Convexity Acceleration Risk | Combines zomma + vomma with time amplifier. Measures gamma acceleration risk |
| **Charm** | Delta Decay | Rate of delta change over time. Extreme for 0DTE near expiry |
| **Vomma** | Vol-of-Vol Sensitivity | How vega changes with volatility. High = prices explode on vol spikes |
| **Zomma** | Gamma-Vol Sensitivity | How gamma changes with volatility. Creates feedback loops |

### Side Sign Convention

```
at_ask / above_ask  ->  +1  (customer buy = destabilizing for dealer)
at_bid / below_bid  ->  -1  (customer sell = stabilizing for dealer)
mid_market          ->   0  (ambiguous, excluded from directional calcs)
```

### Time Window

The backtest focuses on the **late-day window**: 2:00 PM - 3:45 PM ET.
This is when 0DTE gamma effects are most extreme (TTE approaching zero).
PUT returns are measured at 15, 30, 45, and 60 minute horizons after signal,
capped at 4:00 PM ET expiry.

## Data Schema

Each file `data/trades_YYYY-MM-DD.parquet` contains enriched SPX 0DTE option trades.

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Option symbol (e.g., `O:SPXW240102C05900000`) |
| `sip_timestamp` | int64 | SIP timestamp in **nanoseconds** since epoch |
| `price` | **string** | Trade price (GOTCHA: stored as string, not float) |
| `size` | **string** | Contract count (GOTCHA: stored as string, not int) |
| `strike` | float64 | Strike price (derived from ticker) |
| `opt_type` | string | `C` or `P` |
| `bid` | float64 | Best bid at trade time |
| `ask` | float64 | Best ask at trade time |
| `side` | string | `at_bid`, `below_bid`, `at_ask`, `above_ask`, `mid_market` |
| `trade_date` | date | Trading date |
| `conditions` | string | SIP condition codes |

### Data Gotchas

1. **`price` and `size` are strings** -- Both `DataLoader` and `GEXDataLoader` convert these
   with `pd.to_numeric(col, errors="coerce")`. If you load parquets directly, you must convert.

2. **No `spot` column** -- SPX spot price is NOT in the parquet files. Both loaders estimate it
   by finding the highest-volume strike per minute as an ATM proxy. This is approximate.

3. **Nanosecond timestamps** -- `sip_timestamp` is nanoseconds, not milliseconds. Convert with
   `pd.to_datetime(col, unit="ns", utc=True)`. The loaders handle this automatically.

4. **`timestamp` column may not exist** -- Raw parquets only have `sip_timestamp`. The loaders
   create a `timestamp` column from it. The column name in the pipeline is always `timestamp`.

5. **TTE for 0DTE** -- Time-to-expiry stored as 0 in some formats. `MetricCalculator` detects
   this and recalculates from the trade's timestamp to 4:00 PM ET market close.

6. **Complex trades** -- SIP condition codes 12, 13, 14, 15, 33, 37, 38 indicate multi-leg
   trades. `GEXDataLoader` filters these out by default (`exclude_complex=True`).

## Common Workflows

### Explore a single day's GEX profile

```python
from src.config import AnalysisConfig
from src.data_loader import GEXDataLoader
from src.gex_calculator import calculate_gex
from pathlib import Path

config = AnalysisConfig(trade_date="2024-06-14")
loader = GEXDataLoader(config, Path("data"))
df = loader.load()
result = calculate_gex(df, config)

# result.by_strike has per-strike GEX data
# result.sw_net is the net side-weighted GEX
print(f"Net GEX: ${result.sw_net / 1e6:,.2f}M")
```

### Run the full multi-metric backtest

```python
from src import Config, BacktestRunner

config = Config()
runner = BacktestRunner(config)

# Process all 513 days (takes ~30-60 min)
df_all = runner.run()

# Univariate screen: which metrics predict PUT explosions?
df_uni = runner.run_univariate_analysis(outcome_threshold=100.0)

# Composite signals
df_comp = runner.run_composite_analysis()

# Control experiments (placebo, time-shifted, permutation)
controls = runner.run_control_experiments()

runner.save_results()
runner.print_summary()
```

### Calculate Greeks for a set of trades

```python
from src.greeks import BlackScholesGreeks
import numpy as np

greeks = BlackScholesGreeks(risk_free_rate=0.05)
result = greeks.calculate_all(
    S=5900.0,                          # spot
    K=np.array([5850, 5900, 5950]),    # strikes
    T=0.001,                           # TTE in years (~6.5 trading hours)
    sigma=0.20,                        # IV
    is_call=np.array([True, True, True])
)
# result keys: gamma, vomma, zomma, charm
```

### Estimate IV from option prices (vectorized)

```python
from src.black_scholes import calculate_greeks
import numpy as np

greeks = calculate_greeks(
    spot=np.array([5900.0, 5900.0]),
    strike=np.array([5850.0, 5950.0]),
    tte=np.array([0.001, 0.001]),
    rate=0.05,
    price=np.array([52.0, 3.5]),
    is_call=np.array([True, False]),
)
# greeks keys: iv, delta, gamma
```

## Coding Conventions

1. **Vectorized numpy over loops** -- Use `np.where`, `np.clip`, array operations.
   `black_scholes.py` is ~100x faster than row-by-row `apply()`.

2. **Pre-registered thresholds** -- All thresholds live in `config.py` dataclasses.
   DO NOT modify thresholds after analysis begins (prevents p-hacking).
   Use `use_percentiles=True` (default) for data-driven thresholds.

3. **Eastern Time always** -- All timestamps are converted to ET. Market close = 4:00 PM ET.

4. **Side-weighted over traditional** -- Traditional GEX assumes all customer buys.
   Side-weighted uses actual trade direction from quote matching.

5. **Conservative PUT pricing** -- Entry at bid (worst fill), exit at mid.

6. **String-to-numeric conversion** -- Always check and convert `price`/`size` columns
   when loading parquets outside the standard loaders.

## Testing

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/

# Lint
ruff check src/
ruff format --check src/
```

**Ruff config**: line-length 120, target Python 3.10, rules E/F/W/I.

## Dependencies

Core: `pandas>=2.0`, `numpy>=1.24`, `pyarrow>=14.0`, `scipy>=1.10`, `statsmodels>=0.14`,
`tqdm>=4.65`, `matplotlib>=3.7`, `seaborn>=0.12`, `pytz>=2023.3`

Optional: `jupyterlab>=4.0`, `ipywidgets>=8.0` (for notebooks)

Dev: `pytest>=7.0`, `ruff>=0.1`
