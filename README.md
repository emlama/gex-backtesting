# GEX Backtesting Toolkit

SPX 0DTE Gamma Exposure (GEX) analysis toolkit with 305 days of enriched options trade data.

## What's Included

### Dataset (1.3 GB)
- **305 daily parquet files** of SPX 0DTE option trades (2024-01-02 through 2025-03-20)
- Source: Polygon.io flat files, enriched with quote-matched trade side classification
- ~400K-900K trades per day with bid/ask/side labeling

**Schema** (`trades_YYYY-MM-DD.parquet`):

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Option symbol (e.g., `O:SPXW240102C05900000`) |
| `sip_timestamp` | int64 | SIP timestamp (nanoseconds since epoch) |
| `price` | string | Trade price |
| `size` | string | Number of contracts |
| `strike` | float64 | Strike price (derived from ticker) |
| `opt_type` | string | `C` (call) or `P` (put) |
| `bid` | float64 | Best bid at trade time |
| `ask` | float64 | Best ask at trade time |
| `side` | string | Trade classification: `at_bid`, `below_bid`, `at_ask`, `above_ask`, `mid_market` |
| `trade_date` | date | Trading date |
| `conditions` | string | SIP condition codes |

### Notebooks

| Notebook | Description |
|----------|-------------|
| `0dte_gex_charts.ipynb` | **Daily GEX visualization** — Net Drift, Net Flow, GEX by strike, Vomma/Zomma, CAR, Volatility Skew, Trade Distribution. Pick any date and see the full GEX picture. |
| `01_gci_spike_analysis.ipynb` | **GCI concentration study** — Tests if Gamma Concentration Index spikes predict outsized late-day SPX moves. Contingency tables, Fisher's exact tests. |
| `02_multi_metric_backtest.ipynb` | **Multi-metric PUT prediction** — Pre-registered hypothesis test for GCI, PGR, GDW, CAR, Charm, Vomma, Zomma predicting PUT price explosions. Includes control experiments and FDR correction. |

### Analysis Library (`src/`)

| Module | Purpose |
|--------|---------|
| `black_scholes.py` | Vectorized Black-Scholes (IV, delta, gamma) — 100x faster than row-by-row |
| `gex_calculator.py` | Side-weighted GEX calculation (traditional + directional) |
| `greeks.py` | Higher-order Greeks (gamma, vomma, zomma, charm) |
| `metrics.py` | GCI, PGR, GDW, CAR metric calculations |
| `data_loader.py` | Local parquet data loading with enrichment |
| `processor.py` | Backtest orchestration (day processing, multi-date runs) |
| `put_tracker.py` | PUT option price tracking for return measurement |
| `statistics.py` | Spearman correlation, Fisher's exact, FDR correction, permutation tests |
| `visualization.py` | Publication-quality charts (heatmaps, distributions, comparisons) |

## Quick Start

### 1. Install dependencies

```bash
# Using uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -e ".[jupyter]"

# Or pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[jupyter]"
```

### 2. Set up data

Copy or symlink the parquet files into `data/`:

```bash
# Option A: Symlink (if you have the data locally)
ln -s /path/to/trades_parquets/*.parquet data/

# Option B: Copy
cp /path/to/trades_parquets/*.parquet data/

# Verify
ls data/trades_*.parquet | wc -l  # Should show 305
```

### 3. Run notebooks

```bash
jupyter lab notebooks/
```

Start with `0dte_gex_charts.ipynb` — set `TRADE_DATE` to any date and run all cells.

## Key Concepts

### Side-Weighted GEX
Traditional GEX assumes all options are customer buys. Side-weighted GEX uses actual trade direction (from quote-matching) to determine if a trade is a buy or sell, giving a more accurate picture of dealer gamma exposure.

### GCI (Gamma Concentration Index)
Herfindahl-Hirschman Index applied to gamma exposure by strike. High GCI means gamma is concentrated at a few strikes — potential fragility.

### CAR (Convexity Acceleration Risk)
Combines zomma and vomma exposure with time amplification. Measures how quickly gamma can change when volatility spikes near expiry.

### PGR (Protective Gamma Ratio)
Fraction of total gamma within ±$20 of spot. Low PGR means gamma is spread far from spot — less protection against moves.

## Project Structure

```
gex-backtesting/
├── README.md
├── pyproject.toml
├── .gitignore
├── data/                  # Parquet files (not in git — see setup)
│   └── trades_*.parquet
├── src/                   # Analysis library
│   ├── __init__.py
│   ├── black_scholes.py
│   ├── config.py
│   ├── data_loader.py
│   ├── gex_calculator.py
│   ├── greeks.py
│   ├── metrics.py
│   ├── processor.py
│   ├── put_tracker.py
│   ├── statistics.py
│   └── visualization.py
├── notebooks/
│   ├── 0dte_gex_charts.ipynb
│   ├── 01_gci_spike_analysis.ipynb
│   └── 02_multi_metric_backtest.ipynb
└── results/               # Generated output
```

## Requirements

- Python >= 3.10
- pandas, numpy, pyarrow, scipy, statsmodels, matplotlib, seaborn, tqdm
- JupyterLab (optional, for notebooks)
