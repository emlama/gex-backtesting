# src/ -- Analysis Library

## Module Responsibilities

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `config.py` | All parameters as dataclasses. Pre-registered thresholds, time windows, paths | `Config`, `AnalysisConfig`, `Thresholds`, `TimeWindow`, `ET` |
| `data_loader.py` | Load parquets, convert types, estimate spot, filter late-day, create intervals | `DataLoader`, `GEXDataLoader` |
| `processor.py` | Backtest orchestration: loop dates, aggregate results, run analyses | `BacktestRunner`, `DayProcessor` |
| `metrics.py` | Calculate GCI, PGR, GDW, CAR, charm/vomma/zomma exposure per interval | `MetricCalculator`, `IntervalMetrics` |
| `gex_calculator.py` | Traditional + side-weighted GEX by strike (uses `black_scholes.py`) | `calculate_gex()`, `GEXResult` |
| `greeks.py` | Higher-order Greeks class: gamma, vomma, zomma, charm (analytical BS formulas) | `BlackScholesGreeks` |
| `black_scholes.py` | Vectorized numpy BS: IV estimation (Newton-Raphson), delta, gamma (~100x faster) | `calculate_greeks()`, `estimate_iv_from_price()` |
| `put_tracker.py` | Select PUT strikes (N-OTM or max-vomma), track prices, measure returns | `PutTracker`, `PutReturns` |
| `statistics.py` | Spearman correlation, Fisher's exact, bootstrap CI, FDR correction, permutation | `StatisticalAnalyzer` |
| `visualization.py` | Heatmaps, distributions, signal comparisons, metric timeseries (dark theme) | `Visualizer` |

## Why Two Greeks Modules?

- **`greeks.py` (`BlackScholesGreeks` class)** -- Analytical higher-order Greeks: gamma, vomma, zomma, charm.
  Used by `MetricCalculator` (backtest pipeline) and `PutTracker`. Takes spot/strike/TTE/sigma as input.
  Does NOT estimate IV -- requires sigma as a known input.

- **`black_scholes.py` (standalone functions)** -- Vectorized IV estimation from option prices via
  Newton-Raphson, plus delta and gamma. Used by `gex_calculator.py` (chart pipeline).
  Takes spot/strike/TTE/rate/**price** as input and first solves for IV.

**When to use which**: If you already have IV, use `greeks.py`. If you need to derive IV from
observed option prices, use `black_scholes.py`.

## Why Two DataLoaders?

- **`DataLoader`** -- For the backtest pipeline (`Config`-based). Loads trades, filters to the
  late-day time window (2:00-3:45 PM ET), creates 5-minute intervals. Used by `DayProcessor`.

- **`GEXDataLoader`** -- For the chart pipeline (`AnalysisConfig`-based). Loads full-day trades,
  enriches with `opt_type`, `strike`, `trade_dir`, `tte_years`. Filters complex trades.
  Used by `0dte_gex_charts.ipynb` and `calculate_gex()`.

Both handle the same string-to-numeric conversions and spot estimation, but serve different pipelines.

## Data Type Gotchas

1. **`price` and `size` are strings in parquet** -- Both loaders call
   `pd.to_numeric(col, errors="coerce")`. If loading parquets directly, convert first.

2. **`sip_timestamp` is nanoseconds (int64)** -- Convert with `pd.to_datetime(col, unit="ns", utc=True)`.
   The loaders create a `timestamp` column from this.

3. **`opt_type` has two name variants** -- Analytics Bucket uses `opt_type`, some formats use
   `option_type`. `MetricCalculator` checks both: `"opt_type" if "opt_type" in df.columns else "option_type"`.

4. **`opt_type` value formats** -- Can be `"C"/"P"` or `"call"/"put"` (case-insensitive).
   `MetricCalculator` normalizes with `.str.upper()` and checks `isin(["C", "CALL"])`.

## Side-Sign Convention

Calls are positive gamma for dealers (stabilizing when bought by customers).
The sign convention in `MetricCalculator._side_sign_map`:

```
at_ask / above_ask  ->  +1  (customer buy)
at_bid / below_bid  ->  -1  (customer sell)
mid_market          ->   0  (excluded from directional metrics)
```

Gamma exposure is always positive (`abs(gamma) * size * 100`).
Directional exposures (vomma, zomma, charm) are multiplied by the side sign.

## Adding a New Metric

1. **Define the calculation** in `metrics.py` inside `MetricCalculator.calculate()`.
   Add the new field to the `IntervalMetrics` dataclass.

2. **Add to the dict** in `MetricCalculator.metrics_to_dict()` so it propagates to the results DataFrame.

3. **Add threshold** in `config.py` in the `Thresholds` dataclass (for fixed thresholds)
   or let `use_percentiles=True` handle it automatically.

4. **Include in univariate screen** by adding the metric name to the `metrics` list in
   `BacktestRunner.run_univariate_analysis()`.

5. **Test** by running a small backtest: `BacktestRunner(Config()).run(limit=5)` and checking
   that the new column appears in the results DataFrame.
