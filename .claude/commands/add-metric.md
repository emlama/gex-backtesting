Guide for adding a new metric to the GEX backtesting pipeline.

## Arguments

New metric name and description: $ARGUMENTS (e.g. "gamma_skew - Ratio of call gamma to put gamma near spot")

## Instructions

Follow these steps to add the new metric to the pipeline. All paths are relative to the project root `/Users/esaforrian/Documents/GitHub/gex-backtesting/`.

### Step 1: Define the metric in IntervalMetrics

File: `src/metrics.py`
Class: `IntervalMetrics` (dataclass at line ~55)

Add the new field to the dataclass:

```python
@dataclass
class IntervalMetrics:
    # ... existing fields ...
    new_metric: float  # <-- Add here with a descriptive comment
```

### Step 2: Calculate the metric in MetricCalculator

File: `src/metrics.py`
Class: `MetricCalculator`
Method: `calculate()` (line ~107)

Add the calculation logic inside the `calculate()` method. At this point you have access to:
- `df`: DataFrame of trades in the interval (with columns: strike, size, side, gamma, vomma, zomma, charm, gamma_exp, etc.)
- `spot`: Current spot price
- `sigma`: Median implied volatility
- `tte`: Time-to-expiry in years
- `side_sign`: Series mapping trade sides to +1 (destabilizing) / -1 (stabilizing) / 0 (neutral)
- `gamma_by_strike`: Series of total gamma exposure by strike
- `gamma_total`: Total gamma exposure across all strikes
- `net_gex`: Net signed GEX
- `gex_by_strike`: Series of signed GEX by strike

Include the new value in the `IntervalMetrics(...)` return statement.

### Step 3: Add to metrics_to_dict

File: `src/metrics.py`
Method: `MetricCalculator.metrics_to_dict()` (line ~231)

Add the new metric key so it gets included in the output DataFrame:

```python
def metrics_to_dict(self, metrics: IntervalMetrics) -> dict:
    return {
        # ... existing metrics ...
        "new_metric": metrics.new_metric,
    }
```

### Step 4: Register threshold in Config

File: `src/config.py`
Class: `Thresholds` (dataclass at line ~54)

Add a default threshold (or `None` if percentile-based):

```python
@dataclass
class Thresholds:
    # ... existing thresholds ...
    new_metric: Optional[float] = None  # Will use percentile-based threshold
```

### Step 5: Add to univariate screen

File: `src/processor.py`
Method: `BacktestRunner.run_univariate_analysis()` (line ~134)

Add the metric name to the `metrics` list:

```python
metrics = ["gci", "pgr", "gdw", "car_net", "car_gross", "charm_risk",
           "vomma_exp", "zomma_exp", "new_metric"]
```

### Step 6: Update visualizations (optional)

File: `src/visualization.py`
Method: `Visualizer.plot_metric_timeseries()` (line ~306)

The default metrics list only includes `["gci", "pgr", "car_net"]`. To include the new metric in timeseries plots by default, add it to the default list or pass it explicitly when calling.

### Step 7: Test the metric

Run a quick single-day test:

```bash
cd /Users/esaforrian/Documents/GitHub/gex-backtesting
python -c "
from src import Config, DayProcessor
config = Config()
proc = DayProcessor(config)
result = proc.process('2024-01-02')
if result:
    print(result.df[['interval', 'new_metric']].to_string(index=False))
"
```

Then run the full test suite:

```bash
pytest tests/ -v
```

### Summary of files to modify

| File | What to change |
|------|----------------|
| `src/metrics.py` | Add field to `IntervalMetrics`, calculation in `calculate()`, key in `metrics_to_dict()` |
| `src/config.py` | Add threshold to `Thresholds` dataclass |
| `src/processor.py` | Add metric name to `run_univariate_analysis()` metrics list |
| `src/visualization.py` | (Optional) Add to default plot metrics |

Now implement the metric described in `$ARGUMENTS` following these steps. After implementing, run the single-day test to verify it works.
