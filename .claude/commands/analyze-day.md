Deep-dive analysis of all gamma metrics for a single trading day, with interval-by-interval breakdown.

## Arguments

Date to analyze: $ARGUMENTS (format: YYYY-MM-DD, e.g. 2024-01-02)

## Instructions

Run the following Python script from the project root (`/Users/esaforrian/Documents/GitHub/gex-backtesting/`):

```python
import sys
sys.path.insert(0, ".")

from src import Config, DayProcessor, Visualizer, DataLoader

trade_date = "$ARGUMENTS"
config = Config()
loader = DataLoader(config)
processor = DayProcessor(config, loader)
viz = Visualizer(config)

# Process the day
result = processor.process(trade_date)

if result is None:
    print(f"No data for {trade_date}. Check that data/trades_{trade_date}.parquet exists.")
    print("Run ./download_data.sh to download the dataset.")
    sys.exit(1)

df = result.df
print(f"Date: {trade_date}")
print(f"Intervals: {result.n_intervals}")
print(f"Time window: {config.time_window.start_hour}:{config.time_window.start_min:02d} - "
      f"{config.time_window.end_hour}:{config.time_window.end_min:02d} ET")
print(f"Interval size: {config.interval_minutes} minutes")

# Interval-by-interval breakdown
print("\n" + "=" * 100)
print("INTERVAL-BY-INTERVAL METRICS")
print("=" * 100)

display_cols = ["interval", "spot", "gci", "pgr", "gdw", "car_net", "car_gross",
                "charm_risk", "vomma_exp", "zomma_exp", "n_trades", "avg_iv", "tte_years"]
available = [c for c in display_cols if c in df.columns]
print(df[available].to_string(index=False))

# Summary statistics
print("\n" + "=" * 100)
print("METRIC SUMMARY")
print("=" * 100)
metrics = ["gci", "pgr", "gdw", "car_net", "car_gross", "charm_risk", "vomma_exp", "zomma_exp"]
for m in metrics:
    if m in df.columns:
        vals = df[m]
        print(f"  {m:12s}  min={vals.min():10.4f}  max={vals.max():10.4f}  "
              f"mean={vals.mean():10.4f}  std={vals.std():10.4f}")

# PUT return columns if available
gain_cols = [c for c in df.columns if "pct_gain" in c]
if gain_cols:
    print("\n" + "=" * 100)
    print("PUT RETURN TRACKING")
    print("=" * 100)
    for col in sorted(gain_cols):
        vals = df[col].dropna()
        if len(vals) > 0:
            print(f"  {col:30s}  n={len(vals):3d}  mean={vals.mean():8.1f}%  "
                  f"median={vals.median():8.1f}%  max={vals.max():8.1f}%")

# Generate timeseries chart
all_metrics = ["gci", "pgr", "car_net", "charm_risk", "vomma_exp", "zomma_exp"]
plot_metrics = [m for m in all_metrics if m in df.columns]
path = viz.plot_metric_timeseries(df, trade_date, metrics=plot_metrics)
print(f"\nTimeseries chart saved to {path}")
```

After running, display the timeseries chart and provide analysis:
- Which intervals show the highest GCI concentration
- Whether PGR drops (low protective gamma) coincide with GCI spikes
- CAR trend (accelerating convexity risk toward close)
- Charm risk evolution as TTE decreases
- Any notable patterns or anomalies in the day's metrics
