Run the full GEX backtest pipeline with statistical analysis across all available trading dates.

## Arguments

Optional date range or limit: $ARGUMENTS
- Empty: run all available dates
- A number (e.g. "50"): limit to first N dates
- Date range (e.g. "2024-01-02 2024-06-28"): specific start/end dates

## Instructions

Run the following Python script from the project root (`/Users/esaforrian/Documents/GitHub/gex-backtesting/`):

```python
import sys
sys.path.insert(0, ".")

from datetime import date
from src import Config, BacktestRunner

args = "$ARGUMENTS".strip()

config = Config()
runner = BacktestRunner(config)

# Parse arguments
dates = None
limit = None

if args:
    parts = args.split()
    if len(parts) == 1 and parts[0].isdigit():
        limit = int(parts[0])
        print(f"Running backtest limited to first {limit} dates...")
    elif len(parts) == 2:
        start = date.fromisoformat(parts[0])
        end = date.fromisoformat(parts[1])
        all_dates = runner.data_loader.get_available_dates()
        dates = [d for d in all_dates if start <= d <= end]
        print(f"Running backtest for {len(dates)} dates: {parts[0]} to {parts[1]}")
    else:
        print(f"Unrecognized arguments: {args}")
        print("Usage: <empty> | <limit> | <start_date> <end_date>")
        sys.exit(1)
else:
    print("Running full backtest across all available dates...")

# Step 1: Process all days (calculate metrics + PUT returns per interval)
df_all = runner.run(dates=dates, limit=limit)
print(f"\nProcessed {df_all['date'].nunique()} days, {len(df_all)} intervals")

# Step 2: Univariate analysis (each metric vs each time window)
print("\n--- Univariate Screen ---")
df_uni = runner.run_univariate_analysis(outcome_threshold=100.0)
if len(df_uni) > 0:
    print(df_uni[["metric", "window", "put_method", "lift", "ci_low", "ci_high", "p_adjusted", "significant"]].to_string(index=False))
else:
    print("No significant univariate results.")

# Step 3: Composite signal analysis
print("\n--- Composite Signals ---")
df_comp = runner.run_composite_analysis()
if len(df_comp) > 0:
    print(df_comp.to_string(index=False))

# Step 4: Control experiments (placebo, time-shifted, permutation)
print("\n--- Control Experiments ---")
controls = runner.run_control_experiments(n_permutations=500)
for name, result in controls.items():
    print(f"\n{name}:")
    for k, v in result.items():
        print(f"  {k}: {v}")

# Step 5: Save results
files = runner.save_results()
print(f"\nSaved {len(files)} result files:")
for f in files:
    print(f"  {f}")

# Step 6: Generate visualizations
if runner.univariate_results is not None and len(runner.univariate_results) > 0:
    path = runner.viz.plot_correlation_heatmap(runner.univariate_results)
    print(f"Heatmap: {path}")

if runner.composite_results is not None and len(runner.composite_results) > 0:
    path = runner.viz.plot_composite_comparison(runner.composite_results)
    print(f"Composite chart: {path}")

# Print summary
runner.print_summary()
```

After running, summarize:
- Total days and intervals processed
- Best-performing metric (highest lift with FDR significance)
- Whether composite signals outperform individual metrics
- Control experiment results (placebo lift near 1.0, permutation p-value)
- Any metrics that survived FDR correction
