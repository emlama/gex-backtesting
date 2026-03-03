Generate a GEX (Gamma Exposure) analysis chart for a specific trading date.

## Arguments

Date to analyze: $ARGUMENTS (format: YYYY-MM-DD, e.g. 2024-01-02)

## Instructions

Run the following Python script from the project root (`/Users/esaforrian/Documents/GitHub/gex-backtesting/`):

```python
import sys
sys.path.insert(0, ".")

from src.config import AnalysisConfig
from src.data_loader import GEXDataLoader
from src.gex_calculator import calculate_gex
from src.visualization import Visualizer
from src.config import Config
from pathlib import Path

# Configuration
trade_date = "$ARGUMENTS"
config = AnalysisConfig(trade_date=trade_date, strike_range=200)
data_dir = Path("data")

# Load and enrich trade data
loader = GEXDataLoader(config, data_dir)
df = loader.load(verbose=True)

if df.empty:
    print(f"No data found for {trade_date}. Check that data/trades_{trade_date}.parquet exists.")
    print("Run ./download_data.sh to download the dataset.")
    sys.exit(1)

# Calculate GEX
result = calculate_gex(df, config, verbose=True)

# Print strike-level summary
print(f"\nTop 10 strikes by absolute Side-Weighted GEX:")
top = result.by_strike.reindex(columns=["strike", "sw_gex", "trad_gex", "volume"])
top["abs_sw"] = top["sw_gex"].abs()
top = top.sort_values("abs_sw", ascending=False).head(10)
print(top.to_string(index=False))

# Generate bar chart of GEX by strike
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16, 8))
bs = result.by_strike
ax.bar(bs["strike"], bs["sw_gex"], width=3, color=["green" if v > 0 else "red" for v in bs["sw_gex"]], alpha=0.7)
ax.axvline(x=result.spot, color="white", linestyle="--", linewidth=2, label=f"Spot ${result.spot:,.0f}")
ax.set_xlabel("Strike")
ax.set_ylabel("Side-Weighted GEX ($)")
ax.set_title(f"SPX 0DTE Side-Weighted GEX - {trade_date}\nNet: ${result.sw_net/1e6:,.2f}M | Spot: ${result.spot:,.0f}")
ax.legend()
ax.grid(alpha=0.3)
plt.style.use("dark_background")
plt.tight_layout()

out_path = Path("results") / f"gex_chart_{trade_date}.png"
out_path.parent.mkdir(exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nChart saved to {out_path}")
```

After running, display the chart image and summarize:
- Net Side-Weighted GEX (stabilizing vs destabilizing)
- Call vs Put GEX breakdown
- Key strikes with highest gamma concentration
- Buy/sell percentages for calls and puts
