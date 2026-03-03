"""GEX Backtesting: SPX 0DTE Gamma Exposure Analysis Toolkit.

This package provides tools for analyzing gamma-related metrics
(GCI, PGR, GDW, CAR, Charm, Vomma, Zomma) from SPX 0DTE option trade data.

Usage:
    from src import Config, BacktestRunner, DataLoader

    config = Config()
    runner = BacktestRunner(config)
    df = runner.run(limit=10)
"""

from .config import Config
from .data_loader import DataLoader
from .greeks import BlackScholesGreeks
from .black_scholes import calculate_greeks, calculate_gamma, calculate_delta, estimate_iv_from_price
from .gex_calculator import calculate_gex, GEXResult
from .metrics import MetricCalculator
from .processor import BacktestRunner, DayProcessor
from .put_tracker import PutTracker
from .statistics import StatisticalAnalyzer
from .visualization import Visualizer

__all__ = [
    "Config",
    "DataLoader",
    "BlackScholesGreeks",
    "calculate_greeks",
    "calculate_gamma",
    "calculate_delta",
    "estimate_iv_from_price",
    "calculate_gex",
    "GEXResult",
    "MetricCalculator",
    "PutTracker",
    "StatisticalAnalyzer",
    "Visualizer",
    "DayProcessor",
    "BacktestRunner",
]
