"""Write the standalone SPXW bulk-backtest dashboard.

The page is one offline HTML file: `bulk_dashboard.html` is the template and
this module embeds the run's summary, accepted trades, rejections, and
coverage as JSON. No network requests or third-party scripts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

TEMPLATE = Path(__file__).with_name("bulk_dashboard.html")
PLACEHOLDER = "__DASHBOARD_DATA__"
TRADE_COLUMNS = [
    "signal", "structure", "dte", "date", "expiry", "settlement", "width", "put_offset", "call_offset",
    "short_put", "short_call", "credit_points", "expiry_debit_points", "fees_dollars", "pnl_dollars",
    "max_loss_dollars", "return_on_risk", "max_quote_age_s",
]


def records(frame: pd.DataFrame) -> list[dict]:
    """Convert a frame to JSON-safe records, with NaN as null."""
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def payload(summary: pd.DataFrame, rows: pd.DataFrame, coverage: pd.DataFrame, meta: dict) -> dict:
    trades = []
    rejections = []
    if not rows.empty:
        accepted = rows[rows["status"] == "accepted"].sort_values("date")
        for trade, quotes in zip(records(accepted[TRADE_COLUMNS]), accepted["leg_quotes_json"]):
            trade["legs"] = [
                {key: leg[key] for key in ("side", "type", "strike", "bid", "ask")}
                for leg in json.loads(quotes)
            ]
            trades.append(trade)
        rejected = rows[rows["status"] != "accepted"]
        if not rejected.empty:
            counts = rejected.groupby(["signal", "dte", "reason"]).size().rename("candidates").reset_index()
            rejections = records(counts)
    return {
        "meta": meta | {"candidate_rows": len(rows), "accepted": len(trades)},
        "variants": records(summary),
        "trades": trades,
        "rejections": rejections,
        "coverage": records(coverage),
    }


def write_dashboard(
    summary: pd.DataFrame, rows: pd.DataFrame, coverage: pd.DataFrame, meta: dict, destination: Path
) -> None:
    data = json.dumps(payload(summary, rows, coverage, meta), ensure_ascii=False, allow_nan=False)
    # The JSON sits inside a <script> element; keep "</script>" from closing it early.
    data = data.replace("</", "<\\/")
    page = TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, data, 1)
    destination.write_text(page, encoding="utf-8")
