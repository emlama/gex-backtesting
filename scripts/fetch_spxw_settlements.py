"""Download official Cboe SPXW weekly and month-end settlement values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("data/spxw_settlements_2026.csv"))
    args = parser.parse_args()

    rows = []
    for kind, label in (("W", "weekly"), ("M", "month_end")):
        url = f"https://www.cboe.com/index_settlement_values/get_sv_data/{kind}/{args.year}/"
        with urlopen(url, timeout=30) as response:
            payload = json.load(response)
        for item in payload["data"]:
            if item["trading_symbol"] == "SPXW":
                rows.append(
                    {
                        "date": item["expiration_date"],
                        "settlement": float(item["settlement_value"]),
                        "series": label,
                        "source_url": url,
                    }
                )

    result = pd.DataFrame(rows).sort_values("date")
    if result["date"].duplicated().any():
        raise ValueError("Cboe returned multiple SPXW settlements on one date")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} SPXW settlement dates to {args.output.resolve()}")


if __name__ == "__main__":
    main()
