"""Summarize four-leg UW trade-time NBBO snapshots against SPXW settlement.

This is an indicative quote-based expiry calculation, not evidence that a
four-leg combination could have been filled at the displayed leg NBBO.
"""

from __future__ import annotations

import csv
import json

from probe_uw_history import ROOT


def main() -> None:
    signal_file = ROOT / "results/double_breakout_first_hour.csv"
    with signal_file.open(newline="", encoding="utf-8") as file:
        signals = {row["date"]: row for row in csv.DictReader(file)}

    rows = []
    for path in sorted((ROOT / "data/uw").glob("*_first_hour_5pt_snapshot.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        date = report["metadata"]["date"]
        if date not in signals:
            continue
        legs = report["quote_report"].get("legs", {})
        if len(legs) != 4 or any("nbbo_price" not in leg for leg in legs.values()):
            continue
        quote_credit = round(
            legs["short_call"]["nbbo_price"]
            - legs["long_call"]["nbbo_price"]
            + legs["short_put"]["nbbo_price"]
            - legs["long_put"]["nbbo_price"],
            2,
        )
        settlement = float(signals[date]["spxw_settlement"])
        call_debit = min(
            max(settlement - legs["short_call"]["strike"], 0),
            legs["long_call"]["strike"] - legs["short_call"]["strike"],
        )
        put_debit = min(
            max(legs["short_put"]["strike"] - settlement, 0),
            legs["short_put"]["strike"] - legs["long_put"]["strike"],
        )
        expiry_debit = round(call_debit + put_debit, 2)
        quote_times_valid = all(
            leg.get("quote_age_seconds") is not None
            and 0 <= leg["quote_age_seconds"] <= 3
            and leg.get("trade_age_seconds", float("inf")) <= 3
            for leg in legs.values()
        )
        rows.append(
            {
                "date": date,
                "entry_et": signals[date]["entry_et"],
                "short_call": legs["short_call"]["strike"],
                "long_call": legs["long_call"]["strike"],
                "short_put": legs["short_put"]["strike"],
                "long_put": legs["long_put"]["strike"],
                "settlement": settlement,
                "observed_nbbo_credit": quote_credit,
                "expiry_debit": expiry_debit,
                "indicative_pnl_before_fees": round(quote_credit - expiry_debit, 2),
                "all_quote_times_within_3s": quote_times_valid,
                "asof_utc": report["quote_report"].get("asof_utc"),
            }
        )

    if not rows:
        raise SystemExit("No complete four-leg NBBO snapshots found")
    output = ROOT / "results/uw_first_hour_condor_5pt.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['date']}: quoted credit {row['observed_nbbo_credit']:.2f}, "
            f"expiry debit {row['expiry_debit']:.2f}, indicative P&L {row['indicative_pnl_before_fees']:+.2f}, "
            f"quote times verified {row['all_quote_times_within_3s']}"
        )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
