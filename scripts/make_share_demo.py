"""Create a shareable UI demo using only invented option prices and outcomes.

This script deliberately never reads the UW cache, .env, or real backtest output.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from bulk_dashboard import write_dashboard
from run_bulk_spxw import evaluate, summarize

OUTPUT = Path.home() / "Desktop" / "SPXW_展示版_全部數據虛構.html"


def invented_quotes() -> dict:
    # These prices and strikes are invented solely to exercise the dashboard.
    premiums = {
        ("put", 1000): (2.8, 3.0), ("put", 995): (1.6, 1.8),
        ("put", 990): (0.8, 1.0), ("put", 985): (0.3, 0.5),
        ("call", 1100): (2.8, 3.0), ("call", 1105): (1.6, 1.8),
        ("call", 1110): (0.8, 1.0), ("call", 1115): (0.3, 0.5),
    }
    return {
        (kind, strike): {
            "symbol": "FICTIONAL", "bid": bid, "ask": ask,
            "trade_age_s": 1.0, "bid_age_s": 1.0, "ask_age_s": 1.0,
        }
        for (kind, strike), (bid, ask) in premiums.items()
    }


def build_demo() -> None:
    config = {
        "widths": [5, 10], "put_offsets": [-5, 0], "call_offsets": [0, 5],
        "commission_per_contract_per_side": 0.65,
    }
    quote_book = invented_quotes()
    rows, coverage = [], []
    dates = pd.bdate_range("2099-01-05", periods=12)
    signals = [
        ("ema_30_50_put", "put_vertical", 1000.0, None),
        ("ema_30_50_call", "call_vertical", None, 1100.0),
        ("double_breakout", "iron_condor", 1000.0, 1100.0),
    ]
    # Each sequence is intentionally made up, with both profitable and losing outcomes.
    outcomes = {
        "ema_30_50_put": [1008, 1004, 996, 1010, 998, 1006, 1003, 992, 1008, 1005, 997, 1012],
        "ema_30_50_call": [1090, 1096, 1104, 1092, 1101, 1095, 1098, 1108, 1093, 1097, 1102, 1090],
        "double_breakout": [1050, 1045, 998, 1060, 1102, 1038, 1055, 1110, 1040, 1070, 997, 1052],
    }
    for i, day in enumerate(dates):
        day_str = day.date().isoformat()
        for signal, structure, put_anchor, call_anchor in signals:
            event = {
                "signal": signal, "date": day_str, "entry_et": f"{day_str}T10:30:00-05:00",
                "structure": structure, "put_anchor": put_anchor, "call_anchor": call_anchor,
            }
            for dte in (0, 1):
                expiry = (day + pd.offsets.BDay(dte)).date().isoformat()
                settlement = outcomes[signal][i] + (1 if dte else 0)
                rows.extend(evaluate(event, dte, expiry, settlement, quote_book, config))
                coverage.append({"signal": signal, "date": day_str, "dte": dte, "expiry": expiry, "status": "ready"})

    frame = pd.DataFrame(rows)
    meta = {
        "run": "展示版_全部數據虛構", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dtes": [0, 1], "quote_delay_seconds": 0, "max_quote_age_seconds": 3,
        "commission_per_contract_per_side": config["commission_per_contract_per_side"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_dashboard(summarize(frame), frame, pd.DataFrame(coverage), meta, OUTPUT)
    page = OUTPUT.read_text(encoding="utf-8")
    page = page.replace("<title>SPXW 回測儀表板</title>", "<title>SPXW 展示版 · 全部數據虛構</title>")
    page = page.replace(
        '<div class="wrap">',
        '<div class="wrap"><div role="note" style="padding:14px 18px;border-radius:10px;'
        'background:#fff1c7;color:#533b00;font-weight:700;border:2px solid #d8a822">'
        '展示版 · 所有日期、價格、期權報價和回測結果均為虛構；只用來展示介面，不能用作交易判斷。'
        '</div>',
        1,
    )
    replacements = {
        "SPXW · MULTI-LEG": "SPXW · FICTIONAL DEMO",
        "探索性 · 樣本內": "虛構數據 · 介面示例",
        "UW 成交時 NBBO 快照僅供參考；分腿報價不證明複合單可成交。參數排名基於小樣本時屬探索性。":
            "全部報價及結果都是虛構示例；圖表只供展示操作。",
        "candidates.parquet · summary.csv · coverage.csv · manifest.json": "單一離線 HTML · 無 API 連線",
        "訊號 × 到期事件有本機 UW 快取及結算價；仍需通過報價檢查": "虛構訊號及虛構結算價；僅供介面展示",
        "缺少官方結算價": "缺少示例結算價",
        "腿 · UW NBBO 快照": "腿 · 虛構報價",
        "有 UW 報價": "有示例報價",
        "缺 UW 快取": "缺示例報價",
        "UW 快取": "示例報價",
    }
    for old, new in replacements.items():
        page = page.replace(old, new)
    OUTPUT.write_text(page, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    build_demo()
