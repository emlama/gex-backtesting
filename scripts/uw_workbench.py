"""Local SPXW backtest workbench: fetch UW data and run the offline engine.

The API key stays on the server side in the project's ignored .env file.
Only loopback connections are accepted. No Codex or LLM service is involved.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import webbrowser
from datetime import date, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from probe_uw_history import ROOT, api_key, get_json
from run_bulk_spxw import cache_path, expiry_lookup, signal_events

PAGE = Path(__file__).with_name("uw_workbench.html").read_bytes()
RUNS = ROOT / "results/bulk/workbench"
DOWNLOAD_SCRIPT = Path(__file__).with_name("cache_uw_spxw_0dte.py")
BACKTEST_SCRIPT = Path(__file__).with_name("run_bulk_spxw.py")
RUN_FILE = re.compile(
    r"^/runs/(?P<run>[0-9]{8}T[0-9]{6}-[a-f0-9]{6})/"
    r"(?P<file>dashboard\.html|summary\.csv|coverage\.csv|manifest\.json|candidates\.parquet)$"
)


def available_days() -> list[str]:
    source = ROOT / "data/spx_1m_indicators.parquet"
    if not source.exists():
        raise FileNotFoundError("缺少 SPX 分鐘資料，請先按 README 建立指標檔。")
    return sorted(pd.read_parquet(source, columns=["date"])["date"].unique().tolist())


def parse_number_list(value: object, label: str, *, positive: bool = False) -> list[int]:
    if not isinstance(value, str):
        raise ValueError(f"{label} 必須是以逗號分隔的整數。")
    items = [part.strip() for part in value.split(",")]
    if not items or any(not re.fullmatch(r"-?\d+", part) for part in items):
        raise ValueError(f"{label} 必須是以逗號分隔的整數。")
    numbers = list(dict.fromkeys(int(part) for part in items))
    if positive and any(number <= 0 for number in numbers):
        raise ValueError(f"{label} 必須大於零。")
    if len(numbers) > 20:
        raise ValueError(f"{label} 最多 20 個數值。")
    return numbers


def parse_config(payload: dict) -> dict:
    days = available_days()
    start = payload.get("start_date")
    end = payload.get("end_date")
    try:
        if not isinstance(start, str) or not isinstance(end, str):
            raise ValueError
        date.fromisoformat(start)
        date.fromisoformat(end)
    except ValueError:
        raise ValueError("請輸入有效的開始與結束日期。") from None
    if start > end:
        raise ValueError("開始日期不能晚於結束日期。")
    if start < days[0] or end > days[-1]:
        raise ValueError(f"目前 SPX 分鐘資料只涵蓋 {days[0]} 至 {days[-1]}。")
    signals = payload.get("signals")
    if not isinstance(signals, list) or not signals or set(signals) - {"double_breakout", "ema_first_hour"}:
        raise ValueError("請至少選一個目前支援的訊號。")
    dtes = payload.get("dtes")
    if not isinstance(dtes, list) or not dtes or set(dtes) - {0, 1}:
        raise ValueError("請選擇 0DTE 或 1DTE。")
    try:
        max_age = int(payload.get("max_quote_age_seconds", 3))
        fee = float(payload.get("commission_per_contract_per_side", 0.65))
    except (TypeError, ValueError):
        raise ValueError("報價年齡與手續費必須是數字。") from None
    if not 1 <= max_age <= 60:
        raise ValueError("報價最大年齡須為 1 至 60 秒。")
    if not 0 <= fee <= 10:
        raise ValueError("每口每邊手續費須為 $0 至 $10。")
    return {
        "signals": list(dict.fromkeys(signals)),
        "ema_pairs": [[30, 50]],
        "start_date": start,
        "end_date": end,
        "dtes": sorted(set(dtes)),
        "widths": parse_number_list(payload.get("widths", "5,10,20"), "價差寬度", positive=True),
        "put_offsets": parse_number_list(payload.get("put_offsets", "-10,-5,0"), "Put 偏移"),
        "call_offsets": parse_number_list(payload.get("call_offsets", "0,5,10"), "Call 偏移"),
        "quote_delay_seconds": 3,
        "max_quote_age_seconds": max_age,
        "commission_per_contract_per_side": fee,
    }


def plan_run(config: dict) -> dict:
    events = [
        event for event in signal_events(config)
        if config["start_date"] <= event["date"] <= config["end_date"]
    ]
    next_session, settlements = expiry_lookup()
    needed = set()
    eligible = 0
    for event in events:
        for dte in config["dtes"]:
            expiry = event["date"] if dte == 0 else next_session.get(event["date"])
            if not expiry or expiry not in settlements:
                continue
            eligible += 1
            if cache_path(event["date"], dte) is None:
                needed.add(event["date"])
    return {"signal_events": len(events), "eligible_signal_expiries": eligible, "missing_uw_days": sorted(needed)}


class RunState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.phase = "idle"
        self.run_id: str | None = None
        self.message = "尚未執行。"
        self.logs: list[str] = []
        self.result_url: str | None = None

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "phase": self.phase,
                "run_id": self.run_id,
                "message": self.message,
                "logs": self.logs[-120:],
                "result_url": self.result_url,
            }

    def log(self, message: str) -> None:
        with self.lock:
            self.logs.append(message.rstrip())
            self.message = message.rstrip()


STATE = RunState()
CSRF_TOKEN = secrets.token_urlsafe(32)


def run_process(command: list[str]) -> None:
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          bufsize=1) as process:
        assert process.stdout is not None
        for line in process.stdout:
            STATE.log(line)
        code = process.wait()
    if code:
        raise RuntimeError(f"處理程序以狀態 {code} 結束；請查看上方紀錄。")


def worker(run_id: str, config: dict, plan: dict, download_missing: bool) -> None:
    try:
        folder = RUNS / run_id
        folder.mkdir(parents=True, exist_ok=False)
        config["output_dir"] = str(folder.relative_to(ROOT))
        config_path = folder / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        missing = plan["missing_uw_days"]
        if missing and not download_missing:
            STATE.log(f"有 {len(missing)} 個訊號日沒有本機 UW 快取；本次會保留這些日期為缺資料。")
        for index, day in enumerate(missing if download_missing else [], 1):
            with STATE.lock:
                STATE.phase = "downloading"
            STATE.log(f"[{index}/{len(missing)}] 從 UW 下載並整理 {day} 的 SPXW 資料。")
            command = [sys.executable, "-u", str(DOWNLOAD_SCRIPT), day]
            if 1 in config["dtes"]:
                command.append("--include-next-expiry")
            run_process(command)
        with STATE.lock:
            STATE.phase = "backtesting"
        STATE.log("執行本機回測並建立儀表板。")
        run_process([sys.executable, "-u", str(BACKTEST_SCRIPT), "--config", str(config_path)])
        with STATE.lock:
            STATE.phase = "complete"
            STATE.message = "完成。結果已儲存在本機。"
            STATE.result_url = f"/runs/{run_id}/dashboard.html"
    except Exception as exc:
        with STATE.lock:
            STATE.phase = "error"
            STATE.message = str(exc)
            STATE.logs.append(f"錯誤：{exc}")


class Handler(BaseHTTPRequestHandler):
    server_version = "SPXWWorkbench/1.0"

    def log_message(self, format: str, *args: object) -> None:
        # Keep routine status polling out of the terminal.
        if not self.path.startswith("/api/status"):
            super().log_message(format, *args)

    def reply(self, status: HTTPStatus, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; img-src 'self' data:",
        )
        self.end_headers()
        self.wfile.write(data)

    def json_reply(self, status: HTTPStatus, value: dict) -> None:
        self.reply(status, json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            page = PAGE.replace(b"__CSRF_TOKEN__", CSRF_TOKEN.encode())
            self.reply(HTTPStatus.OK, page, "text/html; charset=utf-8")
        elif path == "/api/info":
            try:
                days = available_days()
                cached = {
                    path.stem
                    for folder in ("spxw_0_1dte", "spxw_0dte")
                    for path in (ROOT / "data/uw" / folder).glob("*.parquet")
                }
                self.json_reply(HTTPStatus.OK, {
                    "first_date": days[0], "last_date": days[-1],
                    "default_date": max(cached) if cached else days[-2],
                    "sessions": len(days), "cached_days": len(cached),
                    "key_present": (ROOT / ".env").exists() or bool(os.environ.get("UW_API_KEY")),
                })
            except Exception as exc:
                self.json_reply(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
        elif path == "/api/status":
            self.json_reply(HTTPStatus.OK, STATE.snapshot())
        elif (match := RUN_FILE.fullmatch(path)) is not None:
            target = RUNS / match["run"] / match["file"]
            if not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            mime = "text/html; charset=utf-8" if target.suffix == ".html" else (
                "application/json" if target.suffix == ".json" else (
                    "text/csv; charset=utf-8" if target.suffix == ".csv" else "application/octet-stream"
                )
            )
            self.reply(HTTPStatus.OK, target.read_bytes(), mime)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            self.json_reply(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "只接受 JSON 請求。"})
            return
        origin = self.headers.get("Origin")
        expected = f"http://127.0.0.1:{self.server.server_port}"
        if origin != expected or self.headers.get("X-Workbench-Token") != CSRF_TOKEN:
            self.json_reply(HTTPStatus.FORBIDDEN, {"error": "請從本機工作台操作。"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 50_000:
                raise ValueError("請求大小無效。")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("請求格式無效。")
            if self.path == "/api/check":
                day = available_days()[-2]
                symbols = get_json("/api/stock/SPX/option-chains", api_key(), {"date": day}).get("data", [])
                message = f"UW API 已連接；{day} 的 SPX 期權鏈有 {len(symbols):,} 個合約。"
                self.json_reply(HTTPStatus.OK, {"message": message})
                return
            config = parse_config(payload)
            plan = plan_run(config)
            if self.path == "/api/preview":
                self.json_reply(HTTPStatus.OK, plan)
                return
            if self.path != "/api/run":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            download = payload.get("download_missing") is True
            missing = plan["missing_uw_days"]
            if download and missing and not ((ROOT / ".env").exists() or os.environ.get("UW_API_KEY")):
                api_key()  # Raises a setup instruction without exposing the key.
            if download and len(missing) > 5 and payload.get("allow_large_download") is not True:
                raise ValueError(
                    f"這次需要下載 {len(missing)} 天 UW full-tape（每個 ZIP 通常約 2 GB）。"
                    "請先勾選大量下載，或縮窄日期。"
                )
            with STATE.lock:
                if STATE.phase in {"preparing", "downloading", "backtesting"}:
                    self.json_reply(HTTPStatus.CONFLICT, {"error": "已有回測正在進行。"})
                    return
                run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(3)
                STATE.phase = "preparing"
                STATE.run_id = run_id
                STATE.message = "準備執行。"
                STATE.logs = []
                STATE.result_url = None
            threading.Thread(target=worker, args=(run_id, config, plan, download), daemon=True).start()
            self.json_reply(HTTPStatus.ACCEPTED, {"run_id": run_id, "plan": plan})
        except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
            self.json_reply(HTTPStatus.BAD_REQUEST, {"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0, help="Local port; 0 selects a free port")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically")
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"SPXW 本機工作台：{url}", flush=True)
        print("在此視窗按 Control-C 可關閉工作台。", flush=True)
        if not args.no_open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
