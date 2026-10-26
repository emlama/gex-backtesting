"""Prompt for an Unusual Whales API token and store it in an ignored local .env."""

from __future__ import annotations

import getpass
import json
import os
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def normalize_key(pasted: str) -> str:
    """Accept either the raw token or a copied Authorization header."""
    key = pasted.replace("\ufeff", "").replace("\u200b", "").strip().strip("\"'").strip()
    key = re.sub(r"^(?:export\s+)?UW_API_KEY\s*=\s*", "", key, flags=re.IGNORECASE)
    if key.lower().startswith("authorization:"):
        key = key.partition(":")[2].strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key.strip().strip("\"'").strip()


def format_hint(key: str) -> str:
    """Report token shape without exposing any token characters."""
    return f"Received {len(key)} characters and {key.count('-')} hyphens; expected UUID shape 8-4-4-4-12"


def check_key(key: str) -> str:
    """Let the API validate the token; avoid assuming the token's format."""
    request = Request(
        "https://api.unusualwhales.com/api/stock/SPX/option-chains?date=2026-10-22",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=20):
            return "API accepted the token and historical-chain request"
    except HTTPError as exc:
        if exc.code == 401:
            try:
                reason = json.load(exc).get("reason", "invalid token")
            except (ValueError, AttributeError):
                reason = "invalid token"
            hint = f" {format_hint(key)}." if reason == "malformed_token" else ""
            raise ValueError(
                f"API rejected the token ({reason}).{hint} Copy only the API token from "
                "unusualwhales.com/dashboard/api; nothing was saved"
            ) from None
        if exc.code == 403:
            return "API accepted the token, but historical-chain access returned 403; check subscription access"
        raise ValueError(f"API check returned HTTP {exc.code}; nothing was saved") from None
    except URLError as exc:
        raise ValueError(f"Could not reach the API ({exc.reason}); nothing was saved") from None


def main() -> None:
    target = Path(__file__).resolve().parents[1] / ".env"
    if target.exists():
        raise SystemExit(f"{target} already exists; set UW_API_KEY there or in your environment")

    key = normalize_key(getpass.getpass("Unusual Whales API key (input hidden): "))
    if not key:
        raise SystemExit("No API key was entered; nothing was saved")
    try:
        status = check_key(key)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    old_umask = os.umask(0o077)
    try:
        with target.open("x", encoding="utf-8") as file:
            file.write(f"UW_API_KEY={key}\n")
        target.chmod(0o600)
    finally:
        os.umask(old_umask)
    print(status)
    print(f"Saved the key to ignored, owner-readable {target}")


if __name__ == "__main__":
    main()
