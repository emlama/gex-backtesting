# SPX Options Backtesting

Local SPXW options research using SPX one-minute indicators, Unusual Whales
option tape, and official expiry settlements. The offline batch runner tests
0DTE/1DTE put and call credit verticals plus iron condors across signal and
spread parameters, then creates a standalone chart dashboard. Earlier
price-only expiry-condition studies remain available for comparison.

## Data

- SPX one-minute OHLC: public [spx-0dte-archive](https://github.com/Alexduanran/spx-0dte-archive).
  The current local sample covers 2026-07-17 through 2026-10-23 (70 sessions).
- SPXW PM settlement: Cboe's [weekly](https://www.cboe.com/index_settlement_values/weeklys_settlement_values/)
  and [month-end](https://www.cboe.com/index_settlement_values/end_of_month_settlement_values/) values.
- Generated files under `data/` and `results/` are local and excluded from Git.

## Reproduce

Install Python 3.10+ dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
```

Fetch the source bars and build EMA 9/20/30/50/200, Bollinger Bands 20/2,
and MACD 12/26/9:

```bash
git clone --depth 1 https://github.com/Alexduanran/spx-0dte-archive.git /tmp/spx-0dte-archive
.venv/bin/python scripts/build_spx_minute_indicators.py \
  --source-dir /tmp/spx-0dte-archive/spx-1m
.venv/bin/python scripts/fetch_spxw_settlements.py --year 2026
```

Run the expiry tests:

```bash
.venv/bin/python scripts/backtest_opening_range_expiry.py
.venv/bin/python scripts/backtest_opening_range_ema_expiry.py
.venv/bin/python scripts/backtest_first_hour_ema_open_put.py
.venv/bin/python scripts/backtest_first_hour_ema_open_call.py
.venv/bin/python scripts/backtest_double_breakout_first_hour.py
```

The outputs are `results/opening_range_expiry_proxy.csv`,
`results/opening_range_ema30_expiry.csv`, and
`results/first_hour_ema_open_put.csv` and
`results/first_hour_ema_open_call.csv`, plus
`results/double_breakout_first_hour.csv`. A dated summary table is in
[`results/RESULTS.md`](results/RESULTS.md).

## Unusual Whales historical-data probe

Enter your API key locally through a hidden terminal prompt; the resulting
`.env` is owner-readable and ignored by Git. Do not paste the key into a chat,
command argument, or committed file. If `UW_API_KEY` is already in your
environment, the setup step is unnecessary.

```bash
.venv/bin/python scripts/setup_uw_key.py
.venv/bin/python scripts/probe_uw_history.py --date 2026-10-22 --width 5
```

The probe makes one historical option-chain request and four contract-intraday
requests for the most recent qualifying two-sided first-hour breakout. It
maps theoretical strikes to same-day **SPXW** contracts and reports whether
all four legs had trade bars near the 10:30 ET entry. Historical access depends
on the API subscription. The [option-chain endpoint](https://api.unusualwhales.com/docs/api/stock/option-chains.md)
identifies contracts present on a past date; the [contract-intraday endpoint](https://api.unusualwhales.com/docs/api/option-contract/intraday-data.md)
contains one-minute aggregates of executed trades. Those bars do not establish
simultaneous executable bid/ask prices, so the probe does not calculate spread
fills or P&L. If coverage is sufficient, the next step is to verify whether
historical option-trade NBBO snapshots or another quote source can reconstruct
conservative four-leg entry credits.

## Signal and result definition

The 09:30–09:59 ET bars establish the opening high and low. Starting at 10:00,
the first one-minute **close** above the high signals a theoretical short put
at the opening low; the first close below the low signals a theoretical short
call at the opening high. The EMA variant keeps only an up signal above EMA30
or a down signal below EMA30. EMA30 continues across sessions and uses only
bars known at signal time. `known_at_et` gives the earliest time the signal
bar's close is usable.

The result is whether the theoretical strike has **zero intrinsic value at
expiry**, based on official SPXW PM settlement. 1DTE means the next trading
session, including Friday to Monday when Monday is open. The scripts exclude
incomplete minute days and unavailable settlement dates. The EMA output keeps
the first breakout per direction and marks `first_aligned_breakout_of_day` for
the one-signal-per-day comparison.

Theoretical strikes from exact SPX values may not be listed option strikes.
The price-only expiry files contain no historical option quotes, premiums,
fills, fees, or realized P&L, so expiry-zero rates are not trading win rates.

The optional UW quote probe (`scripts/inspect_uw_tape_window.py`) reads only
the start of a daily full-tape ZIP until it finds trade-time NBBO snapshots for
all four first-hour iron-condor legs. A full ZIP can be multiple gigabytes; the
script defaults to a 768 MiB download cap. The four tested dates need roughly
400–550 MB each. Run `scripts/summarize_uw_condors.py` to create
`results/uw_first_hour_condor_5pt.csv`. The August snapshots have no NBBO
timestamps; all four-leg credit calculations remain indicative and do not
establish executable complex-order fills.

For repeated 0DTE tests, cache a full-tape day once as an SPXW-only Parquet file:

```bash
.venv/bin/python scripts/cache_uw_spxw_0dte.py 2026-10-22
.venv/bin/python scripts/inspect_uw_tape_window.py --date 2026-10-22 --width 5
.venv/bin/python scripts/summarize_uw_condors.py
```

The cache builder resumes interrupted ZIP downloads, filters to same-day SPXW
contracts, checks the Parquet row count, and removes the source ZIP after a
successful conversion. Use `--keep-zip` to retain it. The quote inspector
automatically uses a cached day; `--force-api` bypasses it. On 2026-10-22, a
2.12 GB ZIP containing 12.95 million market trades became a 61.2 MiB Parquet
file containing 1.60 million SPXW 0DTE trades in 140 seconds. The cached
four-leg inspection then ran in about 0.24 seconds and reproduced the 2.65
point indicative credit. These are one-day measurements on this machine, not
historical backtest performance or confirmed trade fills. The cache currently
covers 0DTE only; a 1DTE test needs next-session expiry contracts too.

The [GitHub tool comparison](research/uw_spx_backtesting_tools.md) records why
the reviewed backtest engines still require a UW data adapter and why a local
Parquet candidate-trade table is the next step for large parameter sweeps.

## OSBT integration pilot

Install the pinned [OSBT](https://github.com/dremg/osbt-public) version and run
the one-day 0DTE first-hour condor pilot after caching the UW tape day:

```bash
.venv/bin/python -m pip install -e '.[osbt]'
.venv/bin/python scripts/cache_uw_spxw_0dte.py 2026-10-22
.venv/bin/python scripts/run_osbt_uw_pilot.py --date 2026-10-22
```

The [pilot adapter](scripts/run_osbt_uw_pilot.py) exports a 10:30 ET option
chain in OSBT's expected Parquet layout. It accepts only uncanceled trades
whose bid and ask are positive, uncrossed, timestamped, and no more than three
seconds old at 10:30:03 ET. It joins the last fully known SPX one-minute close,
selects the precomputed opening-range strikes, and runs OSBT's position,
slippage, commission, and trade accounting. A small OSBT subclass supplies
official SPXW PM settlement for the expiry exit. The pilot uses a fixed
10:30 entry, 0DTE, a five-point iron condor, and no intraday stop.

For 2026-10-22, OSBT found one trade: 2.55 points of conservative four-leg
bid/ask credit, zero expiry debit, and $249.80 modeled P&L after $0.65 per
contract per side in OSBT's fee model. The output is
`results/osbt_uw_pilot_2026-10-22.json`. The four bid/ask snapshots are from
separate option trades and are not a verified executable complex-order fill.
This is a successful engine/data integration check, not a multi-day win rate.
The bulk runner below is the main route for repeated tests; broader data
coverage and quote/fill validation remain necessary.

## Local bulk backtests and charts

The batch runner works entirely on local files: no chat or AI tokens are used
while it runs. The initial UW tape downloads still use your UW API access.
Edit [`configs/spxw_bulk.json`](configs/spxw_bulk.json) to choose signals,
EMA fast/slow pairs, date range, 0DTE/1DTE, spread widths, strike offsets,
quote age, and fees. Then run:

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/run_bulk_spxw.py
```

On macOS, you can also double-click [`run_backtests.command`](run_backtests.command)
after the environment is installed. It runs the same local batch job and opens
the resulting dashboard in your browser.

Results appear in `results/bulk/latest/`: `dashboard.html` is a standalone
dashboard that opens in a browser without a server or network access. It
filters variants by signal, expiry, and width, ranks them, and shows each
variant's cumulative P&L, drawdown, win/loss streaks, trade log, leg quotes,
expiry payoff, offset grid, and quote coverage (light or graphite theme). Its
template is `scripts/bulk_dashboard.html`. `candidates.parquet`
contains every accepted or rejected strategy/date combination; `summary.csv`
contains per-variant P&L, profitable rate, quote coverage, maximum drawdown and
its dates, longest win/loss streaks, streak totals, and profit factor;
`coverage.csv` shows missing UW cache or settlement dates; `manifest.json`
records the configuration and run counts. The runner currently supports
first-hour EMA put/call verticals and the two-sided first-hour breakout iron
condor. All legs use conservative bid/ask prices from UW trade-time snapshots,
subject to quote timestamp and age checks. Expiry payoff uses official SPXW
PM settlement and width-capped spread loss. The dashboard labels the outcome
as a modeled result because separate NBBO snapshots do not prove an executable
multi-leg fill.

To build caches for a range of SPX sessions already present in the minute file:

```bash
.venv/bin/python scripts/cache_uw_spxw_range.py \
  --start 2026-10-01 --end 2026-10-22 --include-next-expiry
```

The range command skips completed Parquet days and resumes interrupted ZIP
downloads. `--include-next-expiry` stores both same-day and next-session SPXW
contracts, allowing the configured 1DTE runs. The UW ZIPs are large and the
first conversion remains the slow step. At present only 2026-10-22 has a local
UW cache for both expiries. With the default three-second quote-age limit,
1DTE entry legs on that day are incomplete and are rejected; a 60-second
sensitivity run finds them, but those older snapshots do not demonstrate an
executable 10:30 fill. The example dashboard is a pipeline check, not a
meaningful strategy ranking. Use `coverage.csv` and a much larger date sample
before interpreting performance statistics.

The first-hour EMA test enters at 10:30 ET only when EMA30 > EMA50 at every
09:30–10:29 minute. Its theoretical put strike is that day's 09:30 SPX open.
The inverse test sells a theoretical call at the same strike when EMA50 > EMA30
at every minute in that first hour.

## Archived GEX research

The earlier gamma-exposure code, notebooks, data-download scripts, and audit
are preserved in [`archive/gex/`](archive/gex/README.md). Its original
513-day enriched options-trade dataset is unavailable locally; that archive
is not part of the active SPX price-only workflow.
