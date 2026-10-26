# SPX expiry-condition backtest results

Snapshot: 2026-10-25. Source sample: 70 SPX one-minute sessions from
2026-07-17 through 2026-10-23. The calculations use complete 390-bar days and
published Cboe SPXW PM settlement values. The local per-signal CSVs are
`opening_range_expiry_proxy.csv`, `opening_range_ema30_expiry.csv`, and
`first_hour_ema_open_put.csv`, and `first_hour_ema_open_call.csv`.

## Main result: first signal of each day

| Rule | Expiry | Zero intrinsic value | Rate |
| --- | --- | ---: | ---: |
| Opening-range breakout | 0DTE | 51 / 62 | 82.3% |
| Opening-range breakout + EMA30 direction | 0DTE | 50 / 61 | 82.0% |
| Opening-range breakout + EMA30 direction | 1DTE | 46 / 62 | 74.2% |

The 0DTE and 1DTE EMA rows have different eligible dates. On the **same 58
signals** with both settlements available, 0DTE is **47 / 58 (81.0%)** and
1DTE is **43 / 58 (74.1%)**. Use this paired comparison when comparing expiry
choices. The sample is too small to treat the difference as a reliable edge.

## Additional breakdown

| EMA30-aligned signal set | Expiry | Zero intrinsic value | Rate |
| --- | --- | ---: | ---: |
| Daily first signal, up (short put) | 0DTE | 24 / 30 | 80.0% |
| Daily first signal, up (short put) | 1DTE | 21 / 30 | 70.0% |
| Daily first signal, down (short call) | 0DTE | 26 / 31 | 83.9% |
| Daily first signal, down (short call) | 1DTE | 25 / 32 | 78.1% |
| First breakout on each side (up to two per day) | 0DTE | 68 / 84 | 81.0% |
| First breakout on each side (up to two per day) | 1DTE | 56 / 84 | 66.7% |

Of 90 first breakouts by direction, 89 already agreed with the continuous
one-minute EMA30. The EMA filter therefore removed only one breakout. The
baseline and EMA rows have different eligible samples and should not be read
as a measured improvement from adding EMA.

## Definition and limits

- Opening range: 09:30–09:59 America/New_York high and low.
- Up signal: first one-minute close after 10:00 above the range high; theoretical
  short put strike is the range low. Down signal: first close below the range
  low; theoretical short call strike is the range high.
- EMA filter: up close above EMA30 or down close below EMA30. EMA30 continues
  across sessions; the signal is known after its one-minute bar closes.
- 1DTE: next trading session's SPXW PM expiry, including Friday to Monday when
  Monday is open. Missing settlements are excluded from that expiry's count.
- A success means the option's intrinsic value is zero at official settlement.
  Exact range highs and lows may not be listed strikes. These figures contain
  no option premium, fill, spread, fees, or realized P&L.

Sources: [SPX minute archive](https://github.com/Alexduanran/spx-0dte-archive),
[Cboe SPXW weekly settlement](https://www.cboe.com/index_settlement_values/weeklys_settlement_values/),
and [Cboe SPXW month-end settlement](https://www.cboe.com/index_settlement_values/end_of_month_settlement_values/).

## First-hour EMA alignment: short at the SPX open

Run on 2026-10-26 from `scripts/backtest_first_hour_ema_open_put.py` and
`scripts/backtest_first_hour_ema_open_call.py`. A signal requires **every**
09:30–10:29 ET one-minute bar to satisfy the indicated EMA ordering. The EMAs
continue across trading sessions. The condition is fully known at 10:30 ET,
when the theoretical position is entered at a strike equal to that day's
09:30 SPX opening price; one position is tested per qualifying day.

| First-hour condition | Short option | Expiry | Zero intrinsic value | Rate |
| --- | --- | --- | ---: | ---: |
| EMA30 > EMA50 throughout | Put | 0DTE | 11 / 16 | 68.8% |
| EMA30 > EMA50 throughout | Put | 1DTE | 12 / 15 | 80.0% |
| EMA50 > EMA30 throughout | Call | 0DTE | 14 / 17 | 82.4% |
| EMA50 > EMA30 throughout | Call | 1DTE | 10 / 16 | 62.5% |
| No EMA filter (reference) | Put | 0DTE | 28 / 63 | 44.4% |
| No EMA filter (reference) | Put | 1DTE | 29 / 63 | 46.0% |
| No EMA filter (reference) | Call | 0DTE | 35 / 63 | 55.6% |
| No EMA filter (reference) | Call | 1DTE | 34 / 63 | 54.0% |

There are 67 complete SPX sessions. The Put rule qualifies on 16 days; the
Call rule qualifies on 17 days. For the **same eligible dates** within each
rule, Put is 10 / 15 (0DTE) versus 12 / 15 (1DTE), and Call is 13 / 16
(0DTE) versus 10 / 16 (1DTE). The missing 1DTE settlement is 2026-09-18
for the Put signal on 2026-09-17, and 2026-08-21 for the Call signal on
2026-08-20. At 10:30, three qualifying Puts and four qualifying Calls would
already be in the money. Rounding the theoretical strike outward to a
hypothetical 5-point increment does not change these counts.

The samples are small: the 0DTE 95% Wilson intervals are approximately
44.4%–85.8% for Put and 59.0%–93.8% for Call. These are strike-expiry
condition tests, without option premiums, executable quotes, fills, fees, or
realized P&L. Detailed local files are `first_hour_ema_open_put.csv` and
`first_hour_ema_open_call.csv`.

## Exploratory priority (2026-10-26)

The leading candidate for a historical premium and fill test is the **0DTE
opening-range breakout rule**, especially its downward-breakout short-call
subset. These are in-sample comparisons across partly overlapping signals;
they do not establish a trading edge.

| Candidate | Zero intrinsic at expiry | Mean intrinsic owed at expiry | Largest intrinsic owed |
| --- | ---: | ---: | ---: |
| Opening-range breakout, first/day, either side, 0DTE | 51 / 62 (82.3%) | 2.87 SPX points | 45.80 |
| Its down-breakout short-call subset, 0DTE | 27 / 32 (84.4%) | 1.48 SPX points | 18.47 |
| First-hour EMA50 > EMA30, short call, 0DTE | 14 / 17 (82.4%) | 3.20 SPX points | 37.14 |
| First-hour EMA30 > EMA50, short put, 1DTE | 12 / 15 (80.0%) | 4.19 SPX points | 23.54 |

Mean intrinsic owed is the average of `max(strike - settlement, 0)` for puts
or `max(settlement - strike, 0)` for calls, including zero-loss expiries.
Before fees, average option credit would have to exceed that value for a
positive average expiry P&L in this sample. No historical option entry credit
has been measured, and the exact theoretical strikes may not be listed.

## Vertical credit-spread sensitivity (2026-10-26)

The earlier intrinsic-value comparison describes an unhedged short option.
For a put credit spread, buy a put `width` points below the short strike; for
a call credit spread, buy a call `width` points above it. The spread's expiry
debit is `min(short-option intrinsic value, width)`. These examples use the
same theoretical first-breakout-of-day short strikes and settlements, with a
fixed width in SPX points; no actual listed option series or entry quotes have
been checked.

| First-breakout direction | Expiry | Width | Zero expiry debit | Mean expiry debit | Full-width debit |
| --- | --- | ---: | ---: | ---: | ---: |
| Up, short put spread | 0DTE | 5 | 24 / 30 (80.0%) | 0.96 points | 5 / 30 |
| Down, short call spread | 0DTE | 5 | 27 / 32 (84.4%) | 0.63 points | 4 / 32 |
| Up, short put spread | 0DTE | 10 | 24 / 30 (80.0%) | 1.67 points | 4 / 30 |
| Down, short call spread | 0DTE | 10 | 27 / 32 (84.4%) | 1.21 points | 2 / 32 |
| Up, short put spread | 0DTE | 20 | 24 / 30 (80.0%) | 2.84 points | 3 / 30 |
| Down, short call spread | 0DTE | 20 | 27 / 32 (84.4%) | 1.48 points | 0 / 32 |

At expiry, per-spread P&L before fees is **entry net credit minus expiry
debit**; the maximum loss is **width minus entry net credit**. The mean expiry
debit is therefore the sample's break-even *average* credit before costs, not
a measured price or profit. Put and call spreads may collect different credits,
so the lower debit alone cannot rank their actual returns. Both short and long
theoretical strikes also need mapping to listed SPXW strikes.

## Both opening-range breakouts within the first hour (2026-10-26)

The 09:30–09:59 ET high and low define the opening range. A qualifying day
has **both** a one-minute close above that high and a one-minute close below
that low during 10:00–10:29. At 10:30, short the 0DTE call at the 09:30–10:29
high and the 0DTE put at the 09:30–10:29 low. With equal-width long wings,
this is an iron condor. The entire signal and both theoretical strikes are
known before hypothetical entry. Of 70 source days, 63 have complete minute
data and a published same-day SPXW PM settlement; just four qualify.

| Date | First-hour call strike | First-hour put strike | SPXW PM settlement | Call zero | Put zero | Both zero | Both 5-point spreads' expiry debit |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| 2026-08-07 | 7756.08 | 7719.19 | 7757.64 | No | Yes | No | 1.56 points |
| 2026-08-28 | 7748.25 | 7715.87 | 7711.76 | Yes | No | No | 4.11 points |
| 2026-10-19 | 7716.76 | 7684.79 | 7690.73 | Yes | Yes | Yes | 0 |
| 2026-10-22 | 7702.94 | 7657.51 | 7663.43 | Yes | Yes | Yes | 0 |

The short call expires worthless on **3/4 (75%)**, the short put on **3/4
(75%)**, and both together on **2/4 (50%)** at the exact theoretical strikes.
Mean combined expiry debit for 5-, 10-, or 20-point wings is **1.42 SPX
points** per qualifying day; none
of the observed in-the-money short strikes went more than 5 points in the
money. Under a hypothetical outward rounding of both short strikes to
5-point increments, both would expire worthless on **3/4** days, showing how
sensitive this tiny sample is to strike selection. The later Unusual Whales
analysis below maps the strikes to listed contracts and inspects trade-time
NBBO snapshots. The per-day price-only output is
`double_breakout_first_hour.csv`; the reproducible calculation is in
`scripts/backtest_double_breakout_first_hour.py`.

## Unusual Whales 0DTE four-leg quote probe (2026-10-26)

For all four qualifying days above, the historical SPX option chain contained
same-day PM-settled SPXW contracts. Short call strikes were rounded outward to
the next listed strike at or above the first-hour high; short puts to the next
listed strike at or below the first-hour low. Each long wing is 5 points farther
out. Historical full-tape option trades include leg NBBO snapshots, so the
table below uses the short legs' bids and long legs' asks near 10:30 ET. This
is an **indicative quote calculation**, not a confirmed four-leg fill. The
full-tape CSV was not strictly sorted by execution time; scanning had to
continue until each leg had a post-entry print. Later records with earlier
execution times could alter these initial snapshots.

| Date | Short Call / long Call | Short Put / long Put | Combined quoted credit | Expiry debit | Indicative P&L before fees | Quote timestamps checked |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-08-07 | 7760 / 7765 | 7715 / 7710 | 2.40 | 0.00 | +2.40 | No |
| 2026-08-28 | 7750 / 7755 | 7715 / 7710 | 2.40 | 3.24 | -0.84 | No |
| 2026-10-19 | 7720 / 7725 | 7680 / 7675 | 2.10 | 0.00 | +2.10 | Yes, all within 3 s |
| 2026-10-22 | 7705 / 7710 | 7655 / 7650 | 2.65 | 0.00 | +2.65 | Yes, all within 3 s |

At these **listed** short strikes, both sides expire with zero intrinsic value
on **3/4** days. The arithmetic quote proxy is positive on **3/4** days and
averages **+1.58 SPX points per trade before fees**. The sample is only four
signals. For both August dates, the NBBO values are present in the tape but
their quote timestamps are missing, so quote freshness cannot be checked.
Even for October, four-leg NBBOs sampled from separate trades do not prove a
simultaneous complex-order fill. Commissions, exchange fees, partial fills,
quote size and slippage are not included. These numbers are therefore **not a
verified realized trading win rate or expected return**. The per-trade table
is `uw_first_hour_condor_5pt.csv`; local snapshots are in ignored `data/uw/`.

## Offline multi-leg batch pipeline check (2026-10-26)

The local batch runner now reads cached UW 0DTE and next-session SPXW trades,
generates EMA and first-hour breakout signals, sweeps widths and strike offsets,
calculates capped expiry P&L, and creates an offline dashboard. Only the
2026-10-22 UW tape has been cached, so these numbers verify the data pipeline
and arithmetic rather than any strategy success rate.

| Date | Expiry | Entry quote age limit | Candidate combinations | Accepted | Finding |
| --- | --- | ---: | ---: | ---: | --- |
| 2026-10-22 | 0DTE | 3 seconds | 36 | 36 | Quote-based model runs |
| 2026-10-22 | 1DTE | 3 seconds | 36 | 0 | Required legs lack fresh trade-time NBBOs |

For the 5-point first-hour iron condor with unshifted strikes 7655/7650 puts
and 7705/7710 calls, the strict 10:30:03 ET bid/ask snapshot yields 2.55
points of entry credit, zero expiry debit, and **$249.80 modeled P&L** after
the configured $0.65 per contract per side fee. The independent OSBT pilot
returns the same amount. The earlier 2.65-point quote probe above used a
different as-of time and snapshot selection; it should not be mixed with this
strict 10:30:03 rule. Allowing 60-second-old quotes makes the 1DTE legs
available on this day, but that sensitivity result is unsuitable as evidence
of executable entry pricing. Batch details and charts are under
`results/bulk/latest/` and can be regenerated without chat calls.
