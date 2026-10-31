# SPX Options Backtesting

The active workflow is documented in `README.md`. It builds SPX one-minute
indicators and tests opening-range short-option strikes against official SPXW
PM settlement values. Active code lives in `scripts/`; generated market data
and results live in ignored `data/` and `results/`.

For signal or expiry changes, inspect the relevant script and rerun the
affected backtests. Preserve the distinction between zero intrinsic value at expiry
and realized option P&L. A signal bar's close is usable at `known_at_et`, one
minute after `bar_start_et`.

For GEX research, consult `archive/gex/CLAUDE.md` and `archive/gex/README.md`.
That archive requires an enriched options-trade dataset absent from this
workspace; it is outside the current SPX price-only workflow.
