# Chandelier-Momentum Trading System


Long-only momentum / trend-following system for equities. Consists of two scripts:

- **`backtest.py`** — simulates the strategy on historical OHLCV data and produces performance reports
- **`screener.py`** — scans the live S&P 500 universe for today's setups and calculates position sizes

---

## Requirements

```
pandas
numpy
yfinance
tqdm
```

Install with:
```bash
pip install pandas numpy yfinance tqdm
```

---

## Data

`backtest.py` reads from local CSV files. Expected directory structure:

```
data/
  spx/
    stocks_with_sector.csv   # daily OHLCV for all stocks + sector column
    index.csv                # daily OHLCV for the S&P 500 index
  nasdaq/
    stocks.csv
    index.csv
  jp/
    stocks.csv
    index.csv
  uk/
    stocks_ftse.csv
    index.csv
```

The active market is set via `CSV_PATH` / `SPX_PATH` at the top of `backtest.py`.

`screener.py` downloads everything live from Yahoo Finance — no local data needed.

---

## Usage

### Backtest

```bash
python backtest.py
python backtest.py --diag
```

`--diag` enables additional diagnostic output: filter pass rates, fat-tail PnL distribution, per-ticker stats, and Mansfield RS analysis.

Output: prints a full performance report to stdout and saves `trade_log.csv`.

### Screener

```bash
python screener.py --date 2024-06-14 --cap 50000
python screener.py --date 2024-06-14 --cap 50000 --conservative
python screener.py --date 2024-06-14 --cap 50000 --strict-market
```

| Flag | Description |
|---|---|
| `--date` | Date to scan in `YYYY-MM-DD` format (required) |
| `--cap` | Your current account capital in USD (required) |
| `--conservative` | Wider stop: `risk_dist = CHANDELIER_MULT × ATR` instead of `1.2 × ATR` |
| `--strict-market` | Stricter market filter: requires `EMA20 > EMA50` **and** `close > EMA50` — recommended for UK/JP markets |

---

## Strategy Logic

### Market Regime Filter

No new entries are opened unless the index (S&P 500) is in a bullish regime:
- Default: `EMA20 > EMA50`
- Strict (`--strict-market`): `EMA20 > EMA50` AND `close > EMA50`

### Stock Filters (all must pass simultaneously)

| Filter | Condition |
|---|---|
| Trend | `close > EMA200` |
| Short-term trend | `close > EMA20` |
| EMA20 rising | `EMA20 > EMA20[5 days ago]` |
| Mansfield RS | `> -0.02` (not significantly underperforming the index) |
| MACD slope | Slow MACD (60/130) rising on at least 2 of the last 3 days |
| Liquidity | 20-day average dollar volume `> $10M` |
| Entry trigger | At least one trigger fires (see below) |

### Entry Triggers

At least one of the following must be present on the signal day:

| Trigger | Definition |
|---|---|
| **Pin Bar** | Lower shadow ≥ 2× candle body, above-average volume |
| **Engulfing** | Bullish engulfing candle, volume > 1.2× 20-day average |
| **Breakout** | Close above the 10-day high, above-average volume |
| **Gap & Go** | Gap up above prior high, close above open, volume > 1.5× average |
| **Momentum** | Mansfield RS > 0.5, bullish candle closing in the top 10% of its range |

### Position Sizing

Risk per trade is calculated additively:

```
total_risk_pct = BASE_RISK_UNIT (1%)
              + bonus_mansfield (0.5% if Mansfield RS > 0.1)
              + bonus_volume    (0.5% if relative volume between 1.2× and 2×)
```

Position size is the smaller of:
- **Risk-based**: `(capital × total_risk_pct) / risk_distance`
- **Notional cap**: `(capital × 15%) / entry_price`

`risk_distance` in normal mode = `1.2 × ATR`. In `--conservative` mode = `6.0 × ATR` (same as the trailing stop).

Entry price includes 0.5% slippage.

### Stop Loss & Exit

| Exit type | Trigger |
|---|---|
| `SL_GAP` | Open gaps below the stop level — fill at open |
| `SL_TOUCH` | Intraday low touches the stop — fill at stop price |
| `TIME_STOP` | Position held for more than 365 calendar days |
| `DELISTED` | Stock disappears from data for 14+ days — 50% penalty applied |

The stop trails upward using a **Chandelier exit**: `stop = highest_high_since_entry − 6.0 × ATR`.

---

## Output — Backtest

The backtest prints a summary line followed by a per-year table:

```
EQUITY: 87432.10 | ROI: 774.32% | CAGR: 18.45% | MDD: -24.31% | PF: 2.14 | WIN%: 52.3%
```

Then breakdowns by exit type, sector, and entry trigger.

A `trade_log.csv` is saved with one row per closed trade.

### Monte Carlo

After the main report, 100,000 simulations resample trades in random order (with a 10% random skip rate) to test whether results depend on lucky sequencing.

### Year Bootstrap

10,000 simulations resample whole calendar years with replacement — stress-testing scenarios like three consecutive bear years or a decade without a bull market.

Success thresholds: median final capital > $200,000 and P5 > $50,000.

---

## Output — Screener

Prints a ranked table of candidates for the given date:

| Column | Description |
|---|---|
| `Ticker` | Stock symbol |
| `Sector` | GICS sector |
| `Mansfield` | Mansfield Relative Strength value |
| `RealRisk%` | Actual risk as % of capital (using full Chandelier stop distance) |
| `LimitBy` | `risk` = sized by risk formula / `notional_cap` = clipped by 15% cap |
| `Signal` | Active entry triggers on this day |
| `Price` | Last close |
| `EntryEst` | Estimated entry price after slippage |
| `SL` | Initial stop-loss level |
| `ATR` | 14-day Average True Range |
| `Shares` | Suggested position size |
| `Position$` | Total position value in USD |
| `Wallet%` | Position as % of capital |

Candidates are sorted by `RealRisk%` descending (largest portfolio impact first), then by `Mansfield` as tiebreaker.

> **Note:** The screener sorts by `RealRisk%` to highlight the positions with the greatest impact on the portfolio — useful for manual prioritisation. The backtest sorts by `Mansfield` only, since `RealRisk%` requires next-day entry price which is unavailable without lookahead.

---

## Configuration

Key parameters are defined at the top of each file. Both files share the same values — if you change a parameter in one, update the other.

| Parameter | Default | Description |
|---|---|---|
| `CHANDELIER_MULT` | 6.0 | ATR multiplier for trailing stop |
| `BASE_RISK_UNIT` | 0.01 | Base risk per trade (1% of capital) |
| `BASE_RISK_BONUS` | 0.005 | Additive risk bonus per qualifying condition |
| `MAX_NOTIONAL_PCT` | 0.15 | Maximum position size as fraction of capital |
| `MANSFIELD_PERIOD` | 26 | Weeks for Mansfield RS normalisation window |
| `MANSFIELD_THRESHOLD` | -0.02 | Minimum Mansfield RS for entry |
| `MIN_VOLUME` | 10,000,000 | Minimum average daily dollar volume |
| `TIME_STOP_DAYS` | 365 | Maximum holding period in calendar days |
| `SLIPPAGE` | 0.005 | Entry slippage (0.5%) |
| `VOL_ENGULF_MULT` | 1.2 | Relative volume threshold for engulfing trigger |
| `VOL_GAPGO_MULT` | 1.5 | Relative volume threshold for gap-and-go trigger |

### Excluded Sectors

Configured per market. Default (US):
```python
EXCLUDED_SECTORS = {"Financials", "Energy"}
```
