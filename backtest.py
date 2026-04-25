"""
Custom Trading System Backtester
=================================
Long-only momentum / trend-following strategy with:
  - Chandelier exit (trailing stop)
  - Mansfield relative strength filter
  - Sector diversification cap
  - Time stop
  - Monte Carlo & bootstrap robustness tests
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── CLI flags ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--diag",
    action="store_true",
    help="Enable diagnostic output for filters and fat-tail analysis",
)
args = parser.parse_args()
DIAG = args.diag

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# --- Data paths (uncomment the market you want to backtest) ---
CSV_PATH = "data/spx/stocks_with_sector.csv"
SPX_PATH = "data/spx/index.csv"
# CSV_PATH = "data/nasdaq/stocks.csv"
# SPX_PATH = "data/nasdaq/index.csv"
# CSV_PATH = "data/jp/stocks.csv"
# SPX_PATH = "data/jp/index.csv"
# CSV_PATH = "data/uk/stocks_ftse.csv"
# SPX_PATH = "data/uk/index.csv"

# --- Capital & position sizing ---
INITIAL_CAPITAL = 10_000.0
MAX_POSITIONS = 10           # Maximum open positions at any time
MAX_PER_SECTOR = 3           # Max positions in a single sector
SLIPPAGE = 0.005             # Entry slippage (0.5%)
BASE_RISK_UNIT = 0.01        # Base risk per trade as fraction of equity (1%)
BASE_RISK_BONUS = 0.005      # Bonus risk added when conditions are favorable
MAX_NOTIONAL_PCT = 0.15      # Hard cap: position notional ≤ 15% of equity

# --- Stop loss ---
CHANDELIER_MULT = 6.0        # ATR multiplier for the Chandelier trailing stop

# --- Filters ---
MANSFIELD_PERIOD = 26                    # Weeks for Mansfield RS moving average
MANSFIELD_THRESHOLD = -0.02             # Minimum Mansfield RS to enter a trade
MANSFIELD_RISK_BONUS_THRESHOLD = 0.1   # Mansfield RS level required for the risk bonus
MIN_VOLUME_RISK_BONUS = 1.2            # Relative volume range [min, max] for volume bonus
MAX_VOLUME_RISK_BONUS = 2.0
MIN_VOLUME = 10_000_000                 # Minimum average daily dollar volume

# --- Exit rules ---
TIME_STOP_DAYS = 365         # Force-exit after this many calendar days

# --- Delisting handling ---
DELISTING_PENALTY = 0.50     # Assume 50% loss when a stock disappears from data
DELISTING_DAYS = 14          # Days of missing data before treating as delisted

# --- Technical analysis ---
TECH_DAYS_DATA = 400         # Minimum rows required to compute indicators

# --- Simulation ---
MC_SIM = 100_000             # Number of Monte Carlo trade-sequence simulations
SKIP_CHANCE = 0.1            # Probability of randomly skipping a trade in MC (10%)

# --- Entry trigger volume multipliers ---
VOL_ENGULF_MULT = 1.2   # Minimum relative volume for bullish-engulf trigger
VOL_GAPGO_MULT  = 1.5   # Minimum relative volume for gap-and-go trigger

# --- Sector exclusions (uncomment for the appropriate market) ---
EXCLUDED_SECTORS = {"Financials", "Energy"}          # US
# EXCLUDED_SECTORS = {"Utilities", "Healthcare"}       # Japan
# EXCLUDED_SECTORS = {"Communication Services", "Utilities", "Financials", "Energy"}  # UK


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING & TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def calc_ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential moving average with span=length."""
    return series.ewm(span=length, adjust=False).mean()


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX).
    Returns a Series with smoothed ADX values.
    """
    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()

    up_move = high - high.shift()
    down_move = low.shift() - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(span=period, adjust=False).mean()


def calculate_cagr(
    start_value: float,
    end_value: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    """Compound Annual Growth Rate based on calendar dates."""
    years = (end_date - start_date).days / 365.25
    if years <= 0 or start_value <= 0:
        return 0.0
    return (end_value / start_value) ** (1.0 / years) - 1.0


# ── Validate input files ──────────────────────────────────────────────────────
if not os.path.exists(CSV_PATH) or not os.path.exists(SPX_PATH):
    print("ERROR: Missing input CSV files. Check CSV_PATH / SPX_PATH.")
    exit()

# ── Load and prepare the market index ────────────────────────────────────────
spx = pd.read_csv(SPX_PATH)
spx.columns = [c.lower().strip() for c in spx.columns]

# Detect column names flexibly
date_col = "date" if "date" in spx.columns else spx.columns[0]
close_col = "close" if "close" in spx.columns else spx.columns[4]

spx[date_col] = pd.to_datetime(spx[date_col], utc=True).dt.tz_localize(None).dt.normalize()
spx = spx.sort_values(date_col).drop_duplicates(date_col)
spx_close_series = spx.set_index(date_col)[close_col]

# Market regime filter: EMA20 > EMA50 → bullish
spx["ema20"] = spx[close_col].ewm(span=20, adjust=False).mean()
spx["ema50"] = spx[close_col].ewm(span=50, adjust=False).mean()
spx["market_ok"] = spx["ema20"] > spx["ema50"]
# Alternative, stricter filter (uncomment for non-US markets):
# spx["market_ok"] = (spx["ema20"] > spx["ema50"]) & (spx[close_col] > spx["ema50"])

market_regime = spx.set_index(date_col)["market_ok"].to_dict()
market_close = spx.set_index(date_col)[close_col].to_dict()
market_ema50 = spx.set_index(date_col)["ema50"].to_dict()

# ── Load and prepare the stock universe ──────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.columns = [c.lower().strip() for c in df.columns]

# Normalise ticker column name
if "name" in df.columns:
    df.rename(columns={"name": "ticker"}, inplace=True)
elif "symbol" in df.columns:
    df.rename(columns={"symbol": "ticker"}, inplace=True)

df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()

if "sector" not in df.columns or df["sector"].isna().all():
    df["sector"] = "Unknown"

# Remove excluded sectors
df = df[~df["sector"].isin(EXCLUDED_SECTORS)]

if DIAG:
    print(f"\nStocks in raw data    : {df['ticker'].nunique()}")
    print(f"Date range            : {df['date'].min().date()} → {df['date'].max().date()}")

# ── Diagnostic counters ───────────────────────────────────────────────────────
diag_passed_length = 0
diag_passed_join = 0
diag_filter_counts = {
    "ema200": 0, "ema20": 0, "mansfield": 0,
    "macd_slope": 0, "trigger": 0, "ema20_trending": 0,
    "liq_ok": 0, "all_ok": 0,
}

# ── Compute technical indicators per ticker ───────────────────────────────────
processed = []

for ticker, ticker_df in tqdm(df.groupby("ticker"), desc="Technical Analysis"):
    # Require minimum history to compute slow indicators
    if len(ticker_df) < TECH_DAYS_DATA:
        continue
    diag_passed_length += 1

    ticker_df = ticker_df.copy().sort_values("date")

    # Align stock data with index dates (inner join drops non-trading days)
    ticker_df = ticker_df.join(spx_close_series.rename("spx_close"), on="date", how="inner")
    if ticker_df.empty:
        continue
    diag_passed_join += 1

    close = ticker_df["close"]
    high = ticker_df["high"]
    low = ticker_df["low"]
    open_ = ticker_df["open"]

    # Trend indicators
    ticker_df["ema20"] = calc_ema(close, 20)
    ticker_df["ema200"] = calc_ema(close, 200)
    ticker_df["atr"] = (
        pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        )
        .max(axis=1)
        .rolling(14)
        .mean()
    )

    # Mansfield Relative Strength: RS normalized against its own 26-week average
    rs = ticker_df["close"] / ticker_df["spx_close"]
    ticker_df["mansfield"] = (rs / rs.rolling(MANSFIELD_PERIOD * 5).mean()) - 1

    # Slow MACD (60/130) — slope must be rising on 2 of the last 3 days
    macd_line = calc_ema(close, 60) - calc_ema(close, 130)
    ticker_df["macd_slope_up"] = (macd_line > macd_line.shift(1)).rolling(3).sum() >= 2

    # Liquidity
    ticker_df["vol_ma"] = ticker_df["volume"].rolling(20).mean()
    candle_body = abs(close - open_)
    volume_surge = ticker_df["volume"] > ticker_df["vol_ma"]

    # ── Entry triggers ────────────────────────────────────────────────────────
    # Pin bar: long lower shadow ≥ 2× body on above-average volume
    trig_pinbar = ((open_.combine(close, min) - low) >= 2 * candle_body) & volume_surge

    # Bullish engulfing candle on above-average volume
    trig_engulf = (
        (close.shift(1) < open_.shift(1))       # Previous candle bearish
        & (close > open_)                        # Current candle bullish
        & (open_ <= close.shift(1))              # Open below previous close
        & (close >= open_.shift(1))              # Close above previous open
        & (ticker_df["volume"] > VOL_ENGULF_MULT * ticker_df["vol_ma"])
    )

    # 10-day breakout on above-average volume
    trig_breakout = (close > high.shift(1).rolling(10).max()) & volume_surge

    # Gap-and-go: gap up, close above open, strong volume
    trig_gapgo = (open_ > high.shift(1)) & (close > open_) & (ticker_df["volume"] > VOL_GAPGO_MULT * ticker_df["vol_ma"])

    # Momentum: strong Mansfield RS, bullish candle closing near the high
    trig_momentum = (ticker_df["mansfield"] > 0.5) & (close > open_) & (close >= (high - 0.1 * (high - low)))

    # Combine all triggers
    ticker_df["any_trigger"] = trig_pinbar | trig_engulf | trig_breakout | trig_gapgo | trig_momentum
    ticker_df["trig_pinbar"] = trig_pinbar
    ticker_df["trig_engulf"] = trig_engulf
    ticker_df["trig_breakout"] = trig_breakout
    ticker_df["trig_gapgo"] = trig_gapgo
    ticker_df["trig_momentum"] = trig_momentum

    # Additional filters
    ticker_df["ema20_trending"] = ticker_df["ema20"] > ticker_df["ema20"].shift(5)
    ticker_df["liq_ok"] = (close * ticker_df["volume"]).rolling(20).mean() > MIN_VOLUME

    # ── Extra columns logged at entry (analysis only, not used as filters) ────
    ticker_df["ema50"] = calc_ema(close, 50)

    # RSI-14
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    ticker_df["rsi14"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

    # ATR as % of close price
    ticker_df["atr_pct"] = ticker_df["atr"] / close

    # Volume ratio: today's volume vs 20-day average
    ticker_df["vol_ratio"] = ticker_df["volume"] / ticker_df["vol_ma"]

    # Distance from 52-week high (negative = below high)
    ticker_df["dist_52w_high"] = (close / high.rolling(252).max()) - 1

    # Overextension from EMA50 and EMA200
    ticker_df["overext_ema50"]  = (close - ticker_df["ema50"])  / ticker_df["ema50"]
    ticker_df["overext_ema200"] = (close - ticker_df["ema200"]) / ticker_df["ema200"]

    # MACD value at entry (not just slope direction)
    ticker_df["macd_value"] = macd_line

    # Individual filter flags (used for diagnostic counting)
    f_ema200 = close > ticker_df["ema200"]
    f_ema20 = close > ticker_df["ema20"]
    f_mansfield = ticker_df["mansfield"] > MANSFIELD_THRESHOLD
    f_macd = ticker_df["macd_slope_up"]
    f_trigger = ticker_df["any_trigger"]
    f_trending = ticker_df["ema20_trending"]
    f_liq = ticker_df["liq_ok"]

    # Full setup: ALL filters must pass on the same day
    ticker_df["setup_ok"] = (
        f_ema200 & f_ema20 & f_mansfield & f_macd & f_trigger & f_trending & f_liq
    )

    if DIAG:
        if f_ema200.any():              diag_filter_counts["ema200"] += 1
        if f_ema20.any():               diag_filter_counts["ema20"] += 1
        if f_mansfield.any():           diag_filter_counts["mansfield"] += 1
        if f_macd.any():                diag_filter_counts["macd_slope"] += 1
        if f_trigger.any():             diag_filter_counts["trigger"] += 1
        if f_trending.any():            diag_filter_counts["ema20_trending"] += 1
        if f_liq.any():                 diag_filter_counts["liq_ok"] += 1
        if ticker_df["setup_ok"].any(): diag_filter_counts["all_ok"] += 1

    processed.append(ticker_df.dropna(subset=["atr", "mansfield", "vol_ma"]))

# ── Diagnostics: filter pass rates ───────────────────────────────────────────
if DIAG:
    print(f"\n{'=' * 60}")
    print("FILTER DIAGNOSTICS")
    print(f"{'=' * 60}")
    print(f"Stocks with ≥ {TECH_DAYS_DATA} days of data : {diag_passed_length}")
    print(f"Stocks after index join              : {diag_passed_join}")
    print("\nStocks with at least 1 day passing each filter:")
    for filter_name, count in diag_filter_counts.items():
        bar = "█" * int(count / max(diag_passed_join, 1) * 40)
        pct = count / max(diag_passed_join, 1) * 100
        print(f"  {filter_name:<20}: {count:>4} ({pct:>5.1f}%)  {bar}")
    print(f"\n→ Stocks with full setup signal: {diag_filter_counts['all_ok']}")
    print(f"{'=' * 60}\n")

if not processed:
    print("ERROR: No data passed the filters. Check input files and parameters.")
    exit()

data = pd.concat(processed).sort_values(["date", "ticker"])
all_tickers = data["ticker"].unique().tolist()
print(f"Data prepared for {len(all_tickers)} stocks.")

if DIAG:
    signals_per_year = (
        data[data["setup_ok"]]
        .groupby(data[data["setup_ok"]]["date"].dt.year)["ticker"]
        .count()
    )
    print("\nSetup signals per year:")
    for yr, cnt in signals_per_year.items():
        print(f"  {yr}: {cnt} signals")

# Pre-index each ticker's data by date for O(1) lookups during simulation
ticker_data = {tk: grp.set_index("date") for tk, grp in data.groupby("ticker")}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BACKTEST SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def make_trade_log_entry(
    position: dict,
    ticker: str,
    exit_date: pd.Timestamp,
    exit_price: float,
    pnl: float,
    note: str,
) -> dict:
    """Build a standardised dict for a closed trade log entry."""
    return {
        "ticker": ticker,
        "sector": position["sector"],
        "entry_date": position["entry_date"],
        "exit_date": exit_date,
        "days_held": (exit_date - position["entry_date"]).days,
        "entry_price": round(position["entry"], 4),
        "exit_price": round(exit_price, 4),
        "sl_at_exit": round(position["sl"], 4),
        "size": position["size"],
        "pnl": round(pnl, 4),
        "pnl_pct": round(pnl / (position["entry"] * position["size"]) * 100, 2),
        "win": 1 if pnl > 0 else 0,
        "note": note,
        "total_risk_pct": position.get("total_risk_pct", 0),
        "mansfield_at_entry": round(position.get("mansfield_at_entry", 0), 4),
        "ema20_at_entry": round(position.get("ema20_at_entry", 0), 4),
        "overext_at_entry": round(position.get("overext_at_entry", 0), 4),
        "equity_at_entry": round(position["equity_at_entry"], 2),
        "trigger": position.get("trigger", ""),
        # ── Extra entry context (analysis only) ──
        "rsi14_at_entry":       round(position.get("rsi14_at_entry", 0), 2),
        "atr_pct_at_entry":     round(position.get("atr_pct_at_entry", 0), 4),
        "vol_ratio_at_entry":   round(position.get("vol_ratio_at_entry", 0), 2),
        "dist_52w_high_at_entry": round(position.get("dist_52w_high_at_entry", 0), 4),
        "overext_ema50_at_entry":  round(position.get("overext_ema50_at_entry", 0), 4),
        "overext_ema200_at_entry": round(position.get("overext_ema200_at_entry", 0), 4),
        "macd_value_at_entry":  round(position.get("macd_value_at_entry", 0), 4),
        "stop_dist_pct_at_entry": round(position.get("stop_dist_pct_at_entry", 0), 4),
    }


# State variables
cash = INITIAL_CAPITAL
positions: dict = {}          # ticker → position dict
trade_log: list = []          # list of closed trade dicts
equity_curve: list = []
all_dates = sorted(data["date"].unique())
yearly_start_equity = {all_dates[0].year: INITIAL_CAPITAL}
skipped_log: list = []        # (year, reason) for skipped entry signals

for today in tqdm(all_dates, desc="Backtest"):
    current_year = today.year
    tickers_to_close = []
    open_position_value = 0.0

    # Track equity at the start of each new year
    if current_year not in yearly_start_equity and equity_curve:
        yearly_start_equity[current_year] = equity_curve[-1]

    # ── Manage existing positions ─────────────────────────────────────────────
    for ticker, pos in list(positions.items()):
        if today in ticker_data[ticker].index:
            row = ticker_data[ticker].loc[today]
            pos["last_seen"] = today
            exit_price, exit_note = None, None
            days_held = (today - pos["entry_date"]).days

            # Stop-loss: gap below SL → fill at open (slippage already applied at entry)
            if row["open"] <= pos["sl"]:
                exit_price, exit_note = row["open"], "SL_GAP"
            # Stop-loss: intraday touch
            elif row["low"] <= pos["sl"]:
                exit_price, exit_note = pos["sl"], "SL_TOUCH"
            # Time stop: position held too long
            elif days_held > TIME_STOP_DAYS:
                exit_price, exit_note = row["open"], "TIME_STOP"

            if exit_price is not None:
                pnl = (exit_price - pos["entry"]) * pos["size"]
                cash += exit_price * pos["size"]
                trade_log.append(make_trade_log_entry(pos, ticker, today, exit_price, pnl, exit_note))
                tickers_to_close.append(ticker)
            else:
                # Trail the stop upward using Chandelier logic
                if row["high"] > pos["hi_since_entry"]:
                    pos["hi_since_entry"] = row["high"]
                    pos["sl"] = max(
                        pos["sl"],
                        pos["hi_since_entry"] - CHANDELIER_MULT * row["atr"],
                    )
                pos["last_price"] = row["close"]
                open_position_value += pos["size"] * row["close"]
        else:
            # Stock missing from data — handle potential delisting
            days_missing = (today - pos.get("last_seen", pos["entry_date"])).days
            if days_missing > DELISTING_DAYS:
                exit_price = pos["last_price"] * DELISTING_PENALTY
                pnl = (exit_price - pos["entry"]) * pos["size"]
                cash += exit_price * pos["size"]
                trade_log.append(make_trade_log_entry(pos, ticker, today, exit_price, pnl, "DELISTED"))
                tickers_to_close.append(ticker)
            else:
                # Keep last known price for equity calculation
                open_position_value += pos["size"] * pos["last_price"]

    for ticker in tickers_to_close:
        del positions[ticker]

    total_equity = cash + open_position_value
    equity_curve.append(total_equity)

    # Skip entries when market regime filter is not satisfied
    if not market_regime.get(today, False):
        continue

    # ── Look for new entry signals ────────────────────────────────────────────
    # Sort candidates by Mansfield RS descending (strongest momentum first)
    candidates = data[(data["date"] == today) & data["setup_ok"]].sort_values(
        "mansfield", ascending=False
    )

    for _, signal in candidates.iterrows():
        ticker = signal["ticker"]

        if ticker in positions:
            continue  # Already holding this stock

        if len(positions) >= MAX_POSITIONS:
            skipped_log.append((current_year, "max_positions"))
            break

        # Enforce per-sector cap
        sector_count = sum(1 for p in positions.values() if p["sector"] == signal["sector"])
        if sector_count >= MAX_PER_SECTOR:
            skipped_log.append((current_year, "max_sector"))
            continue

        # Use tomorrow's open as entry (simulate next-day execution)
        next_day = ticker_data[ticker][ticker_data[ticker].index > today].head(1)
        if next_day.empty:
            continue

        entry_price = next_day.iloc[0]["open"] * (1 + SLIPPAGE)

        # --- Dynamic risk sizing ---
        # Optional drawdown-based risk reduction (currently disabled):
        # peak_equity = max(equity_curve) if equity_curve else INITIAL_CAPITAL
        # current_dd = (total_equity - peak_equity) / peak_equity
        # dd_risk_mult = 0.5 if current_dd < -0.15 else 0.75 if current_dd < -0.08 else 1.0
        dd_risk_mult = 1.0

        # Optional market-regime-based risk reduction (currently disabled):
        # m_close = market_close.get(today)
        # m_ema50 = market_ema50.get(today)
        # if m_close and m_ema50 and m_close < m_ema50:
        #     dd_risk_mult = 0.5

        rel_vol = signal["volume"] / signal["vol_ma"]
        bonus_mansfield = BASE_RISK_BONUS if signal["mansfield"] > MANSFIELD_RISK_BONUS_THRESHOLD else 0
        bonus_volume = BASE_RISK_BONUS if MIN_VOLUME_RISK_BONUS <= rel_vol <= MAX_VOLUME_RISK_BONUS else 0
        total_risk_pct = (BASE_RISK_UNIT + bonus_mansfield + bonus_volume) * dd_risk_mult

        # Stop distance = 1.2 × ATR below entry
        stop_distance = 1.2 * signal["atr"]

        # Position size: smaller of risk-based and notional-cap-based
        size = min(
            int((total_equity * total_risk_pct) / stop_distance),
            int((total_equity * MAX_NOTIONAL_PCT) // entry_price),
        )

        if size <= 0:
            continue

        cost = entry_price * size
        if cash < cost:
            skipped_log.append((current_year, "no_cash"))
            continue

        cash -= cost

        # Record which triggers fired on the signal day
        trigger_map = [
            ("BREAKOUT", "trig_breakout"),
            ("GAPGO",    "trig_gapgo"),
            ("ENGULF",   "trig_engulf"),
            ("MOMENTUM", "trig_momentum"),
            ("PINBAR",   "trig_pinbar"),
        ]
        active_triggers = [name for name, col in trigger_map if signal.get(col, False)]
        entry_trigger = ",".join(active_triggers) if active_triggers else "UNKNOWN"

        stop_distance_pct = stop_distance / entry_price

        positions[ticker] = {
            "ticker": ticker,
            "sector": signal["sector"],
            "entry": entry_price,
            "last_price": entry_price,
            "sl": entry_price - (CHANDELIER_MULT * signal["atr"]),
            "size": size,
            "total_risk_pct": total_risk_pct,
            "hi_since_entry": entry_price,
            "entry_date": today,
            "last_seen": today,
            "equity_at_entry": total_equity,
            "mansfield_at_entry": signal["mansfield"],
            "ema20_at_entry": signal["ema20"],
            "overext_at_entry": round((signal["close"] - signal["ema20"]) / signal["ema20"], 4),
            "trigger": entry_trigger,
            # ── Extra entry context ──
            "rsi14_at_entry":         signal.get("rsi14", 0),
            "atr_pct_at_entry":       signal.get("atr_pct", 0),
            "vol_ratio_at_entry":     signal.get("vol_ratio", 0),
            "dist_52w_high_at_entry": signal.get("dist_52w_high", 0),
            "overext_ema50_at_entry":  signal.get("overext_ema50", 0),
            "overext_ema200_at_entry": signal.get("overext_ema200", 0),
            "macd_value_at_entry":    signal.get("macd_value", 0),
            "stop_dist_pct_at_entry": stop_distance_pct,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def calc_profit_factor(pnl_series: pd.Series) -> float:
    """Profit factor = gross wins / gross losses. Returns inf if no losses."""
    gross_wins = pnl_series[pnl_series > 0].sum()
    gross_losses = abs(pnl_series[pnl_series < 0].sum())
    return gross_wins / gross_losses if gross_losses > 0 else float("inf")


print("\n" + "=" * 125)

if not trade_log:
    print("NO TRADES EXECUTED.")
else:
    log_df = pd.DataFrame(trade_log)
    log_df["year"] = log_df["exit_date"].dt.year

    eq_series = pd.Series(equity_curve)
    max_drawdown = ((eq_series - eq_series.cummax()) / eq_series.cummax()).min() * 100
    overall_profit_factor = calc_profit_factor(log_df["pnl"])

    # CAGR
    start_date = all_dates[0]
    end_date = all_dates[-1]
    cagr = calculate_cagr(INITIAL_CAPITAL, equity_curve[-1], start_date, end_date)

    # Equity at year-end (last recorded equity in each calendar year)
    yearly_end_equity = {}
    for d, eq in zip(all_dates, equity_curve):
        yearly_end_equity[d.year] = eq

    roi_pct = (equity_curve[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_rate = log_df["win"].sum() / len(log_df) * 100

    print(
        f"EQUITY: {equity_curve[-1]:.2f}"
        f" | ROI: {roi_pct:.2f}%"
        f" | CAGR: {cagr:.2%}"
        f" | MDD: {max_drawdown:.2f}%"
        f" | PF: {overall_profit_factor:.2f}"
        f" | WIN%: {win_rate:.1f}%"
    )
    print("=" * 125)

    # Build skipped-signals dataframe for per-year reporting
    skipped_df = (
        pd.DataFrame(skipped_log, columns=["year", "reason"])
        if skipped_log
        else pd.DataFrame(columns=["year", "reason"])
    )

    # Per-year performance table
    print(
        f"{'YEAR':<6} | {'START_EQ':>10} | {'END_EQ':>10} | {'PNL':>11}"
        f" | {'ROI':>9} | {'TRADES':>8} | {'WIN%':>6} | {'PF':>6}"
        f" | {'SKIP_POS':>8} | {'SKIP_SEC':>8} | {'SKIP_CASH':>9} | NOTES"
    )
    print("-" * 155)
    for yr, year_group in log_df.groupby("year"):
        year_pnl = year_group["pnl"].sum()
        trade_count = len(year_group)
        start_eq = yearly_start_equity.get(yr, INITIAL_CAPITAL)
        end_eq = yearly_end_equity.get(yr, start_eq)
        year_roi = year_pnl / start_eq * 100
        win_rate_yr = year_group["win"].sum() / trade_count * 100
        year_skipped = (
            skipped_df[skipped_df["year"] == yr]["reason"].value_counts()
            if not skipped_df.empty
            else {}
        )
        skip_positions = year_skipped.get("max_positions", 0)
        skip_sector = year_skipped.get("max_sector", 0)
        skip_cash = year_skipped.get("no_cash", 0)
        print(
            f"{yr:<6} | {start_eq:>10.0f} | {end_eq:>10.0f} | {year_pnl:>+11.2f}"
            f" | {year_roi:>9.2f}% | {trade_count:>8} | {win_rate_yr:>6.1f}%"
            f" | {calc_profit_factor(year_group['pnl']):>6.2f}"
            f" | {skip_positions:>8} | {skip_sector:>8} | {skip_cash:>9}"
            f" | {year_group['note'].value_counts().to_dict()}"
        )

    # Exit type breakdown
    print("\nExit type breakdown:")
    for exit_type, count in log_df["note"].value_counts().items():
        subset = log_df[log_df["note"] == exit_type]
        print(
            f"  {exit_type:<12}: {count:>5} trades"
            f" | avg PnL: {subset['pnl'].mean():>+8.2f}"
            f" | avg days: {subset['days_held'].mean():.1f}"
        )

    # Sector distribution
    print("\nSector distribution:")
    for sector, group in log_df.groupby("sector"):
        print(f"  {sector:<35}: {len(group):>4} trades | avg PnL: {group['pnl'].mean():>+8.2f}")

    # Entry trigger distribution
    print("\nEntry trigger breakdown:")
    for trigger, group in log_df.groupby("trigger"):
        trig_win_rate = group["win"].sum() / len(group) * 100
        print(
            f"  {trigger:<30}: {len(group):>4} trades"
            f" | win%: {trig_win_rate:>5.1f}%"
            f" | avg PnL: {group['pnl'].mean():>+8.2f}"
            f" | avg days: {group['days_held'].mean():.1f}"
        )

    # ── Diagnostic deep-dives ─────────────────────────────────────────────────
    if DIAG:
        # PnL% distribution (fat-tail check)
        print("\nPnL% distribution (fat-tail check):")
        bins = [-100, -20, -10, -5, 0, 5, 10, 20, 50, 100, 500]
        log_df["pnl_bucket"] = pd.cut(log_df["pnl_pct"], bins=bins)
        for bucket, count in log_df["pnl_bucket"].value_counts().sort_index().items():
            bar = "█" * min(count, 80)  # Cap bar width at 80 chars
            print(f"  {str(bucket):<20}: {count:>4}  {bar}")

        # Top trades
        print("\nTop 10 trades by PnL%:")
        cols = ["ticker", "sector", "entry_date", "exit_date", "days_held", "pnl", "pnl_pct", "note"]
        print(log_df.nlargest(10, "pnl_pct")[cols].to_string(index=False))

        # Per-ticker stats
        print("\nPer-ticker results (min. 3 trades, sorted by total PnL):")
        ticker_stats = (
            log_df.groupby("ticker")
            .agg(trades=("pnl", "count"), total_pnl=("pnl", "sum"),
                 avg_pnl_pct=("pnl_pct", "mean"), win_rate=("win", "mean"))
            .query("trades >= 3")
            .sort_values("total_pnl", ascending=False)
        )
        print(ticker_stats.round(2).to_string())

        # Mansfield RS at entry vs outcome
        print("\nMansfield RS at entry vs outcome:")
        log_df["mansfield_bucket"] = pd.cut(
            log_df["mansfield_at_entry"], bins=[-1, 0, 0.1, 0.2, 0.5, 1, 10]
        )
        msf_stats = log_df.groupby("mansfield_bucket")[["pnl_pct", "win"]].agg(["mean", "count"])
        print(msf_stats.round(2))

    log_df.to_csv("trade_log.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MONTE CARLO (random trade-sequence resampling)
# Hypothesis: does the system perform regardless of which order trades occur in?
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nRunning Monte Carlo ({MC_SIM:,} simulations)...")

# Express each trade's P&L as a fraction of equity at the time of entry
trade_returns = (log_df["pnl"] / log_df["equity_at_entry"]).values

mc_final_capitals = []
for _ in range(MC_SIM):
    # Bootstrap trade returns and randomly drop some (simulate execution uncertainty)
    sampled_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
    executed_returns = [r for r in sampled_returns if np.random.random() > SKIP_CHANCE]
    capital = INITIAL_CAPITAL
    for ret in executed_returns:
        capital *= 1 + ret
    mc_final_capitals.append(capital)

mc_results = np.array(mc_final_capitals)
print(
    f"MC Median    : {np.median(mc_results):>12.2f} USD\n"
    f"MC P10       : {np.percentile(mc_results, 10):>12.2f} USD\n"
    f"MC P90       : {np.percentile(mc_results, 90):>12.2f} USD\n"
    f"Loss prob    : {(mc_results < INITIAL_CAPITAL).mean() * 100:.2f}%"
)
print("=" * 125)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — YEAR BOOTSTRAP (random year-sequence resampling)
# Hypothesis: does the system work regardless of which order calendar years occur?
# Unlike Monte Carlo, this resamples whole years (market regimes), not individual
# trades — stress-testing scenarios like three consecutive bear years or a
# decade without a bull market.
# Success thresholds: median > $200,000 and P5 > $50,000
# ══════════════════════════════════════════════════════════════════════════════

BOOTSTRAP_SIMS = 10_000   # 10k is sufficient; year-level bootstrap is fast

log_df["year"] = log_df["exit_date"].dt.year

# Build a per-year dictionary of trade returns (as fraction of equity at entry)
yearly_trade_returns: dict = {}
for yr, group in log_df.groupby("year"):
    yearly_trade_returns[yr] = (group["pnl"] / group["equity_at_entry"]).tolist()

available_years = sorted(yearly_trade_returns.keys())
num_years = len(available_years)

print(f"Running Year Bootstrap ({BOOTSTRAP_SIMS:,} simulations, {num_years} years available)...")

bootstrap_final_capitals = []
for _ in range(BOOTSTRAP_SIMS):
    # Resample years with replacement (same length as original history)
    sampled_years = np.random.choice(available_years, size=num_years, replace=True)
    capital = INITIAL_CAPITAL
    for yr in sampled_years:
        for ret in yearly_trade_returns[yr]:
            capital *= 1 + ret
    bootstrap_final_capitals.append(capital)

bs_results = np.array(bootstrap_final_capitals)
bs_median = np.median(bs_results)
bs_p5 = np.percentile(bs_results, 5)
bs_p10 = np.percentile(bs_results, 10)
bs_p90 = np.percentile(bs_results, 90)
bs_loss_prob = (bs_results < INITIAL_CAPITAL).mean() * 100
bs_prob_100k = (bs_results > 100_000).mean() * 100
bs_prob_200k = (bs_results > 200_000).mean() * 100

verdict_median = "✅ OK" if bs_median > 200_000 else "⚠️  WEAK"
verdict_p5 = "✅ OK" if bs_p5 > 50_000 else "⚠️  WEAK"

print(f"\nYear Bootstrap results ({num_years} years resampled with replacement):")
print(f"  Median final capital  : {bs_median:>12.2f} USD  {verdict_median}  (threshold: >200,000)")
print(f"  P5                    : {bs_p5:>12.2f} USD  {verdict_p5}  (threshold: > 50,000)")
print(f"  P10                   : {bs_p10:>12.2f} USD")
print(f"  P90                   : {bs_p90:>12.2f} USD")
print(f"  Loss probability      : {bs_loss_prob:>11.2f}%")
print(f"  Prob > $100,000       : {bs_prob_100k:>11.2f}%")
print(f"  Prob > $200,000       : {bs_prob_200k:>11.2f}%")

if bs_median > 200_000 and bs_p5 > 50_000:
    print("\n  → Both thresholds met. System is robust to random year-sequence resampling.")
elif bs_median > 200_000:
    print("\n  → Median OK, but left tail is weak. System is sensitive to bad year sequences.")
else:
    print("\n  → Warning: system may be fragile to the ordering of market regimes.")

print("=" * 125)
