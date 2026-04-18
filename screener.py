"""
Live Stock Screener
====================
Scans the S&P 500 universe for setups that match the backtest entry criteria.
Outputs position-sizing suggestions based on current capital and risk parameters.

Usage:
    python screener.py --date 2024-06-14 --cap 50000
    python screener.py --date 2024-06-14 --cap 50000 --conservative
    python screener.py --date 2024-06-14 --cap 50000 --strict-market
"""

import argparse
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  (kept in sync with backtest.py)
# ══════════════════════════════════════════════════════════════════════════════

# --- Volume thresholds for entry triggers ---
VOL_ENGULF_MULT = 1.2   # Minimum relative volume for bullish-engulf trigger
VOL_GAPGO_MULT  = 1.5   # Minimum relative volume for gap-and-go trigger

# --- Mansfield Relative Strength ---
MANSFIELD_PERIOD                = 26
MANSFIELD_THRESHOLD             = -0.02   # Minimum RS to consider a stock
MANSFIELD_RISK_BONUS_THRESHOLD  = 0.1    # RS level required for the Mansfield risk bonus

# --- Liquidity & volume ---
MIN_VOLUME_RISK_BONUS = 1.2       # Relative volume range [min, max] for the volume risk bonus
MAX_VOLUME_RISK_BONUS = 2.0
MIN_VOLUME            = 10_000_000  # Minimum 20-day average dollar volume

# --- Stop-loss & trailing stop ---
CHANDELIER_MULT = 6.0   # ATR multiplier for Chandelier stop (shared with backtest)

# --- Position sizing ---
BASE_RISK_UNIT   = 0.01   # Base risk per trade as a fraction of capital (1%)
BASE_RISK_BONUS  = 0.005  # Additive bonus when Mansfield or volume condition is met
MAX_NOTIONAL_PCT = 0.15   # Hard cap: position notional ≤ 15% of capital

# Risk distance modes:
#   Normal       : risk_dist = RISK_DIST_NORMAL × ATR  (tighter, larger position)
#   Conservative : risk_dist = CHANDELIER_MULT  × ATR  (wider = realistic SL distance)
RISK_DIST_NORMAL = 1.2

SLIPPAGE = 0.005   # 0.5% entry slippage — consistent with backtest

# --- Universe filters ---
EXCLUDED_SECTORS = {"Financials", "Energy"}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def calc_ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential moving average with span=length."""
    return series.ewm(span=length, adjust=False).mean()


def get_sp500_tickers() -> pd.DataFrame | None:
    """
    Fetch the current S&P 500 constituent list from a public CSV.
    Returns a DataFrame with columns ['ticker', 'sector'], or None on failure.
    """
    url = (
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies"
        "/main/data/constituents.csv"
    )
    try:
        df = pd.read_csv(url)
        # Yahoo Finance uses '-' instead of '.' in tickers (e.g. BRK-B)
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)

        sector_col = (
            "Sector" if "Sector" in df.columns
            else "GICS Sector" if "GICS Sector" in df.columns
            else None
        )
        if sector_col:
            return df[["Symbol", sector_col]].rename(
                columns={"Symbol": "ticker", sector_col: "sector"}
            )
        else:
            out = df[["Symbol"]].rename(columns={"Symbol": "ticker"})
            out["sector"] = "Unknown"
            return out
    except Exception as exc:
        logging.debug(f"Failed to fetch ticker list: {exc}")
        return None


def calculate_indicators(stock_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Compute all technical indicators and entry-trigger flags for a single stock.

    Parameters
    ----------
    stock_df  : Daily OHLCV data for the stock (must include a 'date' column).
    market_df : Daily OHLCV data for the market index (used for Mansfield RS).

    Returns
    -------
    DataFrame with indicator columns appended, or None if insufficient history.
    """
    if stock_df is None or len(stock_df) < 250:
        return None

    # Align stock and market data on date
    market_subset = market_df[["date", "close"]].copy().rename(columns={"close": "market_close"})
    df = stock_df.merge(market_subset, on="date", how="left")
    df["market_close"] = df["market_close"].ffill()
    df = df.dropna(subset=["market_close", "close"])

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"]
    volume = df["volume"]

    # Average True Range (14-day)
    df["atr"] = (
        pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        )
        .max(axis=1)
        .rolling(14)
        .mean()
    )

    # Mansfield Relative Strength: RS normalised against its own 26-week mean
    rs = close / df["market_close"]
    df["mansfield"] = (rs / rs.rolling(MANSFIELD_PERIOD * 5).mean()) - 1

    # Slow MACD (60/130) — slope must be rising on 2 of the last 3 days
    macd_line = calc_ema(close, 60) - calc_ema(close, 130)
    df["macd_slope_up"] = (macd_line > macd_line.shift(1)).rolling(3).sum() >= 2

    # Trend EMAs
    df["ema20"]          = calc_ema(close, 20)
    df["ema200"]         = calc_ema(close, 200)
    df["ema20_trending"] = df["ema20"] > df["ema20"].shift(5)

    # Liquidity
    df["vol_ma"] = volume.rolling(20).mean()
    df["liq_ok"] = (close * volume).rolling(20).mean() > MIN_VOLUME

    candle_body   = abs(close - open_)
    volume_surge  = volume > df["vol_ma"]

    # ── Entry triggers (identical definitions to backtest) ────────────────────

    # Pin bar: long lower shadow ≥ 2× body on above-average volume
    df["t_pinbar"] = ((open_.combine(close, min) - low) >= 2 * candle_body) & volume_surge

    # Bullish engulfing candle on above-average volume
    df["t_engulf"] = (
        (close.shift(1) < open_.shift(1))         # Previous candle bearish
        & (close > open_)                          # Current candle bullish
        & (open_ <= close.shift(1))               # Open at or below previous close
        & (close >= open_.shift(1))               # Close at or above previous open
        & (volume > VOL_ENGULF_MULT * df["vol_ma"])
    )

    # 10-day breakout on above-average volume
    df["t_breakout"] = (close > high.shift(1).rolling(10).max()) & volume_surge

    # Gap-and-go: gap up, close above open, strong volume
    df["t_gapgo"] = (open_ > high.shift(1)) & (close > open_) & (volume > VOL_GAPGO_MULT * df["vol_ma"])

    # Momentum: strong Mansfield RS, bullish candle closing near the high
    df["t_momentum"] = (
        (df["mansfield"] > 0.5)
        & (close > open_)
        & (close >= (high - 0.1 * (high - low)))
    )

    df["any_trigger"] = (
        df["t_pinbar"] | df["t_engulf"] | df["t_breakout"] | df["t_gapgo"] | df["t_momentum"]
    )

    # Full setup: all filters must pass simultaneously
    df["setup_ok"] = (
        (close > df["ema200"])
        & (close > df["ema20"])
        & (df["mansfield"] > MANSFIELD_THRESHOLD)
        & (df["macd_slope_up"])
        & (df["any_trigger"])
        & (df["ema20_trending"])
        & (df["liq_ok"])
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="S&P 500 momentum screener")
    parser.add_argument("--date",          type=str,   required=True,
                        help="Scan date in YYYY-MM-DD format")
    parser.add_argument("--cap",           type=float, required=True,
                        help="Your current account capital in USD")
    parser.add_argument("--conservative",  action="store_true",
                        help="Conservative mode: risk_dist = CHANDELIER_MULT × ATR (wider stop)")
    parser.add_argument("--strict-market", action="store_true",
                        help="Stricter market filter: EMA20 > EMA50 AND close > EMA50 (recommended for UK/JP)")
    args = parser.parse_args()

    # Determine risk-distance multiplier and display labels
    if args.conservative:
        risk_dist_mult = CHANDELIER_MULT
        mode_label     = f"CONSERVATIVE (risk_dist = {CHANDELIER_MULT}×ATR)"
    else:
        risk_dist_mult = RISK_DIST_NORMAL
        mode_label     = f"NORMAL (risk_dist = {RISK_DIST_NORMAL}×ATR)"

    market_filter_label = (
        "EMA20 > EMA50 + close > EMA50" if args.strict_market else "EMA20 > EMA50"
    )

    scan_date = pd.to_datetime(args.date).normalize()

    # ── Download market index data ────────────────────────────────────────────
    print(f"Downloading market data for {scan_date.date()}...")
    market_raw = yf.download(
        "^GSPC",
        start=scan_date - timedelta(days=800),
        end=scan_date + timedelta(days=5),
        progress=False,
    )
    if market_raw.empty:
        print("ERROR: Failed to download S&P 500 data.")
        return

    market_df = market_raw.reset_index()
    market_df.columns = [
        col[0].lower() if isinstance(col, tuple) else col.lower()
        for col in market_df.columns
    ]
    market_df["date"] = pd.to_datetime(market_df["date"]).dt.tz_localize(None).dt.normalize()
    market_df["ema20"] = calc_ema(market_df["close"], 20)
    market_df["ema50"] = calc_ema(market_df["close"], 50)

    # Use the most recent trading day on or before the scan date
    market_row = market_df[market_df["date"] <= scan_date].tail(1)
    if market_row.empty:
        print("ERROR: No market data available for the specified date.")
        return

    actual_date = market_row["date"].iloc[0]
    market_ema20  = market_row["ema20"].iloc[0]
    market_ema50  = market_row["ema50"].iloc[0]
    market_close  = market_row["close"].iloc[0]

    # ── Market regime filter ──────────────────────────────────────────────────
    if args.strict_market:
        market_ok = (market_ema20 > market_ema50) and (market_close > market_ema50)
    else:
        market_ok = market_ema20 > market_ema50

    trend_label = "BULL (OK)" if market_ok else "BEAR (STOP)"
    print(
        f"Session : {actual_date.date()}"
        f" | SPX close: {market_close:.0f}"
        f" | EMA20: {market_ema20:.0f}"
        f" | EMA50: {market_ema50:.0f}"
        f" | Trend: {trend_label}"
        f" | Filter: {market_filter_label}"
        f" | Mode: {mode_label}"
    )

    if not market_ok:
        print("Screener stopped — market does not satisfy the regime filter.")
        if args.strict_market and (market_ema20 > market_ema50) and not (market_close > market_ema50):
            print(f"  → EMA20 > EMA50: YES, but close ({market_close:.0f}) < EMA50 ({market_ema50:.0f})")
            print("  → Market is bouncing but price is still below EMA50 — strict filter blocks entry.")
        return

    # ── Load stock universe ───────────────────────────────────────────────────
    ticker_df = get_sp500_tickers()
    if ticker_df is None:
        print("ERROR: Failed to fetch ticker list.")
        return

    if EXCLUDED_SECTORS:
        ticker_df = ticker_df[~ticker_df["sector"].isin(EXCLUDED_SECTORS)]

    tickers    = ticker_df["ticker"].tolist()
    sector_map = ticker_df.set_index("ticker")["sector"].to_dict()

    # ── Bulk download all stock data in a single request ─────────────────────
    print(f"Scanning {len(tickers)} stocks...")
    raw_all = yf.download(
        tickers,
        start=actual_date - timedelta(days=800),
        end=actual_date + timedelta(days=5),
        group_by="ticker",
        progress=True,
    )

    # ── Evaluate each ticker ──────────────────────────────────────────────────
    signals = []

    for ticker in tqdm(tickers, desc="Processing"):
        try:
            stock_df = raw_all[ticker].dropna(subset=["Close"]).copy().reset_index()
            stock_df.columns = [col.lower() for col in stock_df.columns]
            stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.tz_localize(None).dt.normalize()

            result_df = calculate_indicators(stock_df, market_df)
            if result_df is None:
                continue

            day_row = result_df[result_df["date"] == actual_date]
            if day_row.empty or not day_row.iloc[0]["setup_ok"]:
                continue

            row = day_row.iloc[0]

            # ── Risk sizing (additive bonuses, identical to backtest) ──────────
            vol_ratio      = row["volume"] / row["vol_ma"]
            bonus_mansfield = BASE_RISK_BONUS if row["mansfield"] > MANSFIELD_RISK_BONUS_THRESHOLD else 0
            bonus_volume    = BASE_RISK_BONUS if MIN_VOLUME_RISK_BONUS <= vol_ratio <= MAX_VOLUME_RISK_BONUS else 0
            total_risk_pct  = BASE_RISK_UNIT + bonus_mansfield + bonus_volume

            risk_distance = risk_dist_mult * row["atr"]
            if risk_distance <= 0:
                logging.debug(f"{ticker}: risk_distance <= 0, skipping")
                continue

            # Estimated entry price including slippage (mirrors backtest logic)
            entry_est           = row["close"] * (1 + SLIPPAGE)
            shares_by_risk      = int((args.cap * total_risk_pct) / risk_distance)
            shares_by_notional  = int((args.cap * MAX_NOTIONAL_PCT) // entry_est)
            shares              = min(shares_by_risk, shares_by_notional)

            if shares <= 0:
                logging.debug(f"{ticker}: shares=0, skipping")
                continue

            # Actual risk in USD using the full Chandelier stop distance
            sl_price      = row["close"] - (CHANDELIER_MULT * row["atr"])
            real_risk_usd = (entry_est - sl_price) * shares
            real_risk_pct = real_risk_usd / args.cap * 100

            # Flag whether position was capped by notional limit or by risk formula
            limit_by = (
                "notional_cap"
                if shares == shares_by_notional and shares_by_risk > shares_by_notional
                else "risk"
            )

            # Collect active trigger names for display
            trigger_checks = [
                (row["t_pinbar"],   "PinBar"),
                (row["t_engulf"],   "Engulfing"),
                (row["t_breakout"], "Breakout"),
                (row["t_gapgo"],    "GapGo"),
                (row["t_momentum"], "Momentum"),
            ]
            active_triggers = [name for cond, name in trigger_checks if cond]

            signals.append({
                "Ticker":      ticker,
                "Sector":      sector_map.get(ticker, "Unknown"),
                "Mansfield":   round(row["mansfield"], 3),
                "RealRisk%":   round(real_risk_pct, 2),
                "LimitBy":     limit_by,
                "Signal":      "|".join(active_triggers),
                "Price":       round(row["close"], 2),
                "EntryEst":    round(entry_est, 2),
                "SL":          round(sl_price, 2),
                "ATR":         round(row["atr"], 2),
                "Shares":      shares,
                "Position$":   round(shares * entry_est, 2),
                "Wallet%":     round((shares * entry_est / args.cap) * 100, 1),
            })

        except Exception as exc:
            logging.debug(f"{ticker}: {exc}")
            continue

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 125)
    if signals:
        results_df = (
            pd.DataFrame(signals)
            .sort_values(["Mansfield", "RealRisk%"], ascending=[False, False])
            .reset_index(drop=True)
        )
        results_df.insert(0, "#", results_df.index + 1)

        print(
            f"CANDIDATES | Capital: ${args.cap:,.0f}"
            f" | Filter: {market_filter_label}"
            f" | Mode: {mode_label}"
            f" | Notional cap: {MAX_NOTIONAL_PCT * 100:.0f}%"
        )
        print("-" * 125)
        print(results_df.to_string(index=False))
        print("-" * 125)
        print(f"LimitBy: notional_cap = position clipped by the {MAX_NOTIONAL_PCT * 100:.0f}% notional cap")
        print(f"LimitBy: risk         = position sized purely by risk formula (notional cap not hit)")
        print(f"EntryEst = Price × (1 + {SLIPPAGE}) — estimated fill price after slippage (mirrors backtest)")
    else:
        print("No signals found matching all criteria.")

    print("=" * 125)


if __name__ == "__main__":
    main()
