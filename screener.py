import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
import argparse
from datetime import datetime, timedelta
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURACJA ZGODNA Z v23.1 (STABILNA)
# ══════════════════════════════════════════════════════════════════════════════
VOL_ENG_MULT        = 1.2
VOL_GAP_MULT        = 1.5
MANSFIELD_PERIOD    = 26
MANSFIELD_THRESHOLD = -0.02
MANSFIELD_RISK_BONUS_THRESHOLD = 0.1
MIN_VOLUME_RISK_BONUS = 1.2
MAX_VOLUME_RISK_BONUS = 2
LIQ_THRESHOLD       = 10_000_000
CHANDELIER_MULT     = 6.0    # wspolna stala z symulatorem

# Parametry ryzyka
BASE_RISK_UNIT      = 0.01   # 1% bazy
MAX_NOTIONAL_PCT    = 0.15   # 15% max na spolke
EXCLUDED_SECTORS    = set()

# Tryb normalny:       risk_dist = RISK_DIST_NORMAL x ATR  (ciasny, wieksza pozycja)
# Tryb konserwatywny:  risk_dist = CHANDELIER_MULT  x ATR  (szeroki = realny SL)
RISK_DIST_NORMAL    = 1.2
# ══════════════════════════════════════════════════════════════════════════════


def calc_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def get_sp500_tickers():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    try:
        df = pd.read_csv(url)
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        sector_col = 'Sector' if 'Sector' in df.columns else (
            'GICS Sector' if 'GICS Sector' in df.columns else None)
        if sector_col:
            return df[['Symbol', sector_col]].rename(columns={'Symbol': 'ticker', sector_col: 'sector'})
        else:
            df_out = df[['Symbol']].rename(columns={'Symbol': 'ticker'})
            df_out['sector'] = 'Unknown'
            return df_out
    except Exception as e:
        logging.debug(f"Blad pobierania tickerow: {e}")
        return None


def calculate_indicators(df, market_df):
    if df is None or len(df) < 250:
        return None

    market_subset = market_df[['date', 'close']].copy().rename(columns={'close': 'market_close'})
    df = df.merge(market_subset, on='date', how='left')
    df['market_close'] = df['market_close'].ffill()
    df = df.dropna(subset=['market_close', 'close'])

    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']

    df['atr'] = (
        pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1)
        .max(axis=1)
        .rolling(14)
        .mean()
    )

    rs = c / df['market_close']
    df['mansfield'] = (rs / rs.rolling(MANSFIELD_PERIOD * 5).mean()) - 1

    macd_line = calc_ema(c, 60) - calc_ema(c, 130)
    df['macd_slope_up'] = (macd_line > macd_line.shift(1)).rolling(3).sum() >= 2
    df['ema20']          = calc_ema(c, 20)
    df['ema200']         = calc_ema(c, 200)
    df['ema20_trending'] = df['ema20'] > df['ema20'].shift(5)

    df['vol_ma'] = v.rolling(20).mean()
    df['liq_ok'] = (c * v).rolling(20).mean() > LIQ_THRESHOLD

    body  = abs(c - o)
    vol_f = v > df['vol_ma']

    df['t_pinbar']   = ((o.combine(c, min) - l) >= 2 * body) & vol_f
    df['t_engulf']   = (
        (c.shift(1) < o.shift(1)) & (c > o) &
        (c > o.shift(1)) & (o < c.shift(1)) &
        (v > VOL_ENG_MULT * df['vol_ma'])
    )
    df['t_breakout'] = (c > h.shift(1).rolling(10).max()) & vol_f
    df['t_gapgo']    = (o > h.shift(1)) & (c > o) & (v > VOL_GAP_MULT * df['vol_ma'])
    df['t_momentum'] = (df['mansfield'] > 0.5) & (c > o) & (c >= (h - 0.1 * (h - l)))

    df['any_trigger'] = (
        df['t_pinbar'] | df['t_engulf'] | df['t_breakout'] |
        df['t_gapgo']  | df['t_momentum']
    )

    df['setup_ok'] = (
        (c > df['ema200']) &
        (c >= df['ema20'] * 0.995) &
        (df['mansfield'] > MANSFIELD_THRESHOLD) &
        (df['macd_slope_up']) &
        (df['any_trigger']) &
        (df['ema20_trending']) &
        (df['liq_ok'])
    )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date',          type=str,   required=True,  help="Data w formacie YYYY-MM-DD")
    parser.add_argument('--cap',           type=float, required=True,  help="Twoj aktualny kapital")
    parser.add_argument('--conservative',  action='store_true',        help="Tryb konserwatywny: risk_dist = CHANDELIER_MULT x ATR")
    parser.add_argument('--strict-market', action='store_true',        help="Wzmocniony filtr rynkowy: EMA20>EMA50 ORAZ close>EMA50 (zalecany dla UK/JP)")
    args = parser.parse_args()

    if args.conservative:
        risk_dist_mult = CHANDELIER_MULT
        mode_label     = f"KONSERWATYWNY (risk_dist = {CHANDELIER_MULT}xATR)"
    else:
        risk_dist_mult = RISK_DIST_NORMAL
        mode_label     = f"NORMALNY (risk_dist = {RISK_DIST_NORMAL}xATR)"

    market_filter_label = "EMA20>EMA50 + close>EMA50" if args.strict_market else "EMA20>EMA50"

    t_date = pd.to_datetime(args.date).normalize()

    print(f"Pobieranie danych rynkowych dla dnia {t_date.date()}...")
    m_data = yf.download(
        '^GSPC',
        start=t_date - timedelta(days=800),
        end=t_date + timedelta(days=5),
        progress=False
    )
    if m_data.empty:
        print("BLAD: Nie udalo sie pobrac danych S&P 500.")
        return

    m_data = m_data.reset_index()
    m_data.columns = [
        col[0].lower() if isinstance(col, tuple) else col.lower()
        for col in m_data.columns
    ]
    m_data['date'] = pd.to_datetime(m_data['date']).dt.tz_localize(None).dt.normalize()
    m_data['ema20'] = calc_ema(m_data['close'], 20)
    m_data['ema50'] = calc_ema(m_data['close'], 50)

    m_row = m_data[m_data['date'] <= t_date].tail(1)
    if m_row.empty:
        print("BLAD: Brak danych rynkowych dla podanej daty.")
        return

    actual_date = m_row['date'].iloc[0]
    m_ema20     = m_row['ema20'].iloc[0]
    m_ema50     = m_row['ema50'].iloc[0]
    m_close     = m_row['close'].iloc[0]

    # ── Filtr rynkowy ─────────────────────────────────────────────────────────
    if args.strict_market:
        m_ok = (m_ema20 > m_ema50) and (m_close > m_ema50)
    else:
        m_ok = m_ema20 > m_ema50

    print(f"Sesja: {actual_date.date()} | "
          f"SPX close: {m_close:.0f} | EMA20: {m_ema20:.0f} | EMA50: {m_ema50:.0f} | "
          f"Trend: {'HOSSA (OK)' if m_ok else 'BESSA (STOP)'} | "
          f"Filtr: {market_filter_label} | Tryb: {mode_label}")

    if not m_ok:
        print("Skaner zatrzymany — rynek nie spelnia warunkow filtru rynkowego.")
        if args.strict_market and (m_ema20 > m_ema50) and not (m_close > m_ema50):
            print(f"  → EMA20>EMA50: TAK, ale close ({m_close:.0f}) < EMA50 ({m_ema50:.0f})")
            print(f"  → Rynek w odbiciu ale cena nadal ponizej EMA50 — wzmocniony filtr blokuje.")
        return

    ticker_df = get_sp500_tickers()
    if ticker_df is None:
        print("BLAD: Nie udalo sie pobrac listy tickerow.")
        return

    if EXCLUDED_SECTORS:
        ticker_df = ticker_df[~ticker_df['sector'].isin(EXCLUDED_SECTORS)]

    tickers    = ticker_df['ticker'].tolist()
    sector_map = ticker_df.set_index('ticker')['sector'].to_dict()

    print(f"Skanowanie {len(tickers)} spolek...")
    raw_all = yf.download(
        tickers,
        start=actual_date - timedelta(days=800),
        end=actual_date + timedelta(days=5),
        group_by='ticker',
        progress=True
    )

    signals = []
    for ticker in tqdm(tickers, desc="Przetwarzanie"):
        try:
            s_df = raw_all[ticker].dropna(subset=['Close']).copy().reset_index()
            s_df.columns = [col.lower() for col in s_df.columns]
            s_df['date'] = pd.to_datetime(s_df['date']).dt.tz_localize(None).dt.normalize()

            res = calculate_indicators(s_df, m_data)
            if res is None:
                continue

            r = res[res['date'] == actual_date]
            if r.empty or not r.iloc[0]['setup_ok']:
                continue

            row = r.iloc[0]

            # Logika ryzyka v23.1
            risk_mult = 1.0
            vol_ratio = row['volume'] / row['vol_ma']
            if row['mansfield'] > MANSFIELD_RISK_BONUS_THRESHOLD:    risk_mult += 0.5
            if MIN_VOLUME_RISK_BONUS <= vol_ratio <= MAX_VOLUME_RISK_BONUS:  risk_mult += 0.5

            total_risk_pct = BASE_RISK_UNIT * risk_mult
            risk_dist      = risk_dist_mult * row['atr']

            if risk_dist <= 0:
                logging.debug(f"{ticker}: risk_dist <= 0, pomijam")
                continue

            shares_by_risk     = int((args.cap * total_risk_pct) / risk_dist)
            shares_by_notional = int((args.cap * MAX_NOTIONAL_PCT) // row['close'])
            shares             = min(shares_by_risk, shares_by_notional)

            if shares <= 0:
                logging.debug(f"{ticker}: shares=0, pomijam")
                continue

            sl_price      = row['close'] - (CHANDELIER_MULT * row['atr'])
            real_risk_usd = (row['close'] - sl_price) * shares
            real_risk_pct = real_risk_usd / args.cap * 100

            cap_flag = 'NOTIONAL' if (shares == shares_by_notional and shares_by_risk > shares_by_notional) else 'risk'

            trig_list = [
                name for cond, name in zip(
                    [row['t_pinbar'], row['t_engulf'], row['t_breakout'], row['t_gapgo'], row['t_momentum']],
                    ["PinBar", "Engulfing", "Breakout", "GapGo", "Momentum"]
                ) if cond
            ]

            signals.append({
                'Ticker':    ticker,
                'Sektor':    sector_map.get(ticker, 'Unknown'),
                'Mansfield': round(row['mansfield'], 3),
                'RealRisk%': round(real_risk_pct, 2),
                'Limit':     cap_flag,
                'Sygnal':    ", ".join(trig_list),
                'Cena':      round(row['close'], 2),
                'SL':        round(sl_price, 2),
                'Akcje':     shares,
                'Pozycja$':  round(shares * row['close'], 2),
                'Notional%': round((shares * row['close'] / args.cap) * 100, 1),
            })

        except Exception as e:
            logging.debug(f"{ticker}: {e}")
            continue

    print("\n" + "=" * 125)
    if signals:
        df_res = pd.DataFrame(signals).sort_values(['RealRisk%', 'Mansfield'], ascending=[False, False])
        print(f"KANDYDACI | Kapital: {args.cap}$ | Filtr: {market_filter_label} | Tryb: {mode_label} | Limit pozycji: {MAX_NOTIONAL_PCT*100}%")
        print("-" * 125)
        print(df_res.to_string(index=False))
        print("-" * 125)
        print(f"NOTIONAL = pozycja przycieta przez limit {MAX_NOTIONAL_PCT*100}%")
        print(f"risk     = pozycja wyznaczona przez ryzyko (nie trafila w limit notional)")
    else:
        print("Brak sygnalow spelniajacych kryteria.")
    print("=" * 125)


if __name__ == "__main__":
    main()
