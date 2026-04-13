import argparse
import yfinance as yf
from datetime import timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')

# ── Flagi CLI ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--diag', action='store_true', help='Włącz diagnostykę filtrów i fat tail')
args = parser.parse_args()
DIAG = args.diag

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURACJA v23.0 (1% Base + Bonuses | FULL LOGGING)
# ══════════════════════════════════════════════════════════════════════════════
CSV_PATH = "data/spx/stocks_with_sector.csv"
SPX_PATH = "data/spx/index.csv"
#CSV_PATH = "data/nasdaq/stocks.csv"
#SPX_PATH = "data/nasdaq/index.csv"
#CSV_PATH = "data/jp/stocks.csv"
#SPX_PATH = "data/jp/index.csv"
#CSV_PATH = "data/uk/stocks_ftse.csv"
#SPX_PATH = "data/uk/index.csv"
INITIAL_CAPITAL = 10000.0
MAX_POSITIONS = 10
MAX_PER_SECTOR = 3
SLIPPAGE = 0.005
CHANDELIER_MULT = 6.0
BASE_RISK_UNIT = 0.01 #1%
BASE_RISK_BONUS = 0.005
MANSFIELD_PERIOD = 26
MANSFIELD_THRESHOLD = -0.02
MANSFIELD_RISK_BONUS_THRESHOLD = 0.1
MIN_VOLUME_RISK_BONUS = 1.2
MAX_VOLUME_RISK_BONUS = 2
MAX_NOTIONAL_PCT = 0.10 #15%
TIME_STOP_DAYS = 365
DELISTING_PENALTY = 0.50
DELISTING_DAYS = 14
MIN_VOLUME = 10_000_000
TECH_DAYS_DATA = 400
MC_SIM = 100000
SKIP_CHANCE = 0.1 #10%
#EXCLUDED_SECTORS={} #jp
EXCLUDED_SECTORS = {'Financials', 'Energy'} #us
#EXCLUDED_SECTORS = {'Communication Services', 'Utilities', 'Financials', 'Energy'} #uk

# ══════════════════════════════════════════════════════════════════════════════
# [1/4] DANE & WSKAŹNIKI TECHNICZNE
# ══════════════════════════════════════════════════════════════════════════════

def calc_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

# Oblicz ADX dla indeksu
def calc_adx(h, l, c, period=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    up   = h - h.shift()
    down = l.shift() - l
    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    plus_di  = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di))
    return dx.ewm(span=period, adjust=False).mean()

if not os.path.exists(CSV_PATH) or not os.path.exists(SPX_PATH):
    print("BŁĄD: Brakuje plików CSV!")
    exit()

spx = pd.read_csv(SPX_PATH)
spx.columns = [x.lower().strip() for x in spx.columns]
s_date_col = 'date' if 'date' in spx.columns else spx.columns[0]
s_close_col = 'close' if 'close' in spx.columns else spx.columns[4]

spx[s_date_col] = pd.to_datetime(spx[s_date_col], utc=True).dt.tz_localize(None).dt.normalize()
spx = spx.sort_values(s_date_col).drop_duplicates(s_date_col)
spx_idx = spx.set_index(s_date_col)[s_close_col]

spx['ema20d'] = spx[s_close_col].ewm(span=20, adjust=False).mean()
spx['ema50d'] = spx[s_close_col].ewm(span=50, adjust=False).mean()
#spx['market_ok'] = (spx['ema20d'] > spx['ema50d']) & (spx[s_close_col] > spx['ema50d']) #other than us
spx['market_ok'] = spx['ema20d'] > spx['ema50d'] #us
market_dict = spx.set_index(s_date_col)['market_ok'].to_dict()
market_dict_close = spx.set_index(s_date_col)[s_close_col].to_dict()
market_dict_ema50 = spx.set_index(s_date_col)['ema50d'].to_dict()

df = pd.read_csv(CSV_PATH)
df.columns = [x.lower().strip() for x in df.columns]
if 'name' in df.columns:
    df.rename(columns={'name': 'ticker'}, inplace=True)
elif 'symbol' in df.columns:
    df.rename(columns={'symbol': 'ticker'}, inplace=True)
df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None).dt.normalize()
if 'sector' not in df.columns or df['sector'].isna().all():
    df['sector'] = 'Unknown'

df = df[~df['sector'].isin(EXCLUDED_SECTORS)]

if DIAG:
    print(f"\nSpółek w surowych danych: {df['ticker'].nunique()}")
    print(f"Zakres dat: {df['date'].min().date()} → {df['date'].max().date()}")

diag_passed_length = 0
diag_passed_join = 0
diag_filter_counts = {
    'ema200': 0, 'ema20': 0, 'mansfield': 0,
    'macd_slope': 0, 'trigger': 0, 'ema20_trending': 0,
    'liq_ok': 0, 'all_ok': 0
}

processed = []
for tk, t in tqdm(df.groupby('ticker'), desc="Analiza Techniczna"):
    if len(t) < TECH_DAYS_DATA:
        continue
    diag_passed_length += 1

    t = t.copy().sort_values('date')
    t = t.join(spx_idx.rename('spx_c'), on='date', how='inner')
    if t.empty:
        continue
    diag_passed_join += 1

    c, h, l, o = t['close'], t['high'], t['low'], t['open']

    t['ema20']  = calc_ema(c, 20)
    t['ema200'] = calc_ema(c, 200)
    t['atr']    = (pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1)
                   .max(axis=1).rolling(14).mean())

    rs = t['close'] / t['spx_c']
    t['mansfield'] = (rs / rs.rolling(MANSFIELD_PERIOD * 5).mean()) - 1

    macd_line = calc_ema(c, 60) - calc_ema(c, 130)
    t['macd_slope_up'] = (macd_line > macd_line.shift(1)).rolling(3).sum() >= 2

    t['vol_ma'] = t['volume'].rolling(20).mean()
    body  = abs(c - o)
    vol_f = t['volume'] > t['vol_ma']

    t_pinbar   = ((o.combine(c, min) - l) >= 2 * body) & vol_f
    t_engulf   = ((c.shift(1) < o.shift(1)) & (c > o) & (c > o.shift(1)) &
                  (o < c.shift(1)) & (t['volume'] > 1.2 * t['vol_ma']))
    t_breakout = (c > h.shift(1).rolling(10).max()) & vol_f
    t_gapgo    = (o > h.shift(1)) & (c > o) & (t['volume'] > 1.5 * t['vol_ma'])
    t_momentum = (t['mansfield'] > 0.5) & (c > o) & (c >= (h - 0.1 * (h - l)))

    t['any_trigger']    = t_pinbar | t_engulf | t_breakout | t_gapgo | t_momentum
    t['ema20_trending'] = t['ema20'] > t['ema20'].shift(5)
    t['liq_ok']         = (c * t['volume']).rolling(20).mean() > MIN_VOLUME

    f_ema200    = c > t['ema200']
    f_ema20     = c > t['ema20']
    f_mansfield = t['mansfield'] > MANSFIELD_THRESHOLD
    f_macd      = t['macd_slope_up']
    f_trigger   = t['any_trigger']
    f_trending  = t['ema20_trending']
    f_liq       = t['liq_ok']

    t['setup_ok'] = f_ema200 & f_ema20 & f_mansfield & f_macd & f_trigger & f_trending & f_liq

    if DIAG:
        if f_ema200.any():      diag_filter_counts['ema200'] += 1
        if f_ema20.any():       diag_filter_counts['ema20'] += 1
        if f_mansfield.any():   diag_filter_counts['mansfield'] += 1
        if f_macd.any():        diag_filter_counts['macd_slope'] += 1
        if f_trigger.any():     diag_filter_counts['trigger'] += 1
        if f_trending.any():    diag_filter_counts['ema20_trending'] += 1
        if f_liq.any():         diag_filter_counts['liq_ok'] += 1
        if t['setup_ok'].any(): diag_filter_counts['all_ok'] += 1

    processed.append(t.dropna(subset=['atr', 'mansfield', 'vol_ma']))

if DIAG:
    print(f"\n{'='*60}")
    print(f"DIAGNOSTYKA FILTRÓW")
    print(f"{'='*60}")
    print(f"Spółek po filtrze min. {TECH_DAYS_DATA} dni danych : {diag_passed_length}")
    print(f"Spółek po join z indeksem              : {diag_passed_join}")
    print(f"\nIle spółek ma CHOĆBY 1 dzień spełniający filtr:")
    for fname, cnt in diag_filter_counts.items():
        bar = '█' * int(cnt / max(diag_passed_join, 1) * 40)
        print(f"  {fname:<20}: {cnt:>4} ({cnt/max(diag_passed_join,1)*100:>5.1f}%)  {bar}")
    print(f"\n→ Spółek z PEŁNYM sygnałem (setup_ok): {diag_filter_counts['all_ok']}")
    print(f"{'='*60}\n")

if not processed:
    print("BRAK DANYCH po filtrach. Sprawdź dane wejściowe.")
    exit()

data = pd.concat(processed).sort_values(['date', 'ticker'])
all_tickers = data['ticker'].unique().tolist()
print(f"Przygotowano dane dla {len(all_tickers)} spółek.")

if DIAG:
    signals_per_year = (data[data['setup_ok']]
                        .groupby(data[data['setup_ok']]['date'].dt.year)['ticker'].count())
    print(f"\nSygnały setup_ok per rok:")
    for yr, cnt in signals_per_year.items():
        print(f"  {yr}: {cnt} sygnałów")

td_dict = {tk: grp.set_index('date') for tk, grp in data.groupby('ticker')}

# ══════════════════════════════════════════════════════════════════════════════
# [2/4] BACKTEST Z PEŁNYM LOGOWANIEM
# ══════════════════════════════════════════════════════════════════════════════

cash, positions, log = INITIAL_CAPITAL, {}, []
equity_curve = []
all_dates = sorted(data['date'].unique())
yearly_start_equity = {all_dates[0].year: INITIAL_CAPITAL}


def make_log_entry(p, tk, exit_date, exit_price, pnl, note):
    return {
        'ticker': tk,
        'sector': p['sector'],
        'entry_date': p['entry_date'],
        'exit_date': exit_date,
        'days_held': (exit_date - p['entry_date']).days,
        'entry_price': round(p['entry'], 4),
        'exit_price': round(exit_price, 4),
        'sl_at_exit': round(p['sl'], 4),
        'sz': p['sz'],
        'pnl': round(pnl, 4),
        'pnl_pct': round(pnl / (p['entry'] * p['sz']) * 100, 2),
        'win': 1 if pnl > 0 else 0,
        'note': note,
        'total_risk_pct': p.get('total_risk_pct', 0),
        'mansfield_at_entry': round(p.get('mansfield_at_entry', 0), 4),
        'ema20_at_entry': round(p.get('ema20_at_entry', 0), 4),
        'overext_at_entry': round(p.get('overext_at_entry', 0), 4),
        'equity_at_entry': round(p['equity_at_entry'], 2),
    }

for today in tqdm(all_dates, desc="Backtest"):
    curr_year = today.year
    to_close, current_pos_val = [], 0
    if curr_year not in yearly_start_equity and equity_curve:
        yearly_start_equity[curr_year] = equity_curve[-1]

    for tk, p in list(positions.items()):
        if today in td_dict[tk].index:
            r = td_dict[tk].loc[today]
            p['last_seen'] = today
            exit_price, note, days_held = None, None, (today - p['entry_date']).days

            if r['open'] <= p['sl']:
                exit_price, note = r['open'], 'SL_GAP'
            elif r['low'] <= p['sl']:
                exit_price, note = p['sl'], 'SL_TOUCH'
            elif days_held > TIME_STOP_DAYS:
                exit_price, note = r['open'], 'TIME_STOP'

            if exit_price is not None:
                pnl = (exit_price - p['entry']) * p['sz']
                cash += exit_price * p['sz']
                log.append(make_log_entry(p, tk, today, exit_price, pnl, note))
                to_close.append(tk)
            else:
                if r['high'] > p['hi_since_entry']:
                    p['hi_since_entry'] = r['high']
                    p['sl'] = max(p['sl'], p['hi_since_entry'] - (CHANDELIER_MULT * r['atr']))
                p['last_price'] = r['close']
                current_pos_val += p['sz'] * r['close']
        else:
            days_missing = (today - p.get('last_seen', p['entry_date'])).days
            if days_missing > DELISTING_DAYS:
                exit_price = p['last_price'] * DELISTING_PENALTY
                pnl = (exit_price - p['entry']) * p['sz']
                cash += exit_price * p['sz']
                log.append(make_log_entry(p, tk, today, exit_price, pnl, 'DELISTED'))
                to_close.append(tk)
            else:
                current_pos_val += p['sz'] * p['last_price']

    for tk in to_close:
        del positions[tk]

    total_equity = cash + current_pos_val
    equity_curve.append(total_equity)

    if not market_dict.get(today, False):
        continue

    candidates = data[(data['date'] == today) & (data['setup_ok'])].sort_values('mansfield', ascending=False)

    for _, s in candidates.iterrows():
        if s['ticker'] in positions:
            continue
        if len(positions) >= MAX_POSITIONS:
            break
        if sum(1 for pos in positions.values() if pos['sector'] == s['sector']) >= MAX_PER_SECTOR:
            continue

        future = td_dict[s['ticker']][td_dict[s['ticker']].index > today].head(1)
        if future.empty:
            continue

        entry_p = future.iloc[0]['open'] * (1 + SLIPPAGE)
        dd_risk_mult = 1
        #peak_equity = max(equity_curve) if equity_curve else INITIAL_CAPITAL
        #current_dd = (total_equity - peak_equity) / peak_equity
        #if current_dd < -0.15:  # drawdown > 15%
        #    dd_risk_mult = 0.5
        #elif current_dd < -0.08:  # drawdown > 8%
        #    dd_risk_mult = 0.75

        #m_close = market_dict_close.get(today, None)
        #m_ema50 = market_dict_ema50.get(today, None)

        #if m_close is not None and m_ema50 is not None and m_close < m_ema50:
        #    dd_risk_mult = 0.5  # rynek w szarej strefie — handlujemy ale ostrożniej

        rel_vol   = s['volume'] / s['vol_ma']
        bonus_msf = BASE_RISK_BONUS if s['mansfield'] > MANSFIELD_RISK_BONUS_THRESHOLD else 0
        bonus_vol = BASE_RISK_BONUS if MIN_VOLUME_RISK_BONUS <= rel_vol <= MAX_VOLUME_RISK_BONUS else 0
        total_risk_pct = (BASE_RISK_UNIT + bonus_msf + bonus_vol) * dd_risk_mult

        risk_dist = 1.2 * s['atr']
        sz = min(
            int((total_equity * total_risk_pct) / risk_dist),
            int((total_equity * MAX_NOTIONAL_PCT) // entry_p)
        )

        if sz > 0 and cash >= (entry_p * sz):
            cash -= entry_p * sz
            positions[s['ticker']] = {
                'ticker': s['ticker'],
                'sector': s['sector'],
                'entry': entry_p,
                'last_price': entry_p,
                'sl': entry_p - (CHANDELIER_MULT * s['atr']),
                'sz': sz,
                'total_risk_pct': total_risk_pct,
                'hi_since_entry': entry_p,
                'entry_date': today,
                'last_seen': today,
                'equity_at_entry': total_equity,
                'mansfield_at_entry': s['mansfield'],
                'ema20_at_entry': s['ema20'],
                'overext_at_entry': round((s['close'] - s['ema20']) / s['ema20'], 4),
            }

# ══════════════════════════════════════════════════════════════════════════════
# [3/4] RAPORTY
# ══════════════════════════════════════════════════════════════════════════════

def calc_pf(series):
    wins = series[series > 0].sum()
    loss = abs(series[series < 0].sum())
    return wins / loss if loss > 0 else np.inf

print("\n" + "=" * 125)
if not log:
    print("BRAK TRANSAKCJI.")
else:
    t_log = pd.DataFrame(log)
    t_log['year'] = t_log['exit_date'].dt.year
    eq_ser = pd.Series(equity_curve)
    mdd = ((eq_ser - eq_ser.cummax()) / eq_ser.cummax()).min() * 100
    overall_pf = calc_pf(t_log['pnl'])

    roi_pct = (equity_curve[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    print(f"v23.0 FINAL | EQUITY: {equity_curve[-1]:.2f} | ROI: {roi_pct:.2f}% | MDD: {mdd:.2f}% | PF: {overall_pf:.2f} | WIN%: {(t_log['win'].sum() / len(t_log) * 100):.1f}%")
    print("=" * 125)
    print(f"{'YEAR':<6} | {'PNL':>11} | {'ROI':>9} | {'TRADES':>8} | {'WIN%':>6} | {'PF':>6} | NOTES")
    print("-" * 125)
    for yr, y_grp in t_log.groupby('year'):
        pnl_yr, trades = y_grp['pnl'].sum(), len(y_grp)
        roi_yr = pnl_yr / yearly_start_equity.get(yr, INITIAL_CAPITAL) * 100
        wr = y_grp['win'].sum() / trades * 100
        print(f"{yr:<6} | {pnl_yr:>+11.2f} | {roi_yr:>9.2f}% | {trades:>8} | {wr:>6.1f}% | {calc_pf(y_grp['pnl']):>6.2f} | {y_grp['note'].value_counts().to_dict()}")

    print("\nTypy wyjść:")
    for nt, cnt in t_log['note'].value_counts().items():
        print(f"  {nt:<12}: {cnt:>5} transakcji | avg PnL: {t_log[t_log['note'] == nt]['pnl'].mean():>+8.2f} | avg dni: {t_log[t_log['note'] == nt]['days_held'].mean():.1f}")

    print("\nRozkład sektorów:")
    for sec, grp in t_log.groupby('sector'):
        print(f"  {sec:<35}: {len(grp):>4} transakcji | avg PnL: {grp['pnl'].mean():>+8.2f}")

    if DIAG:
        print("\nRozkład PnL% (fat tail check):")
        bins = [-100, -20, -10, -5, 0, 5, 10, 20, 50, 100, 500]
        t_log['pnl_bucket'] = pd.cut(t_log['pnl_pct'], bins=bins)
        for bucket, cnt in t_log['pnl_bucket'].value_counts().sort_index().items():
            bar = '█' * min(cnt, 80)  # cap na 80 żeby nie zawijało
            print(f"  {str(bucket):<20}: {cnt:>4}  {bar}")

        print("\nTop 10 transakcji (wg PnL%):")
        cols = ['ticker', 'sector', 'entry_date', 'exit_date', 'days_held', 'pnl', 'pnl_pct', 'note']
        print(t_log.nlargest(10, 'pnl_pct')[cols].to_string(index=False))

        print("\nWyniki per ticker (min. 3 transakcje, posortowane wg total PnL):")
        ticker_stats = t_log.groupby('ticker').agg(
            trades=('pnl', 'count'),
            total_pnl=('pnl', 'sum'),
            avg_pnl_pct=('pnl_pct', 'mean'),
            win_rate=('win', 'mean')
        ).query('trades >= 3').sort_values('total_pnl', ascending=False)
        print(ticker_stats.round(2).to_string())

        print("\nMansfield przy wejściu vs wynik:")
        t_log['msf_bucket'] = pd.cut(t_log['mansfield_at_entry'],
                                      bins=[-1, 0, 0.1, 0.2, 0.5, 1, 10])
        msf = t_log.groupby('msf_bucket')[['pnl_pct', 'win']].agg(['mean', 'count'])
        print(msf.round(2))

    t_log.to_csv("trade_log_v23_0.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# [4/4] MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nUruchamiam Monte Carlo {MC_SIM} symulacji)...")
returns_pct = (t_log['pnl'] / t_log['equity_at_entry']).values
mc_finals = []
for _ in range(MC_SIM):
    sim_returns = np.random.choice(returns_pct, size=len(returns_pct), replace=True)
    filtered_sim = [r for r in sim_returns if np.random.random() > SKIP_CHANCE]
    cap = INITIAL_CAPITAL
    for r in filtered_sim:
        cap *= (1 + r)
    mc_finals.append(cap)

mc_arr = np.array(mc_finals)
print(
    f"Mediana MC:       {np.median(mc_arr):>10.2f} USD\n"
    f"Percentyl 10%:    {np.percentile(mc_arr, 10):>10.2f} USD\n"
    f"Percentyl 90%:    {np.percentile(mc_arr, 90):>10.2f} USD\n"
    f"Szansa na stratę: {(mc_arr < INITIAL_CAPITAL).mean() * 100:.2f}%"
)
print("=" * 125)
