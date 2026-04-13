import pandas as pd
import yfinance as yf
from tqdm import tqdm
import time
import os
import requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------------
# Pobieranie listy spółek NASDAQ 100
# ------------------------------------------------------------
def get_nasdaq_from_wikipedia():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        time.sleep(1)
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        for table in tables:
            if 'Ticker' in table.columns or 'Symbol' in table.columns:
                ticker_col = 'Ticker' if 'Ticker' in table.columns else 'Symbol'
                tickers = table[ticker_col].dropna().tolist()
                tickers = [t.split('[')[0].strip() for t in tickers]
                if len(tickers) >= 90:
                    print(f"✅ Pobrano {len(tickers)} spółek z Wikipedii")
                    return {ticker: None for ticker in tickers}
    except Exception as e:
        print(f"⚠️ Wikipedia: {e}")
    return {}

def get_nasdaq_from_github():
    urls = [
        "https://raw.githubusercontent.com/fja05680/sp500/master/Nasdaq.csv",
        "https://raw.githubusercontent.com/shilewenuw/get_all_tickers/master/get_all_tickers/nasdaq100.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            if 'Ticker' in df.columns:
                tickers = df['Ticker'].tolist()
            elif 'Symbol' in df.columns:
                tickers = df['Symbol'].tolist()
            else:
                tickers = df.iloc[:, 0].tolist()
            tickers = [str(t).strip().upper() for t in tickers if pd.notna(t)]
            if len(tickers) >= 90:
                print(f"✅ Pobrano {len(tickers)} spółek z GitHub: {url}")
                return {ticker: None for ticker in tickers}
        except Exception:
            continue
    return {}

def get_fallback_tickers_nasdaq():
    fallback = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "PEP",
        "COST", "CSCO", "ADBE", "TXN", "AMGN", "INTC", "QCOM", "AMD", "HON",
        "SBUX", "BKNG", "MDLZ", "ADI", "ISRG", "GILD", "VRTX", "LRCX", "MU"
    ]
    print(f"⚠️ Używam zapasowej listy ({len(fallback)} spółek)")
    return {ticker: None for ticker in fallback}

def get_nasdaq_tickers():
    tickers = get_nasdaq_from_wikipedia()
    if tickers:
        return tickers
    tickers = get_nasdaq_from_github()
    if tickers:
        return tickers
    return get_fallback_tickers_nasdaq()

# ------------------------------------------------------------
# Pobieranie danych (ceny + sektor)
# ------------------------------------------------------------
def download_index_data(ticker="^NDX", start_date="2000-01-01", end_date=None):
    for attempt in range(3):
        try:
            idx = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if idx.empty:
                continue
            idx = idx.reset_index()
            if isinstance(idx.columns, pd.MultiIndex):
                idx.columns = [col[0].lower() for col in idx.columns]
            else:
                idx.columns = [col.lower() for col in idx.columns]
            idx = idx[['date', 'close']].rename(columns={'close': 'index_close'})
            idx['date'] = pd.to_datetime(idx['date'])
            if hasattr(idx['date'].dt, 'tz'):
                idx['date'] = idx['date'].dt.tz_localize(None)
            print(f"✅ Indeks {ticker}: {len(idx)} wierszy")
            return idx
        except Exception as e:
            time.sleep(2)
    return pd.DataFrame(columns=['date', 'index_close'])

def fetch_stock_with_sector(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, auto_adjust=True)
        if data.empty or len(data) < 100:
            return None

        sector = stock.info.get('sector', 'Unknown')

        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0].lower() for col in data.columns]
        else:
            data.columns = [col.lower() for col in data.columns]

        required = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required):
            return None

        data = data[required].copy()
        data['ticker'] = ticker
        data['sector'] = sector
        data['date'] = pd.to_datetime(data['date'])
        if hasattr(data['date'].dt, 'tz'):
            data['date'] = data['date'].dt.tz_localize(None)
        data = data.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        data = data[data['volume'] > 0]
        return data
    except Exception:
        return None

def save_checkpoint(dataframes, filename):
    if not dataframes:
        return
    combined = pd.concat(dataframes, ignore_index=True)
    combined.to_csv(filename, index=False)

def worker(ticker, start_date, end_date):
    return fetch_stock_with_sector(ticker, start_date, end_date)

# ------------------------------------------------------------
# Główna funkcja
# ------------------------------------------------------------
def main():
    START_DATE = "2000-01-01"
    END_DATE = None
    OUTPUT_STOCKS = "data/us/nasdaq100_stocks.csv"
    OUTPUT_INDEX = "data/us/nasdaq100_index.csv"
    CHECKPOINT_INTERVAL = 20
    MAX_WORKERS = 10

    os.makedirs("data/us", exist_ok=True)

    print("Pobieranie indeksu NASDAQ 100 (^NDX)...")
    index_data = download_index_data("^NDX", START_DATE, END_DATE)
    if not index_data.empty:
        index_data.rename(columns={'index_close': 'close'}).to_csv(OUTPUT_INDEX, index=False)
        print(f"✅ Indeks zapisany: {OUTPUT_INDEX}")

    tickers_dict = get_nasdaq_tickers()
    if not tickers_dict:
        print("❌ Brak listy tickerów.")
        return
    tickers = list(tickers_dict.keys())
    print(f"\n✅ Znaleziono {len(tickers)} spółek. Pobieranie danych (ceny + sektor)...")

    all_data = []
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {executor.submit(worker, t, START_DATE, END_DATE): t for t in tickers}
        with tqdm(total=len(future_to_ticker), desc="Pobieranie") as pbar:
            for future in as_completed(future_to_ticker):
                result = future.result()
                if result is not None:
                    all_data.append(result)
                completed += 1
                pbar.update(1)
                if completed % CHECKPOINT_INTERVAL == 0 and all_data:
                    save_checkpoint(all_data, OUTPUT_STOCKS.replace('.csv', '_temp.csv'))
                    print(f"\n  💾 Checkpoint: {completed}/{len(tickers)}")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.sort_values(['date', 'ticker']).reset_index(drop=True)
        final_df.to_csv(OUTPUT_STOCKS, index=False)
        print(f"\n✅ Zapisano {final_df['ticker'].nunique()} spółek → {OUTPUT_STOCKS}")
        print(f"📊 Łącznie wierszy: {len(final_df)}")
        temp_file = OUTPUT_STOCKS.replace('.csv', '_temp.csv')
        if os.path.exists(temp_file):
            os.remove(temp_file)
    else:
        print("❌ Nie pobrano żadnych danych.")

if __name__ == "__main__":
    main()