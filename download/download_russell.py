import pandas as pd
import yfinance as yf
from tqdm import tqdm
import time
import os

def get_russell_tickers_from_github():
    url = "https://raw.githubusercontent.com/ikoniaris/Russell2000/master/russell_2000_components.csv"
    df = pd.read_csv(url)
    # Lista zawiera kolumnę 'Ticker' z symbolami
    tickers = df['Ticker'].tolist()
    # Zwracamy słownik ticker -> 'Unknown' jako sektor (lub możesz dodać własną klasyfikację)
    return {ticker: 'Unknown' for ticker in tickers}

def download_stock_data(ticker, sector, start_date="2000-01-01", end_date=None, retries=3):
    """Pobiera dane OHLCV dla pojedynczej spółki."""
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, auto_adjust=True)
            if data is None or data.empty:
                return None
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
            if hasattr(data['date'].dt, 'tz') and data['date'].dt.tz is not None:
                data['date'] = data['date'].dt.tz_localize(None)

            data = data.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            data = data[data['volume'] > 0]
            if len(data) < 100:
                return None
            return data
        except Exception:
            time.sleep(1)
    return None

def download_index_data(ticker="^RUT", start_date="2000-01-01", end_date=None, retries=3):
    """Pobiera dane indeksu Russell 2000 (^RUT) lub innego."""
    for attempt in range(retries):
        try:
            idx = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if idx is None or idx.empty:
                continue
            idx = idx.reset_index()
            if isinstance(idx.columns, pd.MultiIndex):
                idx.columns = [col[0].lower() for col in idx.columns]
            else:
                idx.columns = [col.lower() for col in idx.columns]

            idx = idx[['date', 'close']].rename(columns={'close': 'index_close'})
            idx['date'] = pd.to_datetime(idx['date'])
            # Usunięcie timezone
            if hasattr(idx['date'].dt, 'tz') and idx['date'].dt.tz is not None:
                idx['date'] = idx['date'].dt.tz_localize(None)

            print(f"✅ Pobrano dane indeksu {ticker}: {len(idx)} wierszy")
            return idx
        except Exception as e:
            print(f"  Próba {attempt + 1}: {e}")
            time.sleep(2)
    return pd.DataFrame(columns=['date', 'index_close'])

def main():
    START_DATE = "2000-01-01"
    END_DATE = None
    OUTPUT_STOCKS = "data/us/russell2000_stocks.csv"
    OUTPUT_INDEX = "data/us/russell2000_index.csv"

    os.makedirs("data/us", exist_ok=True)

    # 1. Pobranie indeksu
    print("Pobieranie danych indeksu Russell 2000 (^RUT)...")
    index_data = download_index_data("^RUT", start_date=START_DATE, end_date=END_DATE)

    if not index_data.empty:
        index_out = index_data.rename(columns={'index_close': 'close'})
        index_out.to_csv(OUTPUT_INDEX, index=False)
        print(f"✅ Indeks zapisany: {OUTPUT_INDEX}")

    # 2. Pobranie listy spółek
    print("\nPobieranie listy spółek Russell 2000 z GitHub...")
    tickers_dict = get_russell_tickers_from_github()
    print(f"✅ Znaleziono {len(tickers_dict)} spółek.")

    if not tickers_dict:
        print("❌ Nie udało się uzyskać żadnej listy tickerów. Kończę.")
        return

    # 3. Pobieranie danych historycznych dla każdej spółki
    print(f"\nPobieranie danych dla {len(tickers_dict)} spółek...")
    all_data = []

    for ticker, sector in tqdm(tickers_dict.items(), desc="Pobieranie"):
        df = download_stock_data(ticker, sector, start_date=START_DATE, end_date=END_DATE)
        if df is not None:
            all_data.append(df)
        time.sleep(0.1)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.sort_values(['date', 'ticker']).reset_index(drop=True)
        final_df.to_csv(OUTPUT_STOCKS, index=False)
        print(f"\n✅ Zapisano {final_df['ticker'].nunique()} spółek do {OUTPUT_STOCKS}")
    else:
        print("❌ Nie udało się pobrać danych dla żadnej spółki.")

if __name__ == "__main__":
    main()
