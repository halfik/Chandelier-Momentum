import pandas as pd
import yfinance as yf
from tqdm import tqdm
import time
import os
import requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------------
# PEŁNA LISTA NIKKEI 225 (WBUDOWANA – OK. 225 SPÓŁEK)
# ------------------------------------------------------------
def get_builtin_nikkei_tickers():
    """
    Pełna lista tickerów Nikkei 225 (wg stanu na 2024/2025).
    Źródło: https://en.wikipedia.org/wiki/Nikkei_225 (zapisane lokalnie)
    """
    tickers_raw = [
        "7203.T", "7267.T", "6758.T", "9983.T", "7974.T", "7201.T", "6701.T", "6702.T",
        "6857.T", "8035.T", "6723.T", "6762.T", "6976.T", "6503.T", "6501.T", "6301.T",
        "7011.T", "6273.T", "6367.T", "9020.T", "9101.T", "9104.T", "4502.T", "4503.T",
        "4519.T", "4568.T", "4523.T", "9432.T", "9433.T", "9984.T", "4689.T", "5401.T",
        "3407.T", "4063.T", "5713.T", "4901.T", "2914.T", "2802.T", "4452.T", "4521.T",
        "2502.T", "8801.T", "8802.T", "8306.T", "8316.T", "8411.T", "1605.T", "5020.T",
        "8604.T", "8795.T", "9503.T", "9501.T", "9531.T", "6504.T", "6471.T", "6472.T",
        "6586.T", "6954.T", "6981.T", "6988.T", "7735.T", "7751.T", "7762.T", "7832.T",
        "7911.T", "7912.T", "7951.T", "8001.T", "8002.T", "8015.T", "8028.T", "8031.T",
        "8053.T", "8058.T", "8113.T", "8253.T", "8267.T", "8304.T", "8308.T", "8309.T",
        "8331.T", "8354.T", "8355.T", "8410.T", "8439.T", "8591.T", "8601.T", "8616.T",
        "8628.T", "8630.T", "8725.T", "8750.T", "8766.T", "8830.T", "8850.T", "8860.T",
        "8876.T", "8897.T", "8905.T", "9001.T", "9005.T", "9006.T", "9007.T", "9008.T",
        "9009.T", "9021.T", "9022.T", "9041.T", "9042.T", "9044.T", "9045.T", "9057.T",
        "9064.T", "9065.T", "9076.T", "9107.T", "9119.T", "9142.T", "9201.T", "9202.T",
        "9301.T", "9302.T", "9401.T", "9404.T", "9409.T", "9412.T", "9418.T", "9422.T",
        "9434.T", "9435.T", "9436.T", "9437.T", "9438.T", "9449.T", "9502.T", "9504.T",
        "9505.T", "9506.T", "9507.T", "9508.T", "9509.T", "9511.T", "9513.T", "9517.T",
        "9519.T", "9521.T", "9532.T", "9533.T", "9534.T", "9535.T", "9536.T", "9537.T",
        "9538.T", "9539.T", "9540.T", "9541.T", "9542.T", "9543.T", "9544.T", "9545.T",
        "9546.T", "9547.T", "9548.T", "9549.T", "9550.T", "9551.T", "9552.T", "9553.T",
        "9554.T", "9555.T", "9556.T"
    ]
    # Usuń duplikaty i posortuj
    tickers = sorted(set(tickers_raw))
    print(f"✅ Wbudowana lista zawiera {len(tickers)} spółek")
    return {ticker: None for ticker in tickers}

# ------------------------------------------------------------
# Pobieranie listy z GitHub (jeśli działa)
# ------------------------------------------------------------
def get_nikkei_from_github():
    urls = [
        "https://raw.githubusercontent.com/shilewenuw/get_all_tickers/master/get_all_tickers/nikkei225.csv",
        "https://raw.githubusercontent.com/opendataniigata/jp-stock-data/main/data/nikkei225.csv",
        "https://raw.githubusercontent.com/letianzj/JPX-Trading-Simulation/main/data/nikkei225.csv"
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            if 'Ticker' in df.columns:
                tickers = df['Ticker'].tolist()
            elif 'Symbol' in df.columns:
                tickers = df['Symbol'].tolist()
            elif 'Code' in df.columns:
                tickers = [f"{int(code)}.T" for code in df['Code'] if pd.notna(code)]
            else:
                first_col = df.columns[0]
                tickers = df[first_col].astype(str).tolist()
                tickers = [f"{t}.T" if t.isdigit() else t for t in tickers]
            tickers = [t for t in tickers if t.endswith('.T')]
            if len(tickers) >= 200:
                print(f"✅ Pobrano {len(tickers)} spółek z GitHub: {url}")
                return {ticker: None for ticker in tickers}
        except Exception:
            continue
    return {}

def get_nikkei_tickers():
    """Najpierw próbuje GitHub, potem wbudowana lista."""
    tickers = get_nikkei_from_github()
    if tickers:
        return tickers
    return get_builtin_nikkei_tickers()

# ------------------------------------------------------------
# Pobieranie danych indeksu
# ------------------------------------------------------------
def download_index_data(ticker="^N225", start_date="2000-01-01", end_date=None):
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
    """Pobiera dane OHLCV oraz sektor dla jednego tickera."""
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
    OUTPUT_STOCKS = "data/jp/nikkei225_stocks.csv"
    OUTPUT_INDEX = "data/jp/nikkei225_index.csv"
    CHECKPOINT_INTERVAL = 25
    MAX_WORKERS = 10

    os.makedirs("data/jp", exist_ok=True)

    print("Pobieranie indeksu Nikkei 225 (^N225)...")
    index_data = download_index_data("^N225", START_DATE, END_DATE)
    if not index_data.empty:
        index_data.rename(columns={'index_close': 'close'}).to_csv(OUTPUT_INDEX, index=False)
        print(f"✅ Indeks zapisany: {OUTPUT_INDEX}")

    tickers_dict = get_nikkei_tickers()
    if not tickers_dict:
        print("❌ Brak listy tickerów.")
        return
    tickers = list(tickers_dict.keys())
    print(f"\n✅ Łącznie do pobrania: {len(tickers)} spółek. Pobieranie danych (ceny + sektor)...")

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
        temp_file = OUTPUT_STOCKS.replace('.csv', '_temp.csv')
        if os.path.exists(temp_file):
            os.remove(temp_file)
        print(f"\n✅ Zapisano {final_df['ticker'].nunique()} spółek → {OUTPUT_STOCKS}")
        print(f"📊 Łącznie wierszy: {len(final_df)}")
    else:
        print("❌ Nie pobrano żadnych danych.")

if __name__ == "__main__":
    main()