import pandas as pd
import yfinance as yf
from tqdm import tqdm
import time
import numpy as np


def get_sp500_tickers_and_sectors():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    df = pd.read_csv(url)
    tickers = df['Symbol'].tolist()
    sectors = df['GICS Sector'].tolist()
    ticker_to_sector = dict(zip(tickers, sectors))
    tickers_yf = [ticker.replace('.', '-') for ticker in tickers]
    return tickers_yf, ticker_to_sector


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Spłaszcza MultiIndex zwracany przez nowsze wersje yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]
    return df


def download_spy_data(start_date="2000-01-01", end_date=None, retries=3):
    for attempt in range(retries):
        try:
            spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
            if spy is None or not isinstance(spy, pd.DataFrame) or spy.empty:
                continue
            spy = flatten_columns(spy).reset_index()
            spy.columns = [col.lower() for col in spy.columns]
            spy = spy[['date', 'close']].rename(columns={'close': 'spy_close'})
            spy['date'] = pd.to_datetime(spy['date']).dt.tz_localize(None)
            return spy.reset_index(drop=True)
        except Exception:
            time.sleep(1)
    print("❌ Nie udało się pobrać danych SPY po kilku próbach.")
    return pd.DataFrame(columns=['date', 'spy_close'])


def download_stock_data(ticker, sector, start_date="2000-01-01", end_date=None):
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)
        if data.empty:
            return None
        data = flatten_columns(data).reset_index()
        data['name'] = ticker
        data['sector'] = sector
        data.columns = [col.lower() for col in data.columns]
        required = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required):
            return None
        data = data[required + ['name', 'sector']]
        data['date'] = pd.to_datetime(data['date']).dt.tz_localize(None)
        return data
    except Exception:
        return None


def main():
    print("Pobieranie listy spółek S&P 500...")
    tickers, ticker_to_sector = get_sp500_tickers_and_sectors()
    print(f"Znaleziono {len(tickers)} tickerów.")

    print("Pobieranie danych SPY...")
    spy_data = download_spy_data()
    if spy_data.empty:
        print("⚠️ Uwaga: brak danych SPY. Kolumna spy_close będzie pusta.")

    all_data = []
    start_date = "2000-01-01"
    end_date = None
    failed = []

    for ticker in tqdm(tickers, desc="Pobieranie danych"):
        original_ticker = ticker.replace('-', '.')
        sector = ticker_to_sector.get(original_ticker, 'Unknown')
        df = download_stock_data(ticker, sector, start_date, end_date)
        if df is not None:
            all_data.append(df)
        else:
            failed.append(ticker)
        time.sleep(0.1)

    if not all_data:
        print("❌ Nie udało się pobrać danych dla żadnej spółki.")
        return

    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.sort_values(['date', 'name']).reset_index(drop=True)

    if not spy_data.empty:
        final_df = final_df.merge(spy_data, on='date', how='left')
    else:
        final_df['spy_close'] = np.nan

    output_file = "sp500_all_stocks_with_spy.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\n✅ Zapisano {len(final_df)} wierszy do pliku {output_file}")
    print(f"Pominięto {len(failed)} tickerów.")


if __name__ == "__main__":
    main()
