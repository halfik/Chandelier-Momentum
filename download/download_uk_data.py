import pandas as pd
import yfinance as yf
from tqdm import tqdm
import time
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# FTSE 100 + FTSE 250 — tickery i sektory
# ══════════════════════════════════════════════════════════════════════════════

FTSE100_TICKERS = {
    # Consumer Discretionary
    "AUTO.L": "Consumer Discretionary", "MKS.L": "Consumer Discretionary",
    "NXT.L": "Consumer Discretionary", "JD.L": "Consumer Discretionary",
    "WPP.L": "Consumer Discretionary", "IHG.L": "Consumer Discretionary",
    "CPG.L": "Consumer Discretionary", "DCC.L": "Consumer Discretionary",
    "WTB.L": "Consumer Discretionary", "OCDO.L": "Consumer Discretionary",
    "TUI.L": "Consumer Discretionary",
    # Consumer Staples
    "ULVR.L": "Consumer Staples", "BATS.L": "Consumer Staples",
    "DGE.L": "Consumer Staples", "ABF.L": "Consumer Staples",
    "TSCO.L": "Consumer Staples", "SBRY.L": "Consumer Staples",
    "IMB.L": "Consumer Staples", "KGF.L": "Consumer Staples",
    "CCH.L": "Consumer Staples",
    # Health Care
    "AZN.L": "Health Care", "GSK.L": "Health Care", "HLN.L": "Health Care",
    "SN.L": "Health Care", "STJ.L": "Health Care", "HIK.L": "Health Care",
    "COB.L": "Health Care",
    # Industrials
    "BA.L": "Industrials", "RR.L": "Industrials", "FERG.L": "Industrials",
    "SMT.L": "Industrials", "EXPN.L": "Industrials", "DHL.L": "Industrials",
    "DPLM.L": "Industrials", "BNZL.L": "Industrials", "BME.L": "Industrials",
    "IAG.L": "Industrials", "EZJ.L": "Industrials", "INF.L": "Industrials",
    "SJP.L": "Industrials", "RS1.L": "Industrials",
    # Information Technology
    "LSEG.L": "Information Technology", "SGE.L": "Information Technology",
    "REL.L": "Information Technology", "PSON.L": "Information Technology",
    "SDR.L": "Information Technology", "DXCM.L": "Information Technology",
    # Communication Services
    "VOD.L": "Communication Services", "BT-A.L": "Communication Services",
    "ITV.L": "Communication Services", "WPP.L": "Communication Services",
    # Materials
    "RIO.L": "Materials", "GLEN.L": "Materials", "ANTO.L": "Materials",
    "FRES.L": "Materials", "MNDI.L": "Materials", "CRDA.L": "Materials",
    "SKG.L": "Materials", "SMDS.L": "Materials", "MRO.L": "Materials",
    "ADM.L": "Materials",
    # Real Estate
    "LAND.L": "Real Estate", "BLND.L": "Real Estate", "SGRO.L": "Real Estate",
    "LMP.L": "Real Estate", "PSN.L": "Real Estate", "BKG.L": "Real Estate",
    # Utilities
    "NG.L": "Utilities", "SSE.L": "Utilities", "SVT.L": "Utilities",
    "UU.L": "Utilities",
    # Financials (zostanie wykluczone przez backtest, ale pobieramy dla kompletności)
    "HSBA.L": "Financials", "BARC.L": "Financials", "LLOY.L": "Financials",
    "NWG.L": "Financials", "STAN.L": "Financials", "LGEN.L": "Financials",
    "AV.L": "Financials", "PRU.L": "Financials", "MNG.L": "Financials",
    "PHNX.L": "Financials", "PSH.L": "Financials", "III.L": "Financials",
    # Energy (zostanie wykluczone, ale pobieramy)
    "SHEL.L": "Energy", "BP.L": "Energy",
}

# Wybrane spółki FTSE 250 (poza Finance/Energy) z potencjałem trendowym
FTSE250_EXTRA = {
    "RMV.L": "Communication Services",   # Rightmove
    "HLMA.L": "Industrials",              # Halma
    "SMIN.L": "Industrials",              # Smiths Group
    "RKT.L": "Consumer Staples",          # Reckitt
    "AHT.L": "Information Technology",    # Ashtead Technology
    "FLTR.L": "Consumer Discretionary",   # Flutter Entertainment
    "SPX.L": "Industrials",               # SPX Technologies
    "CNA.L": "Industrials",               # Centrica
    "TW.L": "Real Estate",                # Taylor Wimpey
    "BDEV.L": "Real Estate",              # Barratt Developments
    "RTO.L": "Industrials",               # Rentokil
    "IMI.L": "Industrials",               # IMI
    "WEIR.L": "Industrials",              # Weir Group
    "MNZS.L": "Health Care",              # Mencap
    "HAS.L": "Health Care",               # Hays
    "BBY.L": "Industrials",               # Balfour Beatty
    "CLLN.L": "Industrials",              # Carillion (uwaga: delisted 2018)
    "MONY.L": "Communication Services",   # Moneysupermarket
    "PETS.L": "Consumer Discretionary",   # Pets at Home
    "DARK.L": "Information Technology",   # Darktrace
    "WISE.L": "Information Technology",   # Wise
    "OXB.L": "Health Care",               # Oxford Biomedica
    "QQ.L": "Information Technology",     # QinetiQ
    "NETW.L": "Information Technology",   # Network International
    "CTEC.L": "Health Care",              # ConvaTec
    "DOCS.L": "Health Care",              # Docmorris
    "HYVE.L": "Industrials",              # Hyve Group
    "TRN.L": "Industrials",               # Trainline
    "MNKS.L": "Information Technology",   # Monks Investment Trust
    "PCT.L": "Information Technology",    # Polar Capital Technology
}

ALL_TICKERS = {**FTSE100_TICKERS, **FTSE250_EXTRA}


def download_index_data(ticker="^FTSE", start_date="2000-01-01", end_date=None, retries=3):
    """Pobiera dane indeksu FTSE 100."""
    for attempt in range(retries):
        try:
            idx = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if idx is None or idx.empty:
                continue
            idx = idx.reset_index()
            # Obsługa MultiIndex kolumn (yfinance >= 0.2.x)
            if isinstance(idx.columns, pd.MultiIndex):
                idx.columns = [col[0].lower() for col in idx.columns]
            else:
                idx.columns = [col.lower() for col in idx.columns]
            idx = idx[['date', 'close']].rename(columns={'close': 'index_close'})
            idx['date'] = pd.to_datetime(idx['date'])
            idx = idx.reset_index(drop=True)
            print(f"✅ Pobrano dane indeksu {ticker}: {len(idx)} wierszy")
            return idx
        except Exception as e:
            print(f"  Próba {attempt+1}: {e}")
            time.sleep(2)
    print(f"❌ Nie udało się pobrać danych indeksu {ticker}.")
    return pd.DataFrame(columns=['date', 'index_close'])


def download_stock_data(ticker, sector, start_date="2000-01-01", end_date=None, retries=3):
    """Pobiera dane OHLCV dla pojedynczej spółki."""
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, auto_adjust=True)
            if data is None or data.empty:
                return None
            data = data.reset_index()
            # Obsługa MultiIndex
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
            # Usuń timezone z dat
            if hasattr(data['date'].dt, 'tz') and data['date'].dt.tz is not None:
                data['date'] = data['date'].dt.tz_localize(None)
            # Usuń wiersze z brakującymi danymi
            data = data.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            data = data[data['volume'] > 0]
            if len(data) < 50:
                return None
            return data
        except Exception:
            time.sleep(1)
    return None


def main():
    START_DATE = "2000-01-01"
    END_DATE = None  # None = do dziś
    OUTPUT_STOCKS = "data/uk/stocks_ftse.csv"
    OUTPUT_INDEX  = "data/uk/index.csv"

    import os
    os.makedirs("data/uk", exist_ok=True)

    # --- Pobierz indeks FTSE 100 ---
    print("Pobieranie danych indeksu FTSE 100 (^FTSE)...")
    index_data = download_index_data("^FTSE", start_date=START_DATE, end_date=END_DATE)

    if not index_data.empty:
        # Zapisz w formacie zgodnym z backtestem (kolumna 'close')
        index_out = index_data.rename(columns={'index_close': 'close'})
        index_out.to_csv(OUTPUT_INDEX, index=False)
        print(f"✅ Indeks zapisany: {OUTPUT_INDEX}")
    else:
        print("⚠️ Brak danych indeksu — backtest może nie działać poprawnie.")

    # --- Pobierz dane spółek ---
    print(f"\nPobieranie danych dla {len(ALL_TICKERS)} spółek FTSE...")
    all_data = []
    failed = []

    for ticker, sector in tqdm(ALL_TICKERS.items(), desc="Pobieranie"):
        df = download_stock_data(ticker, sector, start_date=START_DATE, end_date=END_DATE)
        if df is not None:
            all_data.append(df)
        else:
            failed.append(ticker)
        time.sleep(0.15)  # rate limiting

    if not all_data:
        print("❌ Nie udało się pobrać danych dla żadnej spółki.")
        return

    # --- Złącz i zapisz ---
    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.sort_values(['date', 'ticker']).reset_index(drop=True)

    final_df.to_csv(OUTPUT_STOCKS, index=False)

    # --- Podsumowanie ---
    print(f"\n{'='*60}")
    print(f"✅ Zapisano: {OUTPUT_STOCKS}")
    print(f"   Wierszy:  {len(final_df):,}")
    print(f"   Spółek:   {final_df['ticker'].nunique()}")
    print(f"   Zakres:   {final_df['date'].min().date()} → {final_df['date'].max().date()}")
    print(f"\nRozkład sektorów:")
    sector_counts = final_df.groupby('sector')['ticker'].nunique().sort_values(ascending=False)
    for sec, cnt in sector_counts.items():
        print(f"  {sec:<35}: {cnt:>3} spółek")
    if failed:
        print(f"\n⚠️ Nie pobrano ({len(failed)} tickerów): {', '.join(failed)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
