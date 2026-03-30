"""
Data Fetcher - Downloads historical OHLCV data via yfinance.
"""
import os
import yfinance as yf
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

INSTRUMENTS = {
    # Forex pairs
    "GBPJPY": "GBPJPY=X",
    "GBPUSD": "GBPUSD=X",
    "EURUSD": "EURUSD=X",
    "EURJPY": "EURJPY=X",
    "USDJPY": "USDJPY=X",
    # Stocks
    "NVDA": "NVDA",
    "TSLA": "TSLA",
    "SPY": "SPY",
}


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from yfinance (e.g. ('Close','QQQ') → 'Close')."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def download_data(symbol_key: str, period: str = "2y", interval: str = "1h") -> pd.DataFrame:
    """Download OHLCV data. Caches to CSV with staleness check.
    
    Cache rules:
    - period='5d' with interval='15m': cache max 15 min (live trading)
    - period='2y': cache indefinitely (backtest data)
    """
    ensure_data_dir()
    cache_file = os.path.join(DATA_DIR, f"{symbol_key}_{interval}.csv")

    # Check cache with staleness
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df = _flatten_columns(df)
        
        # For short-period live data, check cache age
        if period in ("5d", "1d", "1wk") and len(df) > 0:
            import time
            cache_age_min = (time.time() - os.path.getmtime(cache_file)) / 60
            if cache_age_min < 15:  # fresh enough for live trading
                print(f"  [cached] {symbol_key} {interval}: {len(df)} bars ({cache_age_min:.0f}m old)")
                return df
            else:
                print(f"  [stale] {symbol_key} {interval} cache is {cache_age_min:.0f}m old — refreshing...")
        elif len(df) > 100:
            print(f"  [cached] {symbol_key} {interval}: {len(df)} bars")
            return df

    ticker_symbol = INSTRUMENTS.get(symbol_key, symbol_key)
    print(f"  [downloading] {symbol_key} ({ticker_symbol}) {interval} {period}...")

    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            print(f"  [warning] No data for {symbol_key}")
            return pd.DataFrame()
        df = _flatten_columns(df)
        df.to_csv(cache_file)
        print(f"  [ok] {symbol_key}: {len(df)} bars")
        return df
    except Exception as e:
        print(f"  [error] {symbol_key}: {e}")
        return pd.DataFrame()


def download_all(period: str = "2y", interval: str = "1h") -> dict:
    """Download data for all instruments."""
    print("Downloading market data...")
    data = {}
    for key in INSTRUMENTS:
        df = download_data(key, period=period, interval=interval)
        if not df.empty:
            data[key] = df
    print(f"Downloaded {len(data)} instruments.\n")
    return data


def is_forex(symbol: str) -> bool:
    return symbol in ("GBPJPY", "GBPUSD", "EURUSD", "EURJPY", "USDJPY")
