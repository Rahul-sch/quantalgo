#!/usr/bin/env python3
"""
OANDA v20 Fetcher — Phase 2 Data Pipeline
Replaces yfinance for all forex data needs.

SCAFFOLD — implement per DATA_PIPELINE_SPEC.md
DO NOT import from phase1 production files.
"""

import os
import time
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)
EST = ZoneInfo("US/Eastern")

# ─── Config ──────────────────────────────────────────────────────────────────

OANDA_PRACTICE_URL = "https://api-fxpractice.oanda.com"
OANDA_LIVE_URL     = "https://api-fxtrade.oanda.com"
STREAM_PRACTICE    = "https://stream-fxpractice.oanda.com"
STREAM_LIVE        = "https://stream-fxtrade.oanda.com"

GRANULARITY_MAP = {
    "1m":  "M1",  "5m":  "M5",  "15m": "M15", "30m": "M30",
    "1h":  "H1",  "4h":  "H4",  "1d":  "D",   "1w":  "W",
}

MAX_CANDLES_PER_REQUEST = 5000
DEFAULT_TIMEOUT_SEC     = 10
RETRY_ATTEMPTS          = 3


# ─── Circuit Breaker State ────────────────────────────────────────────────────

class CBState(Enum):
    CLOSED    = "closed"     # normal, route to OANDA
    OPEN      = "open"       # failing, skip to fallback
    HALF_OPEN = "half_open"  # testing recovery


@dataclass
class CircuitBreaker:
    state:           CBState = CBState.CLOSED
    failure_count:   int     = 0
    last_failure_ts: float   = 0.0
    open_duration:   int     = 300   # 5 min before half-open retry
    failure_thresh:  int     = 3

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_ts = time.time()
        if self.failure_count >= self.failure_thresh:
            self.state = CBState.OPEN
            logger.warning(f"[CB] Circuit OPEN after {self.failure_count} failures")

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = CBState.CLOSED

    def is_available(self) -> bool:
        if self.state == CBState.CLOSED:
            return True
        if self.state == CBState.OPEN:
            if time.time() - self.last_failure_ts > self.open_duration:
                self.state = CBState.HALF_OPEN
                logger.info("[CB] Circuit HALF-OPEN — testing recovery")
                return True
            return False
        return True  # HALF_OPEN: allow one attempt


# ─── Cache Manager ────────────────────────────────────────────────────────────

CACHE_DIR = os.path.expanduser("~/.quantalgo/cache")
CACHE_TTL = {
    "M1":  5 * 60,       # 5 min
    "M5":  5 * 60,
    "M15": 15 * 60,      # 15 min
    "M30": 15 * 60,
    "H1":  60 * 60,      # 1 hour
    "D":   24 * 3600,    # 1 day
}


def _cache_path(instrument: str, granularity: str, date_str: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"oanda_{instrument}_{granularity}_{date_str}.parquet")


def _load_cache(instrument: str, granularity: str) -> Optional[pd.DataFrame]:
    """Load cached data if within TTL."""
    date_str = datetime.now(EST).strftime("%Y-%m-%d")
    path = _cache_path(instrument, granularity, date_str)
    if not os.path.exists(path):
        return None
    ttl = CACHE_TTL.get(granularity, 900)
    if time.time() - os.path.getmtime(path) > ttl:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _save_cache(df: pd.DataFrame, instrument: str, granularity: str) -> None:
    """Save DataFrame to parquet cache."""
    date_str = datetime.now(EST).strftime("%Y-%m-%d")
    path = _cache_path(instrument, granularity, date_str)
    tmp = path + ".tmp"
    df.to_parquet(tmp)
    os.replace(tmp, path)


# ─── OANDA Fetcher ───────────────────────────────────────────────────────────

class OandaFetcher:
    """
    Async-capable OANDA v20 data fetcher with circuit breaker + cache.

    Usage:
        fetcher = OandaFetcher()
        df = fetcher.get_candles("EUR_USD", "M15", count=200)
        price = fetcher.get_live_price("EUR_USD")
    """

    def __init__(self):
        self.api_token  = os.getenv("OANDA_API_TOKEN", "")
        self.account_id = os.getenv("OANDA_ACCOUNT_ID", "")
        self.env        = os.getenv("OANDA_ENV", "practice")
        self.base_url   = OANDA_PRACTICE_URL if self.env == "practice" else OANDA_LIVE_URL
        self.stream_url = STREAM_PRACTICE if self.env == "practice" else STREAM_LIVE
        self.cb         = CircuitBreaker()
        self._session   = None  # aiohttp session (lazy init)

        if not self.api_token:
            logger.warning("[OANDA] No API token — will fall back to yfinance")

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    # ── Core REST: candles ────────────────────────────────────────────────────

    def get_candles(
        self,
        instrument: str,
        granularity: str = "M15",
        count: int = 500,
        from_dt: Optional[datetime] = None,
        to_dt: Optional[datetime] = None,
        price: str = "BA",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLC candles. Returns DataFrame with bid/ask/mid columns.
        Falls back through waterfall on failure.

        Args:
            instrument:  OANDA format e.g. "EUR_USD"
            granularity: "M15", "H1", "D", etc.
            count:       number of bars (max 5000)
            from_dt:     start time (UTC). If set, count is ignored.
            to_dt:       end time (UTC). Defaults to now.
            price:       "B"=bid, "A"=ask, "BA"=both, "M"=mid
            use_cache:   check local parquet cache first
        """
        # Check cache
        if use_cache and from_dt is None:
            cached = _load_cache(instrument, granularity)
            if cached is not None:
                logger.debug(f"[cache] {instrument} {granularity} from cache")
                return cached

        # Try OANDA
        if self.api_token and self.cb.is_available():
            try:
                df = self._fetch_oanda_candles(instrument, granularity, count,
                                                from_dt, to_dt, price)
                self.cb.record_success()
                if use_cache:
                    _save_cache(df, instrument, granularity)
                return df
            except Exception as e:
                self.cb.record_failure()
                logger.warning(f"[OANDA] fetch failed: {e} — falling back")

        # Fallback: Massive/Polygon
        try:
            df = self._fetch_polygon_candles(instrument, granularity, count, from_dt, to_dt)
            if df is not None and not df.empty:
                logger.info(f"[fallback] {instrument} from Polygon")
                return df
        except Exception as e:
            logger.warning(f"[Polygon] fetch failed: {e} — falling back to yfinance")

        # Last resort: yfinance (legacy)
        logger.warning(f"[fallback] {instrument} from yfinance — STALE DATA RISK")
        return self._fetch_yfinance_fallback(instrument, granularity, count)

    def get_live_price(self, instrument: str) -> Optional[Dict[str, float]]:
        """
        Get current bid/ask/mid snapshot.
        Returns: {bid, ask, mid, spread_pips, timestamp}
        """
        # TODO: implement via GET /v3/accounts/{id}/pricing
        raise NotImplementedError("Implement in Week 1 build")

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def stream_prices(
        self,
        instruments: List[str],
        callback: Callable,
    ) -> None:
        """
        SSE streaming price feed via OANDA.
        Calls callback(instrument, bid, ask, timestamp) on each tick.

        Usage:
            async def on_tick(instrument, bid, ask, ts):
                print(f"{instrument}: {bid}/{ask}")
            asyncio.run(fetcher.stream_prices(["EUR_USD"], on_tick))
        """
        # TODO: implement via GET /v3/accounts/{id}/pricing/stream
        # Use aiohttp ClientSession with chunked response reading
        raise NotImplementedError("Implement in Week 2 build")

    # ── Private: OANDA REST ───────────────────────────────────────────────────

    def _fetch_oanda_candles(
        self,
        instrument: str,
        granularity: str,
        count: int,
        from_dt: Optional[datetime],
        to_dt: Optional[datetime],
        price: str,
    ) -> pd.DataFrame:
        """
        Raw OANDA candle fetch. Handles pagination for count > 5000.
        TODO: implement with httpx or requests + pagination loop.
        """
        import requests

        params: Dict[str, Any] = {
            "granularity": granularity,
            "price": price,
        }
        if from_dt and to_dt:
            params["from"] = from_dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
            params["to"]   = to_dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        else:
            params["count"] = min(count, MAX_CANDLES_PER_REQUEST)

        url = f"{self.base_url}/v3/instruments/{instrument}/candles"
        resp = requests.get(url, headers=self._headers, params=params,
                            timeout=DEFAULT_TIMEOUT_SEC)
        resp.raise_for_status()
        data = resp.json()

        return self._parse_candles(data["candles"], instrument)

    def _parse_candles(self, candles: List[Dict], instrument: str) -> pd.DataFrame:
        """Parse OANDA candle JSON into DataFrame with bid/ask/mid columns."""
        rows = []
        for c in candles:
            if not c.get("complete", True):
                continue  # skip incomplete (current) bar
            row = {"time": pd.Timestamp(c["time"], tz="UTC").tz_convert("US/Eastern")}
            for side in ("bid", "ask", "mid"):
                if side in c:
                    for field_name in ("o", "h", "l", "c"):
                        col = f"{side}_{field_name}"
                        row[col] = float(c[side][field_name])
            row["volume"] = c.get("volume", 0)
            rows.append(row)

        df = pd.DataFrame(rows).set_index("time")

        # Compute mid from bid/ask if not provided
        if "mid_c" not in df.columns and "bid_c" in df.columns and "ask_c" in df.columns:
            for f in ("o", "h", "l", "c"):
                df[f"mid_{f}"] = (df[f"bid_{f}"] + df[f"ask_{f}"]) / 2

        # Backward-compat aliases (so existing code using High/Low/Close still works)
        if "mid_h" in df.columns:
            df["High"]  = df["mid_h"]
            df["Low"]   = df["mid_l"]
            df["Open"]  = df["mid_o"]
            df["Close"] = df["mid_c"]

        return df

    # ── Private: Fallbacks ────────────────────────────────────────────────────

    def _fetch_polygon_candles(
        self,
        instrument: str,
        granularity: str,
        count: int,
        from_dt: Optional[datetime],
        to_dt: Optional[datetime],
    ) -> Optional[pd.DataFrame]:
        """Fallback: Massive/Polygon forex aggregates."""
        # TODO: implement
        # Ticker format: C:EURUSD (not EUR_USD)
        return None

    def _fetch_yfinance_fallback(
        self, instrument: str, granularity: str, count: int
    ) -> pd.DataFrame:
        """Last-resort yfinance fetch. Logs warning."""
        import yfinance as yf
        # Convert OANDA instrument format to yfinance: EUR_USD → EURUSD=X
        yf_ticker = instrument.replace("_", "") + "=X"
        period_map = {"M15": "5d", "H1": "1mo", "D": "2y"}
        period = period_map.get(granularity, "5d")
        interval_map = {"M15": "15m", "H1": "1h", "D": "1d"}
        interval = interval_map.get(granularity, "15m")
        df = yf.download(yf_ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        return df
