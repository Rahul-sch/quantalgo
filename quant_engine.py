#!/usr/bin/env python3
"""
INSTITUTIONAL-GRADE CONTINUATION ENGINE
Vectorized execution, multi-timeframe confluence, anti-chop filters,
walk-forward validation, and realistic friction modeling.

Usage:
    python3 quant_engine.py                    # Run backtest with defaults
    python3 quant_engine.py --optimize         # Walk-forward optimization
    python3 quant_engine.py --symbol QQQ       # Single symbol
    python3 quant_engine.py --no-filter        # Disable session filter
"""
import sys
import os
import json
import argparse
import itertools
import time
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Load .env early so API keys are available to all functions
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(__file__))
from data_fetcher import download_data, INSTRUMENTS

# Ensure QQQ is available
if "QQQ" not in INSTRUMENTS:
    INSTRUMENTS["QQQ"] = "QQQ"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 0. CENTRALIZED CONFIG — NO HARDCODED VALUES IN LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """All tunable parameters in one place."""
    # ── Tickers ──
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ"])
    is_futures: Dict[str, bool] = field(default_factory=lambda: {
        "SPY": False, "QQQ": False, "ES": True, "NQ": True,
    })

    # ── Capital & Risk ──
    initial_capital: float = 10000.0
    risk_pct: float = 0.01  # 1% risk per trade

    # ── Strategy Parameters ──
    atr_period: int = 14
    atr_multiplier_sl: float = 1.0  # Wider stop: 1.0x ATR (was 0.5x)
    rr_ratio: float = 3.0  # must overcome friction — need home runs
    displacement_threshold: float = 1.0
    swing_lookback: int = 10
    fvg_max_age: int = 20       # max bars old an FVG can be
    trend_lookback: int = 20

    # ── Session Filter (EST) — AM + PM ──
    # AM: 9:30-11:30 ET (historically 32.4% WR)
    # PM: 1:30-3:30 ET (historically 27.3% WR — enabled for live forward-testing)
    session_filter: bool = True
    am_start: int = 570         # 9:30 in minutes
    am_end: int = 690           # 11:30
    pm_start: int = 810         # 13:30 (1:30 PM) in minutes
    pm_end: int = 930           # 15:30 (3:30 PM) in minutes

    # ── Retest Entry Logic ──
    use_retest_entry: bool = True   # Wait for price to pull back into FVG (limit order)
    retest_max_bars: int = 5        # Max bars to wait for retest after FVG forms

    # ── Confluence: 1H EMA ──
    ema_period: int = 20
    use_htf_filter: bool = True

    # ── Anti-Chop ──
    adx_period: int = 14
    adx_threshold: float = 18.0  # relaxed — still filters chop
    use_adx_filter: bool = True
    rvol_multiplier: float = 1.2  # relaxed from 1.5 — still confirms volume
    rvol_period: int = 10
    use_rvol_filter: bool = True

    # ── Friction ──
    futures_slippage_pts: float = 0.5
    etf_slippage_pct: float = 0.00005  # 0.005% (realistic for liquid ETFs)
    commission_round_trip: float = 2.40

    # ── Walk-Forward ──
    wf_windows: int = 4
    wf_train_days: int = 10
    wf_test_days: int = 5

    # ── Optimizer Grid ──
    opt_atr_sl: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.25])
    opt_rr: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0])
    opt_disp: List[float] = field(default_factory=lambda: [1.0, 1.3])

    # ── Regime Filters ──
    use_vix_filter: bool = True
    vix_threshold: float = 25.0     # halt trading when VIX >= this level
    use_weekly_trend_filter: bool = True
    weekly_ema_period: int = 10     # weekly EMA period for trend confirmation

    # ── Top-Down Macro Veto ──
    # If price < 20-Month SMA: ALL long signals disabled (macro downtrend)
    # If price > 20-Month SMA: ALL short signals disabled (macro uptrend)
    use_macro_veto: bool = True
    macro_sma_period: int = 20      # 20-Month SMA
    macro_veto_longs_below_sma: bool = True   # block longs in macro downtrend
    macro_veto_shorts_above_sma: bool = True  # block shorts in macro uptrend

    # ── Economic Calendar Blackout ──
    use_blackout_filter: bool = True
    blackout_minutes: int = 30  # minutes before/after high-impact events


CFG = Config()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FRICTION MODULE — Realistic Execution Costs
# ═══════════════════════════════════════════════════════════════════════════════

def apply_friction(
    entry: float, exit_price: float, direction: str, symbol: str, cfg: Config
) -> Tuple[float, float, float]:
    """
    Apply slippage + commission. Returns (adj_entry, adj_exit, commission).
    """
    is_fut = cfg.is_futures.get(symbol, False)

    if is_fut:
        slip = cfg.futures_slippage_pts
        if direction == "buy":
            adj_entry = entry + slip
            adj_exit = exit_price - slip
        else:
            adj_entry = entry - slip
            adj_exit = exit_price + slip
    else:
        slip_pct = cfg.etf_slippage_pct
        if direction == "buy":
            adj_entry = entry * (1 + slip_pct)
            adj_exit = exit_price * (1 - slip_pct)
        else:
            adj_entry = entry * (1 - slip_pct)
            adj_exit = exit_price * (1 + slip_pct)

    return adj_entry, adj_exit, cfg.commission_round_trip


# ═══════════════════════════════════════════════════════════════════════════════
# 1b. REGIME FILTER MODULE — VIX + Weekly Trend
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_vix_yfinance(start_date: str, end_date: str) -> Optional[pd.Series]:
    """
    Fetch VIX daily closes from yfinance (^VIX index).
    Returns a Series indexed by UTC DatetimeIndex, or None on failure.
    Uses UTC throughout to avoid tz-aware datetime conversion errors.
    """
    try:
        import yfinance as yf
        vix = yf.download(
            "^VIX",
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if vix.empty:
            return None
        close = vix["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        # Normalize index to UTC DatetimeIndex (avoids tz-aware conversion errors)
        close.index = pd.DatetimeIndex(close.index)
        if close.index.tz is None:
            close.index = close.index.tz_localize("UTC")
        else:
            close.index = close.index.tz_convert("UTC")
        return close
    except Exception as e:
        print(f"    [regime] yfinance VIX fetch error: {e}")
        return None


def compute_vix_regime(df_15m: pd.DataFrame, vix_threshold: float = 25.0) -> pd.Series:
    """
    Fetch VIX daily data via Massive API and map to 15m bars.
    Returns a boolean Series: True = safe to trade (VIX below threshold).
    Uses .shift(1) on daily VIX to prevent lookahead bias.

    FAIL-SAFE: If API unavailable, defaults to False (BLOCK ALL TRADES).
    Never fly blind — if we can't verify VIX is safe, we don't trade.
    """
    fail_safe = pd.Series(False, index=df_15m.index)

    try:
        # utc=True is required when index contains tz-aware datetime objects
        # (Python 3.9 / pandas 1.x raises ValueError without it)
        idx = pd.DatetimeIndex(pd.to_datetime(df_15m.index, utc=True))
        idx_est = idx.tz_convert("US/Eastern")

        start_date = str(idx_est[0].date())
        end_date = str((idx_est[-1] + pd.Timedelta(days=2)).date())

        vix_close = _fetch_vix_yfinance(start_date, end_date)

        if vix_close is None or vix_close.empty:
            print(f"    [regime] ⚠️  VIX data UNAVAILABLE — defaulting to BLOCK ALL TRADES (fail-safe)")
            return fail_safe

        # shift(1): we only know yesterday's closing VIX (no lookahead)
        vix_shifted = vix_close.shift(1)
        # vix_close is already UTC from _fetch_vix_yfinance

        # Convert df index to UTC, normalize to day boundary for ffill reindex
        df_idx_utc = pd.DatetimeIndex(pd.to_datetime(df_15m.index, utc=True))
        df_dates_utc = df_idx_utc.normalize()  # floor to midnight UTC, keeps tz
        vix_15m = vix_shifted.reindex(df_dates_utc, method="ffill")
        vix_15m.index = df_15m.index  # restore original bar index

        safe = vix_15m < vix_threshold
        # FAIL-SAFE: any bar with no VIX data → block (False), not allow
        safe = safe.fillna(False)
        safe.index = df_15m.index

        n_blocked = int((~safe).sum())
        n_total = len(safe)
        print(f"    [regime] VIX filter (threshold={vix_threshold}): "
              f"{n_blocked}/{n_total} bars blocked "
              f"({n_blocked/n_total*100:.1f}% of session)")

        return safe

    except Exception as e:
        print(f"    [regime] ⚠️  VIX filter CRITICAL ERROR: {e} — defaulting to BLOCK ALL TRADES")
        return fail_safe


def compute_weekly_trend(df_15m: pd.DataFrame, ema_period: int = 10) -> pd.Series:
    """
    Resample 15m to weekly, compute EMA slope for macro trend direction.
    Returns Series: 1 (bullish), -1 (bearish), 0 (flat/ambiguous).
    Uses .shift(1) on weekly bars to prevent lookahead bias.
    Only trade longs in bullish regime, shorts in bearish regime.
    """
    try:
        df_work = df_15m.copy()
        if not isinstance(df_work.index, pd.DatetimeIndex):
            df_work.index = pd.to_datetime(df_work.index, utc=True)
        if df_work.index.tz is None:
            try:
                df_work.index = df_work.index.tz_localize("US/Eastern")
            except Exception:
                df_work.index = df_work.index.tz_localize("UTC").tz_convert("US/Eastern")
        else:
            df_work.index = df_work.index.tz_convert("US/Eastern")

        # Resample to weekly
        df_weekly = df_work["Close"].resample("W").last().dropna()
        ema_weekly = df_weekly.ewm(span=ema_period, adjust=False).mean()
        slope = ema_weekly.diff().shift(1)  # shift to prevent lookahead

        trend = pd.Series(0, index=slope.index, dtype=int)
        trend[slope > 0] = 1
        trend[slope < 0] = -1

        # Map back to 15m
        trend_15m = trend.reindex(df_work.index, method="ffill").fillna(0).astype(int)
        return trend_15m

    except Exception as e:
        print(f"    [regime] Weekly trend error: {e} — skipping")
        return pd.Series(0, index=df_15m.index)


def _fetch_qqq_monthly(start_date: str, end_date: str) -> Optional[pd.Series]:
    """
    Fetch QQQ monthly closing prices.
    Primary: Massive/Polygon API (monthly aggregates).
    Fallback: yfinance monthly (if Polygon fails — free tier rate limits monthly data).
    Returns a Series indexed by US/Eastern datetime, or None on complete failure.
    FAIL-SAFE: returns None on ANY error.
    """
    # Try Massive/Polygon first
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("MASSIVE_API_KEY")
        if api_key:
            import urllib.request
            url = (f"https://api.polygon.io/v2/aggs/ticker/QQQ/range/1/month"
                   f"/{start_date}/{end_date}?adjusted=true&sort=asc&limit=500&apiKey={api_key}")
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            results = data.get("results", [])
            if results:
                dates = pd.to_datetime([r["t"] for r in results], unit="ms", utc=True)
                closes = [r["c"] for r in results]
                series = pd.Series(closes, index=dates.tz_convert("US/Eastern"))
                return series
    except Exception as e:
        print(f"    [macro] Massive monthly fetch error: {e} — trying yfinance fallback")

    # Fallback: yfinance monthly (acceptable for monthly macro data — changes rarely)
    try:
        import yfinance as yf
        monthly = yf.download(
            "QQQ",
            start=start_date,
            end=end_date,
            interval="1mo",
            progress=False,
            auto_adjust=True,
        )
        if monthly.empty:
            return None
        close = monthly["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if close.index.tz is None:
            close.index = pd.to_datetime(close.index).tz_localize("US/Eastern")
        else:
            close.index = close.index.tz_convert("US/Eastern")
        return close
    except Exception as e:
        print(f"    [macro] yfinance monthly fallback error: {e}")
        return None


def compute_macro_veto(df_15m: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    Top-Down Macro Veto using 20-Month SMA via Massive API.

    Logic (zero lookahead bias):
    - Fetch monthly QQQ closes via Massive/Polygon API
    - Compute 20-Month SMA on monthly close
    - Shift by 1 month (.shift(1)) — we only know LAST month's SMA
    - Map to 15m bars via forward-fill
    - Returns Series: 1=bullish(longs ok), -1=bearish(shorts ok), 0=neutral

    FAIL-SAFE: If API fails, returns 0 (neutral — no veto applied).
    Macro veto is a permissive filter; a brief API hiccup should not
    halt trading. Monthly macro regime changes rarely intraday.
    """
    try:
        # utc=True required for tz-aware datetime objects (Python 3.9 / pandas 1.x)
        idx = pd.DatetimeIndex(pd.to_datetime(df_15m.index, utc=True))
        idx_est = idx.tz_convert("US/Eastern")

        start = str((idx_est[0] - pd.DateOffset(months=cfg.macro_sma_period + 6)).date())
        end = str((idx_est[-1] + pd.DateOffset(months=1)).date())

        close = _fetch_qqq_monthly(start, end)

        if close is None or close.empty or len(close) < cfg.macro_sma_period:
            n = 0 if close is None else len(close)
            print(f"    [macro] ⚠️  Monthly data UNAVAILABLE ({n} bars) — macro veto DISABLED (neutral)")
            return pd.Series(0, index=df_15m.index)

        # 20-Month SMA, shift(1) = no lookahead
        sma_20m = close.rolling(cfg.macro_sma_period).mean().shift(1)

        macro_regime = pd.Series(0, index=sma_20m.index, dtype=int)
        macro_regime[close > sma_20m] = 1
        macro_regime[close <= sma_20m] = -1

        # Normalize monthly index to UTC for safe reindex
        if close.index.tz is None:
            m_idx_utc = close.index.tz_localize("UTC")
        else:
            m_idx_utc = close.index.tz_convert("UTC")
        macro_series = pd.Series(macro_regime.values, index=m_idx_utc)

        df_idx_utc = pd.DatetimeIndex(pd.to_datetime(df_15m.index, utc=True))
        df_dates_utc = df_idx_utc.normalize()  # floor to midnight UTC
        macro_15m = macro_series.reindex(df_dates_utc, method="ffill").fillna(0).astype(int)
        macro_15m.index = df_15m.index

        n_bull = int((macro_15m == 1).sum())
        n_bear = int((macro_15m == -1).sum())
        pct_bull = n_bull / len(macro_15m) * 100 if len(macro_15m) > 0 else 0
        print(f"    [macro] 20-Month SMA veto: "
              f"{n_bull} bars BULLISH ({pct_bull:.0f}%) | "
              f"{n_bear} bars BEARISH ({100-pct_bull:.0f}%)")

        return macro_15m

    except Exception as e:
        print(f"    [macro] ⚠️  Veto CRITICAL ERROR: {e} — returning neutral (no veto)")
        return pd.Series(0, index=df_15m.index)


# ═══════════════════════════════════════════════════════════════════════════════
# 1c. ECONOMIC CALENDAR BLACKOUT — High-Impact Event Filter
# ═══════════════════════════════════════════════════════════════════════════════

BLACKOUT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "blackout_windows.json")
BLACKOUT_CACHE_TTL = 86400  # 24 hours in seconds


def _get_hardcoded_blackout_events(lookback_days: int = 90) -> List[datetime]:
    """
    Fallback: generate known high-impact US economic event dates.
    Covers NFP (first Friday of month), FOMC (8 per year), CPI (~12th of month).
    Returns list of naive datetimes in US/Eastern.
    """
    today = date.today()
    start = today - timedelta(days=lookback_days)
    end = today + timedelta(days=lookback_days)

    events: List[datetime] = []

    # FOMC meeting dates 2023-2026 (approximate; 8 per year)
    fomc_dates = [
        # 2024
        date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1),
        date(2024, 6, 12), date(2024, 7, 31), date(2024, 9, 18),
        date(2024, 11, 7), date(2024, 12, 18),
        # 2025
        date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
        date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
        date(2025, 11, 5), date(2025, 12, 17),
        # 2026
        date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
        date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
        date(2026, 11, 4), date(2026, 12, 16),
    ]
    for d in fomc_dates:
        if start <= d <= end:
            events.append(datetime(d.year, d.month, d.day, 14, 0))  # 2:00 PM ET

    # NFP: first Friday of each month, 8:30 AM
    current = date(start.year, start.month, 1)
    while current <= end:
        first_friday = current
        while first_friday.weekday() != 4:  # 4 = Friday
            first_friday += timedelta(days=1)
        if start <= first_friday <= end:
            events.append(datetime(first_friday.year, first_friday.month, first_friday.day, 8, 30))
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    # CPI: ~12th of each month, 8:30 AM
    current = date(start.year, start.month, 1)
    while current <= end:
        cpi_day = date(current.year, current.month, 12)
        while cpi_day.weekday() >= 5:
            cpi_day += timedelta(days=1)
        if start <= cpi_day <= end:
            events.append(datetime(cpi_day.year, cpi_day.month, cpi_day.day, 8, 30))
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    return events


def get_blackout_windows(lookback_days: int = 90, blackout_minutes: int = 30) -> List[Tuple[datetime, datetime]]:
    """
    Fetch high-impact US economic events and return blackout windows.
    Each window = (start_dt, end_dt) = event_time +/- blackout_minutes.

    Sources:
    1. Finnhub API (if FINNHUB_API_KEY env var set)
    2. Hardcoded fallback (NFP, FOMC, CPI)
    3. Cached to data/blackout_windows.json with 24h TTL
    """
    os.makedirs(os.path.dirname(BLACKOUT_CACHE_PATH), exist_ok=True)

    # Check cache
    if os.path.exists(BLACKOUT_CACHE_PATH):
        try:
            with open(BLACKOUT_CACHE_PATH) as f:
                cached = json.load(f)
            age = time.time() - cached.get("timestamp", 0)
            if age < BLACKOUT_CACHE_TTL:
                windows = [(datetime.fromisoformat(w[0]), datetime.fromisoformat(w[1]))
                           for w in cached["windows"]]
                print(f"    [blackout] Using cached blackout windows ({len(windows)} events, age={age/3600:.1f}h)")
                return windows
        except Exception as e:
            print(f"    [blackout] Cache read error: {e} — regenerating")

    event_times: List[datetime] = []
    source = "hardcoded"

    # Try Finnhub API
    finnhub_key = os.environ.get("FINNHUB_API_KEY")
    if finnhub_key:
        try:
            import urllib.request
            today = date.today()
            from_date = (today - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            to_date = (today + timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            url = (f"https://finnhub.io/api/v1/calendar/economic"
                   f"?from={from_date}&to={to_date}&token={finnhub_key}")
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            eco_events = data.get("economicCalendar", [])
            for ev in eco_events:
                if ev.get("country") != "US":
                    continue
                impact = str(ev.get("impact", "")).lower()
                if impact not in ("high", "3"):
                    continue
                ev_time_str = ev.get("time") or ev.get("date")
                if ev_time_str:
                    try:
                        import pytz
                        ev_dt = datetime.fromisoformat(ev_time_str.replace("Z", "+00:00"))
                        et = pytz.timezone("US/Eastern")
                        ev_dt_et = ev_dt.astimezone(et).replace(tzinfo=None)
                        event_times.append(ev_dt_et)
                    except Exception:
                        pass
            if event_times:
                source = "finnhub"
                print(f"    [blackout] Fetched {len(event_times)} high-impact events from Finnhub")
        except Exception as e:
            print(f"    [blackout] Finnhub API error: {e} — falling back to hardcoded")
            event_times = []

    # Fallback to hardcoded
    if not event_times:
        event_times = _get_hardcoded_blackout_events(lookback_days)
        source = "hardcoded"
        print(f"    [blackout] Using {len(event_times)} hardcoded economic event dates")

    # Build windows: event +/- blackout_minutes
    delta = timedelta(minutes=blackout_minutes)
    windows = [(dt - delta, dt + delta) for dt in event_times]

    # Cache result
    try:
        cache_data = {
            "timestamp": time.time(),
            "source": source,
            "windows": [(w[0].isoformat(), w[1].isoformat()) for w in windows]
        }
        with open(BLACKOUT_CACHE_PATH, "w") as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        print(f"    [blackout] Cache write error: {e}")

    return windows


def apply_blackout_filter(signals_df: pd.DataFrame, blackout_windows: List[Tuple[datetime, datetime]]) -> pd.DataFrame:
    """
    Set signal=0 for any bar whose timestamp falls within a blackout window.
    Expects signals_df to have a DatetimeIndex.
    Returns filtered DataFrame.
    """
    if not blackout_windows:
        return signals_df

    df = signals_df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    # Convert to naive ET for comparison
    try:
        ts = df.index
        if ts.tz is not None:
            ts_naive = ts.tz_convert("US/Eastern").tz_localize(None)
        else:
            ts_naive = ts
    except Exception:
        ts_naive = df.index

    blackout_mask = pd.Series(False, index=df.index)
    for start_dt, end_dt in blackout_windows:
        s = start_dt.replace(tzinfo=None) if (hasattr(start_dt, 'tzinfo') and start_dt.tzinfo) else start_dt
        e = end_dt.replace(tzinfo=None) if (hasattr(end_dt, 'tzinfo') and end_dt.tzinfo) else end_dt
        blackout_mask |= (ts_naive >= s) & (ts_naive <= e)

    n_blocked = blackout_mask.sum()
    if n_blocked > 0 and "signal" in df.columns:
        df.loc[blackout_mask, "signal"] = 0
        print(f"    [blackout] Blocked {n_blocked} bars during economic events")

    return df


# Module-level cache for live paper trader
_LIVE_BLACKOUT_WINDOWS: List[Tuple[datetime, datetime]] = []


def refresh_blackout_calendar(cfg: Optional[Config] = None) -> List[Tuple[datetime, datetime]]:
    """
    Called once at paper_trader.py startup (or 08:00 AM refresh).
    Fetches THIS WEEK's high-impact economic calendar via Finnhub API.
    Falls back to algorithmic computation if API unavailable.

    Logs exactly what events were found today so the trader can verify.
    Updates the module-level _LIVE_BLACKOUT_WINDOWS cache.
    """
    global _LIVE_BLACKOUT_WINDOWS

    blackout_min = cfg.blackout_minutes if cfg else 30
    today = date.today()
    # Fetch today + 7 days for the week ahead
    windows = get_blackout_windows(lookback_days=0, blackout_minutes=blackout_min)
    # Override cache TTL for live use: always get fresh windows for today
    # Re-fetch with explicit range covering this week
    try:
        from dotenv import load_dotenv
        load_dotenv()
        finnhub_key = os.environ.get("FINNHUB_API_KEY")
        import urllib.request
        from_date = today.strftime("%Y-%m-%d")
        to_date = (today + timedelta(days=7)).strftime("%Y-%m-%d")
        if finnhub_key:
            url = (f"https://finnhub.io/api/v1/calendar/economic"
                   f"?from={from_date}&to={to_date}&token={finnhub_key}")
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            eco_events = data.get("economicCalendar", [])
            event_times = []
            for ev in eco_events:
                if ev.get("country") != "US":
                    continue
                impact = str(ev.get("impact", "")).lower()
                if impact not in ("high", "3"):
                    continue
                ev_time_str = ev.get("time") or ev.get("date")
                if ev_time_str:
                    try:
                        import pytz
                        ev_dt = datetime.fromisoformat(ev_time_str.replace("Z", "+00:00"))
                        et = pytz.timezone("US/Eastern")
                        ev_dt_et = ev_dt.astimezone(et).replace(tzinfo=None)
                        event_times.append(ev_dt_et)
                    except Exception:
                        pass
            if event_times:
                delta = timedelta(minutes=blackout_min)
                windows = [(dt - delta, dt + delta) for dt in event_times]
                print(f"📅 Fetched {len(event_times)} high-impact events from Finnhub (live)")
    except Exception as e:
        print(f"    [blackout] Live refresh error: {e} — using cached/computed windows")

    _LIVE_BLACKOUT_WINDOWS = windows

    # Log today's blackouts so trader can verify
    today_windows = [
        (s, e) for s, e in windows
        if s.date() == today or e.date() == today
    ]
    if today_windows:
        today_str = ", ".join(
            f"{s.strftime('%H:%M')}-{e.strftime('%H:%M')}"
            for s, e in sorted(today_windows)
        )
        print(f"📅 Blackout windows TODAY ({today}): [{today_str}] (±{blackout_min}min around events)")
    else:
        print(f"📅 No high-impact events scheduled today ({today}) — no blackouts")

    return windows


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONFLUENCE MODULE — 1H EMA (Lookahead-Bias Free)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_htf_ema_signal(df_5m: pd.DataFrame, ema_period: int = 20) -> pd.Series:
    """
    Resample 5m data to 1H, compute EMA, then map back to 5m bars.
    CRITICAL: Uses .shift(1) on 1H to prevent lookahead bias —
    5m bars only see the PREVIOUS CLOSED 1H candle's EMA.

    Returns: pd.Series aligned to df_5m index with values: 1 (bullish), -1 (bearish), 0 (neutral)
    """
    df_work = df_5m.copy()

    # Ensure DatetimeIndex — bulletproof conversion
    if not isinstance(df_work.index, pd.DatetimeIndex):
        df_work.index = pd.to_datetime(df_work.index, utc=True)
    if df_work.index.tz is None:
        try:
            df_work.index = df_work.index.tz_localize("US/Eastern")
        except Exception:
            df_work.index = df_work.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        df_work.index = df_work.index.tz_convert("US/Eastern")

    # Resample to 1H
    df_1h = df_work["Close"].resample("1h").last().dropna()
    ema_1h = df_1h.ewm(span=ema_period, adjust=False).mean()

    # EMA slope: positive = bullish, negative = bearish
    ema_slope = ema_1h.diff()

    # SHIFT by 1 to prevent lookahead: 5m bar reads PREVIOUS closed 1H candle
    ema_slope_shifted = ema_slope.shift(1)

    # Map signal to 5m bars using forward-fill (each 5m bar gets the last known 1H signal)
    signal_1h = pd.Series(0, index=ema_slope_shifted.index, dtype=int)
    signal_1h[ema_slope_shifted > 0] = 1
    signal_1h[ema_slope_shifted < 0] = -1

    # Reindex to 5m and forward-fill — ensure no NaN gaps
    signal_5m = signal_1h.reindex(df_work.index, method="ffill")
    signal_5m = signal_5m.ffill().fillna(0).astype(int)

    return signal_5m


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANTI-CHOP MODULE — ADX & RVOL Filters (Vectorized)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Vectorized ADX calculation."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()

    return adx.fillna(0)


def compute_rvol(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Relative Volume: current volume / SMA of volume."""
    vol_sma = df["Volume"].rolling(period).mean()
    return (df["Volume"] / vol_sma.replace(0, np.nan)).fillna(0)


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION — Vectorized Pre-computation + Targeted Loop
# ═══════════════════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Pre-compute all indicators in vectorized fashion. No loops."""
    df_ind = df.copy()

    # ATR
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df_ind["atr"] = tr.rolling(cfg.atr_period).mean()

    # Trend: SMA short vs long
    df_ind["sma_short"] = df["Close"].rolling(5).mean()
    df_ind["sma_long"] = df["Close"].rolling(cfg.trend_lookback).mean()
    df_ind["trend"] = 0
    df_ind.loc[df_ind["sma_short"] > df_ind["sma_long"], "trend"] = 1
    df_ind.loc[df_ind["sma_short"] < df_ind["sma_long"], "trend"] = -1

    # ADX
    if cfg.use_adx_filter:
        df_ind["adx"] = compute_adx(df, cfg.adx_period)
    else:
        df_ind["adx"] = 100.0  # always pass

    # RVOL
    if cfg.use_rvol_filter:
        df_ind["rvol"] = compute_rvol(df, cfg.rvol_period)
    else:
        df_ind["rvol"] = 10.0  # always pass

    # Candle range
    df_ind["candle_range"] = df["High"] - df["Low"]

    # Displacement flag
    df_ind["is_displacement"] = df_ind["candle_range"] > (df_ind["atr"] * cfg.displacement_threshold)

    # Session filter (EST) — bulletproof timezone handling
    if cfg.session_filter:
        try:
            idx = df.index
            # Step 1: Force to DatetimeIndex no matter what
            if not isinstance(idx, pd.DatetimeIndex):
                idx = pd.to_datetime(idx, utc=True)
            # Step 2: If no timezone, assume US/Eastern (yfinance US stocks)
            if idx.tz is None:
                try:
                    idx = idx.tz_localize("US/Eastern")
                except Exception:
                    idx = idx.tz_localize("UTC").tz_convert("US/Eastern")
            else:
                # Convert whatever timezone to Eastern
                idx = idx.tz_convert("US/Eastern")

            minutes = idx.hour * 60 + idx.minute
            in_am = (minutes >= cfg.am_start) & (minutes <= cfg.am_end)
            in_pm = (minutes >= cfg.pm_start) & (minutes <= cfg.pm_end)
            df_ind["in_session"] = (in_am | in_pm).values
        except Exception:
            # If all else fails, let everything through
            df_ind["in_session"] = True
    else:
        df_ind["in_session"] = True

    # HTF EMA signal
    if cfg.use_htf_filter:
        df_ind["htf_signal"] = compute_htf_ema_signal(df, cfg.ema_period)
    else:
        df_ind["htf_signal"] = 0

    # ── Regime filters ──
    # VIX filter: block trading when fear index is too high
    if cfg.use_vix_filter:
        df_ind["vix_safe"] = compute_vix_regime(df, cfg.vix_threshold)
    else:
        df_ind["vix_safe"] = True

    # Weekly trend filter: only trade in direction of macro trend
    if cfg.use_weekly_trend_filter:
        df_ind["weekly_trend"] = compute_weekly_trend(df, cfg.weekly_ema_period)
    else:
        df_ind["weekly_trend"] = 0  # 0 = no filter, allow both directions

    # Top-Down Macro Veto: 20-Month SMA
    if cfg.use_macro_veto:
        df_ind["macro_regime"] = compute_macro_veto(df, cfg)
    else:
        df_ind["macro_regime"] = 0  # 0 = no veto, allow both directions

    # Economic Calendar Blackout
    if cfg.use_blackout_filter:
        blackout_windows = get_blackout_windows(
            lookback_days=90, blackout_minutes=cfg.blackout_minutes
        )
        if blackout_windows:
            try:
                idx = df.index
                if not isinstance(idx, pd.DatetimeIndex):
                    idx = pd.to_datetime(idx, utc=True)
                if idx.tz is not None:
                    ts_naive = idx.tz_convert("US/Eastern").tz_localize(None)
                else:
                    ts_naive = idx

                blackout_mask = pd.Series(False, index=df.index)
                for start_dt, end_dt in blackout_windows:
                    s = start_dt.replace(tzinfo=None) if (hasattr(start_dt, 'tzinfo') and start_dt.tzinfo) else start_dt
                    e = end_dt.replace(tzinfo=None) if (hasattr(end_dt, 'tzinfo') and end_dt.tzinfo) else end_dt
                    blackout_mask |= (ts_naive >= s) & (ts_naive <= e)

                df_ind["in_blackout"] = blackout_mask.values
                n_blackout = blackout_mask.sum()
                if n_blackout > 0:
                    print(f"    [blackout] {n_blackout} bars flagged during economic events")
            except Exception as e:
                print(f"    [blackout] Error applying blackout filter: {e}")
                df_ind["in_blackout"] = False
        else:
            df_ind["in_blackout"] = False
    else:
        df_ind["in_blackout"] = False

    return df_ind


def detect_fvgs_vectorized(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Vectorized FVG detection.
    Returns: bull_fvg_top, bull_fvg_bot, bear_fvg_top, bear_fvg_bot (Series, NaN where no FVG)
    """
    h = df["High"].values
    l = df["Low"].values
    n = len(df)

    bull_top = np.full(n, np.nan)
    bull_bot = np.full(n, np.nan)
    bear_top = np.full(n, np.nan)
    bear_bot = np.full(n, np.nan)

    for i in range(2, n):
        # Bullish FVG: candle[i-2] high < candle[i] low
        if h[i-2] < l[i]:
            bull_top[i] = l[i]
            bull_bot[i] = h[i-2]
        # Bearish FVG: candle[i-2] low > candle[i] high
        if l[i-2] > h[i]:
            bear_top[i] = l[i-2]
            bear_bot[i] = h[i]

    return (pd.Series(bull_top, index=df.index), pd.Series(bull_bot, index=df.index),
            pd.Series(bear_top, index=df.index), pd.Series(bear_bot, index=df.index))


def find_swings_vectorized(df: pd.DataFrame, lookback: int = 10) -> Tuple[pd.Series, pd.Series]:
    """Vectorized swing high/low detection using rolling max/min."""
    window = 2 * lookback + 1
    roll_max = df["High"].rolling(window, center=True).max()
    roll_min = df["Low"].rolling(window, center=True).min()

    swing_highs = pd.Series(np.nan, index=df.index)
    swing_lows = pd.Series(np.nan, index=df.index)

    swing_highs[df["High"] == roll_max] = df["High"][df["High"] == roll_max]
    swing_lows[df["Low"] == roll_min] = df["Low"][df["Low"] == roll_min]

    return swing_highs, swing_lows


def find_irl_target(direction: str, bar_idx: int, entry_price: float,
                    tp_price: float, high_arr, low_arr, pivot_bars: int = 3) -> float:
    """
    Find the nearest Internal Range Liquidity (IRL) between entry and TP.
    Uses fast n-bar pivots (no lookahead — only pivots confirmed before bar_idx).

    For LONGS: nearest pivot high ABOVE entry but BELOW TP.
    For SHORTS: nearest pivot low BELOW entry but ABOVE TP.

    Returns the IRL price, or NaN if none found.
    """
    if direction == "buy":
        candidates = []
        # Scan backward from entry bar for recent pivot highs
        for j in range(bar_idx - 1, max(pivot_bars, bar_idx - 60), -1):
            if j < pivot_bars or j >= len(high_arr) - pivot_bars:
                continue
            is_pivot = True
            for k in range(1, pivot_bars + 1):
                if high_arr[j] < high_arr[j - k] or high_arr[j] < high_arr[j + k]:
                    is_pivot = False; break
            if is_pivot and high_arr[j] > entry_price and high_arr[j] < tp_price:
                candidates.append(high_arr[j])
        return min(candidates) if candidates else float('nan')
    else:  # sell
        candidates = []
        for j in range(bar_idx - 1, max(pivot_bars, bar_idx - 60), -1):
            if j < pivot_bars or j >= len(low_arr) - pivot_bars:
                continue
            is_pivot = True
            for k in range(1, pivot_bars + 1):
                if low_arr[j] > low_arr[j - k] or low_arr[j] > low_arr[j + k]:
                    is_pivot = False; break
            if is_pivot and low_arr[j] < entry_price and low_arr[j] > tp_price:
                candidates.append(low_arr[j])
        return max(candidates) if candidates else float('nan')


def generate_signals(df: pd.DataFrame, cfg: Config) -> List[Dict[str, Any]]:
    """
    Generate continuation signals using RETEST ENTRY logic.

    Strategy:
    1. Detect FVG formation on displacement candle
    2. ARM a limit order at the FVG edge (top for longs, bottom for shorts)
    3. Only ENTER if price pulls back INTO the FVG within retest_max_bars
    4. Entry = FVG edge (not market close) — much better fill price
    5. Stop = 1.0x ATR below/above entry (wider, survives noise)
    6. Target = entry +/- (risk * rr_ratio)

    Why retest entry is better:
    - Avoids chasing the initial displacement candle (worst entry)
    - Enters at known support/resistance level (FVG boundary)
    - Higher RR because entry is closer to the true support
    - Eliminates most false breakouts that immediately reverse
    """
    if len(df) < 50:
        return []

    ind = compute_indicators(df, cfg)
    bull_fvg_top, bull_fvg_bot, bear_fvg_top, bear_fvg_bot = detect_fvgs_vectorized(df)
    swing_highs, swing_lows = find_swings_vectorized(df, cfg.swing_lookback)

    signals = []
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    atr = ind["atr"].values
    trend = ind["trend"].values
    adx = ind["adx"].values
    rvol = ind["rvol"].values
    in_session = ind["in_session"].values
    htf = ind["htf_signal"].values
    vix_safe = ind["vix_safe"].values.astype(bool)
    weekly_trend = ind["weekly_trend"].values
    macro_regime = ind["macro_regime"].values  # +1=bullish, -1=bearish, 0=neutral

    _skip = {"atr": 0, "session": 0, "adx": 0, "regime": 0,
             "macro_veto": 0, "no_retest": 0, "invalidated": 0, "passed": 0}

    # armed_bull / armed_bear: fvg_bar -> (limit_price, fvg_other_edge, armed_bar,
    #                                       drawn_liq, atr_at_arm, adx_at_arm, rvol_at_arm)
    armed_bull: Dict[int, tuple] = {}
    armed_bear: Dict[int, tuple] = {}
    used_bars: set = set()

    for i in range(30, len(df)):
        if np.isnan(atr[i]) or atr[i] <= 0:
            _skip["atr"] += 1
            continue

        # ── ARM new limit orders when an FVG forms ──
        if not np.isnan(bull_fvg_top.iloc[i]):
            fvg_top = float(bull_fvg_top.iloc[i])
            fvg_bot = float(bull_fvg_bot.iloc[i])
            if trend[i] == 1:
                htf_ok = not cfg.use_htf_filter or htf[i] >= 0
                weekly_ok = not cfg.use_weekly_trend_filter or weekly_trend[i] >= 0
                adx_ok = adx[i] >= cfg.adx_threshold
                vix_ok = not cfg.use_vix_filter or vix_safe[i]
                # MACRO VETO: no longs if price below 20-Month SMA
                macro_ok = (not cfg.use_macro_veto or
                            not cfg.macro_veto_longs_below_sma or
                            macro_regime[i] >= 0)
                if macro_ok and htf_ok and weekly_ok and adx_ok and vix_ok:
                    drawn_liq = None
                    for j in range(i - 1, max(i - 50, 0), -1):
                        sh = swing_highs.iloc[j]
                        if not np.isnan(sh) and sh > fvg_top:
                            drawn_liq = sh
                            break
                    if drawn_liq is None:
                        recent_high = np.max(high[max(0, i - 20):i])
                        if recent_high > fvg_top * 1.001:
                            drawn_liq = recent_high
                    if drawn_liq is not None:
                        armed_bull[i] = (fvg_top, fvg_bot, i, drawn_liq,
                                         atr[i], adx[i], rvol[i])

        if not np.isnan(bear_fvg_top.iloc[i]):
            fvg_top = float(bear_fvg_top.iloc[i])
            fvg_bot = float(bear_fvg_bot.iloc[i])
            if trend[i] == -1:
                htf_ok = not cfg.use_htf_filter or htf[i] <= 0
                weekly_ok = not cfg.use_weekly_trend_filter or weekly_trend[i] <= 0
                adx_ok = adx[i] >= cfg.adx_threshold
                vix_ok = not cfg.use_vix_filter or vix_safe[i]
                # MACRO VETO: no shorts if price above 20-Month SMA
                macro_ok = (not cfg.use_macro_veto or
                            not cfg.macro_veto_shorts_above_sma or
                            macro_regime[i] <= 0)
                if macro_ok and htf_ok and weekly_ok and adx_ok and vix_ok:
                    drawn_liq = None
                    for j in range(i - 1, max(i - 50, 0), -1):
                        sl_val = swing_lows.iloc[j]
                        if not np.isnan(sl_val) and sl_val < fvg_bot:
                            drawn_liq = sl_val
                            break
                    if drawn_liq is None:
                        recent_low = np.min(low[max(0, i - 20):i])
                        if recent_low < fvg_bot * 0.999:
                            drawn_liq = recent_low
                    if drawn_liq is not None:
                        armed_bear[i] = (fvg_bot, fvg_top, i, drawn_liq,
                                         atr[i], adx[i], rvol[i])

        # ── Check if any armed limit order was filled this bar ──

        to_remove_bull = []
        for fvg_bar, (limit_price, fvg_bot, armed_bar, drawn_liq,
                      atr_arm, adx_arm, rvol_arm) in armed_bull.items():
            bars_elapsed = i - armed_bar
            if bars_elapsed > cfg.retest_max_bars:
                to_remove_bull.append(fvg_bar)
                _skip["no_retest"] += 1
                continue
            # Invalidated: price breaks below FVG bottom (structure destroyed)
            if low[i] < fvg_bot:
                to_remove_bull.append(fvg_bar)
                _skip["invalidated"] += 1
                continue
            if bars_elapsed == 0:
                continue
            # Retest: candle low touched limit price
            if not (low[i] <= limit_price <= high[i]):
                continue
            if not in_session[i]:
                continue
            if rvol[i] < cfg.rvol_multiplier:
                continue
            # Build trade
            entry = limit_price
            sl = entry - atr_arm * cfg.atr_multiplier_sl
            risk = entry - sl
            if risk <= 0:
                continue
            tp = entry + risk * cfg.rr_ratio
            if drawn_liq < tp:
                tp = drawn_liq
            reward = tp - entry
            if reward / risk < 1.0:
                continue
            if i not in used_bars:
                used_bars.add(i)
                _skip["passed"] += 1
                irl = find_irl_target("buy", i, entry, tp, high, low)
                signals.append({
                    "bar": i, "direction": "buy",
                    "entry": round(entry, 4),
                    "sl": round(sl, 4),
                    "tp": round(tp, 4),
                    "irl_target": round(irl, 4) if not np.isnan(irl) else None,
                    "strategy": "continuation_retest",
                    "reason": (f"Bull retest FVG={limit_price:.2f} "
                               f"SL=1.0xATR({atr_arm:.2f}) DL={drawn_liq:.2f} "
                               f"RR={reward/risk:.1f} ADX={adx_arm:.0f} RVOL={rvol[i]:.2f}"
                               f"{f' IRL={irl:.2f}' if not np.isnan(irl) else ''}"),
                })
            to_remove_bull.append(fvg_bar)

        for k in to_remove_bull:
            armed_bull.pop(k, None)

        to_remove_bear = []
        for fvg_bar, (limit_price, fvg_top, armed_bar, drawn_liq,
                      atr_arm, adx_arm, rvol_arm) in armed_bear.items():
            bars_elapsed = i - armed_bar
            if bars_elapsed > cfg.retest_max_bars:
                to_remove_bear.append(fvg_bar)
                _skip["no_retest"] += 1
                continue
            if high[i] > fvg_top:
                to_remove_bear.append(fvg_bar)
                _skip["invalidated"] += 1
                continue
            if bars_elapsed == 0:
                continue
            if not (low[i] <= limit_price <= high[i]):
                continue
            if not in_session[i]:
                continue
            if rvol[i] < cfg.rvol_multiplier:
                continue
            entry = limit_price
            sl = entry + atr_arm * cfg.atr_multiplier_sl
            risk = sl - entry
            if risk <= 0:
                continue
            tp = entry - risk * cfg.rr_ratio
            if drawn_liq > tp:
                tp = drawn_liq
            reward = entry - tp
            if reward / risk < 1.0:
                continue
            if i not in used_bars:
                used_bars.add(i)
                _skip["passed"] += 1
                irl = find_irl_target("sell", i, entry, tp, high, low)
                signals.append({
                    "bar": i, "direction": "sell",
                    "entry": round(entry, 4),
                    "sl": round(sl, 4),
                    "tp": round(tp, 4),
                    "irl_target": round(irl, 4) if not np.isnan(irl) else None,
                    "strategy": "continuation_retest",
                    "reason": (f"Bear retest FVG={limit_price:.2f} "
                               f"SL=1.0xATR({atr_arm:.2f}) DL={drawn_liq:.2f} "
                               f"RR={reward/risk:.1f} ADX={adx_arm:.0f} RVOL={rvol[i]:.2f}"
                               f"{f' IRL={irl:.2f}' if not np.isnan(irl) else ''}"),
                })
            to_remove_bear.append(fvg_bar)

        for k in to_remove_bear:
            armed_bear.pop(k, None)

    # Diagnostics
    total_bars = len(df) - 30
    macro_bull = int((macro_regime == 1).sum())
    macro_bear = int((macro_regime == -1).sum())
    print(f"    Filter diagnostics ({total_bars} bars scanned):")
    print(f"      Skipped — ATR: {_skip['atr']} | Session: {_skip['session']} | "
          f"ADX<{cfg.adx_threshold}: {_skip['adx']} | Regime(VIX): {_skip['regime']}")
    print(f"      Macro regime — Bullish: {macro_bull} bars | Bearish: {macro_bear} bars")
    print(f"      No retest (expired): {_skip['no_retest']} | "
          f"Invalidated (structure break): {_skip['invalidated']}")
    print(f"      Signals generated: {len(signals)}")
    return signals



# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION ENGINE — With Friction
# ═══════════════════════════════════════════════════════════════════════════════

def execute_backtest(
    df: pd.DataFrame, signals: List[Dict], symbol: str, cfg: Config
) -> Dict[str, Any]:
    """Run backtest with full friction modeling. Returns gross and net P&L."""
    if not signals:
        return _empty_result(symbol, cfg.initial_capital)

    trades = []
    capital = cfg.initial_capital
    equity_curve = [capital]
    gross_pnl_total = 0.0
    net_pnl_total = 0.0
    total_commission = 0.0
    total_slippage = 0.0

    for sig in signals:
        entry = sig["entry"]
        sl = sig["sl"]
        tp = sig["tp"]
        direction = sig["direction"]
        bar_idx = sig["bar"]
        risk_per_trade = capital * cfg.risk_pct

        sl_dist = abs(entry - sl)
        if sl_dist <= 0:
            continue

        lot_size = risk_per_trade / sl_dist

        # Simulate trade forward
        outcome = _simulate_trade(df, bar_idx, direction, entry, sl, tp)
        if outcome is None:
            continue

        raw_exit = outcome["exit_price"]

        # Apply friction
        adj_entry, adj_exit, commission = apply_friction(entry, raw_exit, direction, symbol, cfg)

        if direction == "buy":
            gross_pnl = (raw_exit - entry) * lot_size
            net_pnl = (adj_exit - adj_entry) * lot_size - commission
        else:
            gross_pnl = (entry - raw_exit) * lot_size
            net_pnl = (adj_entry - adj_exit) * lot_size - commission

        slip_cost = abs(gross_pnl - (net_pnl + commission))
        total_slippage += slip_cost
        total_commission += commission
        gross_pnl_total += gross_pnl
        net_pnl_total += net_pnl

        capital += net_pnl
        equity_curve.append(capital)

        trades.append({
            "direction": direction,
            "entry": round(entry, 4), "exit": round(raw_exit, 4),
            "adj_entry": round(adj_entry, 4), "adj_exit": round(adj_exit, 4),
            "gross_pnl": round(gross_pnl, 2), "net_pnl": round(net_pnl, 2),
            "commission": round(commission, 2),
            "win": net_pnl > 0, "outcome": outcome["type"],
            "bar": bar_idx,
        })

    return _compute_metrics(trades, equity_curve, symbol, cfg.initial_capital,
                           gross_pnl_total, net_pnl_total, total_commission, total_slippage)


def _simulate_trade(df, start_bar, direction, entry, sl, tp, max_bars=50):
    """Simulate trade forward from entry bar."""
    for i in range(start_bar + 1, min(start_bar + max_bars, len(df))):
        h = df["High"].iloc[i]
        l = df["Low"].iloc[i]
        if direction == "buy":
            if l <= sl:
                return {"exit_price": sl, "type": "sl_hit", "bars": i - start_bar}
            if h >= tp:
                return {"exit_price": tp, "type": "tp_hit", "bars": i - start_bar}
        else:
            if h >= sl:
                return {"exit_price": sl, "type": "sl_hit", "bars": i - start_bar}
            if l <= tp:
                return {"exit_price": tp, "type": "tp_hit", "bars": i - start_bar}
    exit_price = df["Close"].iloc[min(start_bar + max_bars - 1, len(df) - 1)]
    return {"exit_price": exit_price, "type": "timeout", "bars": max_bars}


def _compute_metrics(trades, equity_curve, symbol, initial_capital,
                     gross_pnl, net_pnl, total_commission, total_slippage):
    if not trades:
        return _empty_result(symbol, initial_capital)

    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    net_pnls = [t["net_pnl"] for t in trades]
    wr = len(wins) / len(trades) * 100
    avg_win = np.mean([t["net_pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t["net_pnl"]) for t in losses]) if losses else 0
    pf_denom = sum(abs(t["net_pnl"]) for t in losses)
    pf = sum(t["net_pnl"] for t in wins) / pf_denom if pf_denom > 0 else float("inf")

    peak = initial_capital
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    sharpe = 0
    if len(net_pnls) > 10 and np.std(net_pnls) > 0:
        sharpe = np.mean(net_pnls) / np.std(net_pnls) * np.sqrt(252)
        sharpe = max(min(sharpe, 100), -100)

    return {
        "symbol": symbol,
        "total_trades": len(trades),
        "wins": len(wins), "losses": len(losses),
        "win_rate": round(wr, 1),
        "gross_pnl": round(gross_pnl, 2),
        "net_pnl": round(net_pnl, 2),
        "total_commission": round(total_commission, 2),
        "total_slippage": round(total_slippage, 2),
        "total_return_pct": round(net_pnl / initial_capital * 100, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(pf, 2) if pf != float("inf") else "inf",
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "equity_curve": equity_curve,
        "trades": trades,
    }


def _empty_result(symbol, capital):
    return {
        "symbol": symbol, "total_trades": 0, "wins": 0, "losses": 0,
        "win_rate": 0, "gross_pnl": 0, "net_pnl": 0,
        "total_commission": 0, "total_slippage": 0, "total_return_pct": 0,
        "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
        "max_drawdown_pct": 0, "sharpe_ratio": 0,
        "equity_curve": [capital], "trades": [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. WALK-FORWARD VALIDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward_validate(df: pd.DataFrame, symbol: str, cfg: Config) -> Dict[str, Any]:
    """
    Walk-forward: split 60 days into 4 windows (15 days each).
    Train on first 10 days, test on last 5 days per window.
    """
    # Ensure DatetimeIndex for date splitting
    df_work = df.copy()
    if not isinstance(df_work.index, pd.DatetimeIndex):
        df_work.index = pd.to_datetime(df_work.index, utc=True)

    # Get unique trading dates
    if df_work.index.tz is not None:
        dates = df_work.index.tz_convert("US/Eastern").date
    else:
        dates = df_work.index.date
    unique_dates = sorted(set(dates))

    total_days = len(unique_dates)
    days_per_window = total_days // cfg.wf_windows

    print(f"\n  Walk-Forward: {total_days} trading days, {cfg.wf_windows} windows × {days_per_window} days")

    # Parameter grid
    combos = list(itertools.product(cfg.opt_atr_sl, cfg.opt_rr, cfg.opt_disp))
    print(f"  Testing {len(combos)} parameter combos per window\n")

    windows = []
    oos_equity = [cfg.initial_capital]  # Continuous OOS equity curve
    running_capital = cfg.initial_capital

    for w in range(cfg.wf_windows):
        w_start = w * days_per_window
        w_end = w_start + days_per_window if w < cfg.wf_windows - 1 else total_days

        train_end = w_start + cfg.wf_train_days
        test_start = train_end
        test_end = w_end

        if train_end >= total_days or test_start >= total_days:
            continue

        train_dates = set(unique_dates[w_start:train_end])
        test_dates = set(unique_dates[test_start:test_end])

        # Slice data by dates
        train_mask = pd.Series(dates, index=df_work.index).isin(train_dates)
        test_mask = pd.Series(dates, index=df_work.index).isin(test_dates)

        df_train = df_work[train_mask].copy()
        df_test = df_work[test_mask].copy()

        if len(df_train) < 50 or len(df_test) < 20:
            continue

        # ── OPTIMIZE on train ──
        best_score = -999999
        best_params = {}
        best_train_pnl = 0

        for atr_sl, rr, disp in combos:
            test_cfg = Config(
                atr_multiplier_sl=atr_sl, rr_ratio=rr, displacement_threshold=disp,
                session_filter=cfg.session_filter, use_htf_filter=cfg.use_htf_filter,
                use_adx_filter=cfg.use_adx_filter, use_rvol_filter=cfg.use_rvol_filter,
                adx_threshold=cfg.adx_threshold, rvol_multiplier=cfg.rvol_multiplier,
                ema_period=cfg.ema_period, adx_period=cfg.adx_period,
                rvol_period=cfg.rvol_period, etf_slippage_pct=cfg.etf_slippage_pct,
                commission_round_trip=cfg.commission_round_trip,
                futures_slippage_pts=cfg.futures_slippage_pts,
                initial_capital=cfg.initial_capital, risk_pct=cfg.risk_pct,
            )
            sigs = generate_signals(df_train, test_cfg)
            if len(sigs) < 3:
                continue
            m = execute_backtest(df_train, sigs, symbol, test_cfg)
            pnl = m.get("net_pnl", 0)
            trades = m.get("total_trades", 0)
            pf = m.get("profit_factor", 0)
            if isinstance(pf, str):
                pf = 10.0
            score = pnl * 0.5 + pf * 100 * 0.3 + trades * 0.2
            if score > best_score:
                best_score = score
                best_params = {"atr_sl": atr_sl, "rr": rr, "disp": disp}
                best_train_pnl = pnl

        # ── TEST on out-of-sample ──
        oos_cfg = Config(
            atr_multiplier_sl=best_params.get("atr_sl", 0.5),
            rr_ratio=best_params.get("rr", 2.5),
            displacement_threshold=best_params.get("disp", 1.0),
            session_filter=cfg.session_filter, use_htf_filter=cfg.use_htf_filter,
            use_adx_filter=cfg.use_adx_filter, use_rvol_filter=cfg.use_rvol_filter,
            adx_threshold=cfg.adx_threshold, rvol_multiplier=cfg.rvol_multiplier,
            ema_period=cfg.ema_period, adx_period=cfg.adx_period,
            rvol_period=cfg.rvol_period, etf_slippage_pct=cfg.etf_slippage_pct,
            commission_round_trip=cfg.commission_round_trip,
            futures_slippage_pts=cfg.futures_slippage_pts,
            initial_capital=running_capital, risk_pct=cfg.risk_pct,
        )
        oos_sigs = generate_signals(df_test, oos_cfg)
        oos_metrics = execute_backtest(df_test, oos_sigs, symbol, oos_cfg)

        oos_pnl = oos_metrics.get("net_pnl", 0)
        oos_trades = oos_metrics.get("total_trades", 0)
        oos_wr = oos_metrics.get("win_rate", 0)
        running_capital += oos_pnl

        # Append OOS equity points
        for eq in oos_metrics.get("equity_curve", [])[1:]:
            oos_equity.append(eq)

        win_str = "✅" if oos_pnl > 0 else "❌"
        print(f"  Window {w+1}/{cfg.wf_windows}: "
              f"train [{unique_dates[w_start]}→{unique_dates[min(train_end-1, total_days-1)]}] "
              f"test [{unique_dates[test_start]}→{unique_dates[min(test_end-1, total_days-1)]}]")
        print(f"    Best params: ATR={best_params.get('atr_sl','?')} RR={best_params.get('rr','?')} "
              f"Disp={best_params.get('disp','?')}")
        print(f"    Train P&L: ${best_train_pnl:+.2f} | {win_str} OOS P&L: ${oos_pnl:+.2f} | "
              f"WR: {oos_wr:.1f}% | Trades: {oos_trades}")

        windows.append({
            "window": w + 1, "best_params": best_params,
            "train_pnl": round(best_train_pnl, 2),
            "oos_pnl": round(oos_pnl, 2), "oos_trades": oos_trades,
            "oos_wr": round(oos_wr, 1), "profitable": oos_pnl > 0,
        })

    # ── ROBUSTNESS MATRIX ──
    profitable_windows = sum(1 for w in windows if w["profitable"])
    robust = profitable_windows >= 3

    print(f"\n  {'='*55}")
    print(f"  ROBUSTNESS MATRIX — {symbol}")
    print(f"  {'='*55}")
    print(f"  {'Window':>8} {'Params':>20} {'Train P&L':>12} {'OOS P&L':>12} {'Status':>8}")
    print(f"  {'─'*62}")
    for w in windows:
        p = w["best_params"]
        param_str = f"ATR={p.get('atr_sl','-')} RR={p.get('rr','-')}"
        status = "✅ PASS" if w["profitable"] else "❌ FAIL"
        print(f"  {w['window']:>8} {param_str:>20} {w['train_pnl']:>+12.2f} {w['oos_pnl']:>+12.2f} {status:>8}")

    print(f"\n  Profitable OOS windows: {profitable_windows}/{len(windows)}")
    print(f"  {'✅ ROBUST — LIVE-READY' if robust else '⚠️  FRAGILE — NOT LIVE-READY'}")

    # Find most common winning params → "Live-Ready" set
    winning_params = [w["best_params"] for w in windows if w["profitable"]]
    live_params = {}
    if winning_params:
        for key in winning_params[0]:
            vals = [p[key] for p in winning_params]
            live_params[key] = round(float(np.median(vals)), 2)
        print(f"\n  📊 LIVE-READY PARAMS: {live_params}")

    total_oos_pnl = sum(w["oos_pnl"] for w in windows)
    print(f"  📊 TOTAL OOS NET P&L: ${total_oos_pnl:+.2f}")

    # ── EQUITY CURVE PNG ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(oos_equity, color="#00d4ff", linewidth=1.5)
        ax.axhline(y=cfg.initial_capital, color="gray", linestyle="--", alpha=0.5)
        ax.fill_between(range(len(oos_equity)), cfg.initial_capital, oos_equity,
                        where=[e >= cfg.initial_capital for e in oos_equity],
                        alpha=0.15, color="#00ff88")
        ax.fill_between(range(len(oos_equity)), cfg.initial_capital, oos_equity,
                        where=[e < cfg.initial_capital for e in oos_equity],
                        alpha=0.15, color="#ff4444")
        ax.set_title(f"Walk-Forward OOS Equity — {symbol} Continuation", fontsize=14, color="white")
        ax.set_xlabel("Trade #", color="gray")
        ax.set_ylabel("Equity ($)", color="gray")
        ax.set_facecolor("#0a0e1a")
        fig.set_facecolor("#0a0e1a")
        ax.tick_params(colors="gray")
        ax.spines["bottom"].set_color("gray")
        ax.spines["left"].set_color("gray")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        png_path = os.path.join(RESULTS_DIR, f"walk_forward_equity_{symbol}.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  📈 Equity curve saved: {png_path}")
    except ImportError:
        print("  ⚠️  matplotlib not installed — skipping equity curve PNG")

    return {
        "symbol": symbol, "windows": windows,
        "profitable_windows": profitable_windows,
        "is_robust": robust, "live_params": live_params,
        "total_oos_pnl": round(total_oos_pnl, 2),
        "oos_equity": oos_equity,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_backtest_report(m: Dict[str, Any], label: str) -> None:
    pf = m.get("profit_factor", 0)
    pf_str = f"{pf:.2f}" if isinstance(pf, (int, float)) else str(pf)
    print(f"\n  {label}:")
    print(f"    Trades: {m.get('total_trades', 0)} | WR: {m.get('win_rate', 0):.1f}%")
    print(f"    Gross P&L: ${m.get('gross_pnl', 0):+.2f}")
    print(f"    Net P&L:   ${m.get('net_pnl', 0):+.2f}")
    print(f"    Commission: ${m.get('total_commission', 0):.2f} | Slippage: ${m.get('total_slippage', 0):.2f}")
    print(f"    PF: {pf_str} | Sharpe: {m.get('sharpe_ratio', 0):.2f} | Max DD: {m.get('max_drawdown_pct', 0):.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Institutional Continuation Engine")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--optimize", action="store_true", help="Run walk-forward optimization")
    parser.add_argument("--no-filter", action="store_true", help="Disable NY session filter")
    parser.add_argument("--no-htf", action="store_true", help="Disable 1H EMA filter")
    parser.add_argument("--no-adx", action="store_true", help="Disable ADX filter")
    parser.add_argument("--no-rvol", action="store_true", help="Disable RVOL filter")
    parser.add_argument("--no-vix", action="store_true", help="Disable VIX regime filter")
    parser.add_argument("--no-weekly", action="store_true", help="Disable weekly trend filter")
    parser.add_argument("--vix-threshold", type=float, default=25.0, help="VIX threshold (default: 25)")
    parser.add_argument("--atr-sl", type=float, default=None)
    parser.add_argument("--rr", type=float, default=None)
    parser.add_argument("--tf", type=str, default="15m", help="Timeframe: 5m, 15m, 1h (default: 15m)")
    args = parser.parse_args()

    cfg = Config()
    if args.no_filter:
        cfg.session_filter = False
    if args.no_htf:
        cfg.use_htf_filter = False
    if args.no_adx:
        cfg.use_adx_filter = False
    if args.no_rvol:
        cfg.use_rvol_filter = False
    if args.no_vix:
        cfg.use_vix_filter = False
    if args.no_weekly:
        cfg.use_weekly_trend_filter = False
    if args.vix_threshold:
        cfg.vix_threshold = args.vix_threshold
    if args.atr_sl:
        cfg.atr_multiplier_sl = args.atr_sl
    if args.rr:
        cfg.rr_ratio = args.rr

    symbols = [args.symbol] if args.symbol else cfg.symbols

    print("\n" + "=" * 60)
    print("  INSTITUTIONAL CONTINUATION ENGINE v2.0")
    print("  Friction · Confluence · Anti-Chop · Walk-Forward")
    print("=" * 60)
    print(f"  Session Filter: {'ON' if cfg.session_filter else 'OFF'}")
    print(f"  1H EMA Filter:  {'ON' if cfg.use_htf_filter else 'OFF'}")
    adx_str = f"ON (>{cfg.adx_threshold})" if cfg.use_adx_filter else "OFF"
    rvol_str = f"ON (>{cfg.rvol_multiplier}x)" if cfg.use_rvol_filter else "OFF"
    print(f"  ADX Filter:     {adx_str}")
    print(f"  RVOL Filter:    {rvol_str}")
    slip_display = cfg.etf_slippage_pct * 100
    print(f"  Friction:       ETF slip {slip_display:.3f}% | Comm ${cfg.commission_round_trip}/RT")

    tf = args.tf
    print(f"  Timeframe:      {tf}")

    if args.optimize:
        # Walk-forward validation
        all_wf = {}
        for sym in symbols:
            print(f"\n{'═'*60}")
            print(f"  {sym} — Walk-Forward Validation ({tf})")
            print(f"{'═'*60}")

            df = download_data(sym, period="60d", interval=tf)
            if df.empty:
                continue

            result = walk_forward_validate(df, sym, cfg)
            all_wf[sym] = result

        # Save
        out = {k: {kk: vv for kk, vv in v.items() if kk != "oos_equity"}
               for k, v in all_wf.items()}
        with open(os.path.join(RESULTS_DIR, "wf_institutional_results.json"), "w") as f:
            json.dump(out, f, indent=2, default=str)

    else:
        # Standard backtest
        for sym in symbols:
            print(f"\n{'═'*60}")
            print(f"  {sym} — Backtest ({tf})")

            df = download_data(sym, period="60d", interval=tf)
            if df.empty:
                continue

            print(f"  {len(df)} bars loaded")
            signals = generate_signals(df, cfg)
            metrics = execute_backtest(df, signals, sym, cfg)
            print_backtest_report(metrics, f"{sym} (all filters ON)")

            # Also run without filters for comparison
            cfg_raw = Config(
                atr_multiplier_sl=cfg.atr_multiplier_sl, rr_ratio=cfg.rr_ratio,
                displacement_threshold=cfg.displacement_threshold,
                session_filter=False, use_htf_filter=False,
                use_adx_filter=False, use_rvol_filter=False,
            )
            signals_raw = generate_signals(df, cfg_raw)
            metrics_raw = execute_backtest(df, signals_raw, sym, cfg_raw)
            print_backtest_report(metrics_raw, f"{sym} (NO filters — raw)")

    print("\n" + "=" * 60)
    print("  DONE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
