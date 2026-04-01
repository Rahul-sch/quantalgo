#!/usr/bin/env python3
"""
GOLDBACH FOREX SCANNER — PO3/PO9 Discount/Premium Model
Autonomous scanner for EUR/USD and GBP/USD using institutional Goldbach levels.

Dealing Range: Previous NY session (5 PM EST close) High/Low
Math: PO3 (33.3%) and PO9 (11.1%) levels in pip space
Output: Writes signals to trade_state.json using fcntl file locks

Usage:
    python3 goldbach_forex.py              # scan both pairs
    python3 goldbach_forex.py --pair EURUSD  # single pair
    python3 goldbach_forex.py --dry-run    # test without writing to ledger

Dealing Range Convention:
    Forex has no closing bell. We use the institutional standard:
    NY Session Close = 5:00 PM EST (22:00 UTC winter / 21:00 UTC summer)
    "Previous day" = 5 PM EST yesterday → 5 PM EST today
"""

import os
import sys
import json
import fcntl
import argparse
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Any, Optional, Tuple
from zoneinfo import ZoneInfo
from dataclasses import dataclass

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

EST = ZoneInfo("US/Eastern")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
STATE_LEDGER_FILE = os.path.join(RESULTS_DIR, "trade_state.json")
FOREX_LOG_FILE = os.path.join(RESULTS_DIR, "forex_trades.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Pair definitions
PAIRS = {
    # ── Majors ──
    "EURUSD": {
        "ticker": "EURUSD=X",
        "pip_size": 0.0001,
        "pip_value": 10.0,      # $10 per pip per standard lot (100k units)
        "spread_pips": 1.2,
        "name": "EUR/USD",
    },
    "GBPUSD": {
        "ticker": "GBPUSD=X",
        "pip_size": 0.0001,
        "pip_value": 10.0,
        "spread_pips": 1.5,
        "name": "GBP/USD",
    },
    "USDJPY": {
        "ticker": "JPY=X",
        "pip_size": 0.01,       # JPY pairs: 1 pip = 0.01
        "pip_value": 6.67,      # ~$6.67 per pip at USD/JPY ~150 (100k/rate)
        "spread_pips": 1.0,
        "name": "USD/JPY",
    },
    "AUDUSD": {
        "ticker": "AUDUSD=X",
        "pip_size": 0.0001,
        "pip_value": 10.0,
        "spread_pips": 1.4,
        "name": "AUD/USD",
    },
    "USDCHF": {
        "ticker": "CHF=X",
        "pip_size": 0.0001,
        "pip_value": 11.0,      # ~$11 per pip (inverse of CHF rate)
        "spread_pips": 1.5,
        "name": "USD/CHF",
    },
    "USDCAD": {
        "ticker": "CAD=X",
        "pip_size": 0.0001,
        "pip_value": 7.30,      # ~$7.30 per pip at USD/CAD ~1.37
        "spread_pips": 1.8,
        "name": "USD/CAD",
    },
    "NZDUSD": {
        "ticker": "NZDUSD=X",
        "pip_size": 0.0001,
        "pip_value": 10.0,
        "spread_pips": 1.8,
        "name": "NZD/USD",
    },
    # ── Volatile Crosses (bigger ranges, more data) ──
    "EURJPY": {
        "ticker": "EURJPY=X",
        "pip_size": 0.01,
        "pip_value": 6.67,
        "spread_pips": 1.5,
        "name": "EUR/JPY",
    },
    "GBPJPY": {
        "ticker": "GBPJPY=X",
        "pip_size": 0.01,
        "pip_value": 6.67,
        "spread_pips": 2.5,
        "name": "GBP/JPY",
    },
    "EURGBP": {
        "ticker": "EURGBP=X",
        "pip_size": 0.0001,
        "pip_value": 12.50,     # ~$12.50 per pip (GBP-denominated)
        "spread_pips": 1.2,
        "name": "EUR/GBP",
    },
}

# Risk parameters
ACCOUNT_SIZE    = 10_000.0  # $10k demo account
RISK_PCT        = 0.01      # 1% risk per trade
RISK_PER_TRADE  = ACCOUNT_SIZE * RISK_PCT  # hard cap: $100.00 per trade
MAX_DAILY_TRADES = 4        # max signals per day per pair


@dataclass
class GoldbachLevels:
    """Computed Goldbach levels for a dealing range."""
    pair: str
    date: str
    prev_high: float
    prev_low: float
    dealing_range: float
    equilibrium: float          # 50% of range (midpoint)
    # PO3 levels (33.3% increments from low)
    po3_1: float                # low + 33.3% (discount ceiling)
    po3_2: float                # low + 66.6% (premium floor)
    # PO9 levels (11.1% increments from low)
    po9: List[float]            # 9 levels from low to high
    # Pip math
    range_pips: float
    pip_size: float
    # Zone classification
    current_price: float
    zone: str                   # "deep_discount" | "discount" | "neutral" | "premium" | "deep_premium"


# ═══════════════════════════════════════════════════════════════════════════════
# SAFE I/O — shared with paper_trader.py (same fcntl locking)
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception:
        return default


def _safe_write_json(path: str, data: Any) -> None:
    lock_path = path + ".lock"
    tmp_path = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, path)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA — Fetch + NY Session Dealing Range
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_forex_data(pair_key: str, days: int = 10) -> pd.DataFrame:
    """
    Fetch intraday forex data via yfinance.
    Uses 15m bars for granularity, falls back to 1h if needed.
    """
    import yfinance as yf
    pair_info = PAIRS[pair_key]
    ticker = pair_info["ticker"]

    # yfinance max 60 days for 15m, 730 days for 1h
    try:
        df = yf.download(ticker, period=f"{days}d", interval="15m",
                         progress=False, auto_adjust=True)
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        if not df.empty and len(df) > 20:
            print(f"  [ok] {pair_info['name']}: {len(df)} bars (15m)")
            return df
    except Exception as e:
        print(f"  [warn] 15m fetch failed for {pair_key}: {e}")

    # Fallback to 1h
    try:
        df = yf.download(ticker, period=f"{days}d", interval="1h",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        if not df.empty:
            print(f"  [ok] {pair_info['name']}: {len(df)} bars (1h)")
            return df
    except Exception as e:
        print(f"  [error] {pair_key}: {e}")

    return pd.DataFrame()


def compute_ny_dealing_range(df: pd.DataFrame, pair_key: str) -> Optional[GoldbachLevels]:
    """
    Compute the previous NY session dealing range.

    NY Session = 5:00 PM EST (day-1) → 5:00 PM EST (day)
    This is the institutional standard for forex "daily" bars.

    Returns GoldbachLevels with all PO3/PO9 computations.
    """
    pair_info = PAIRS[pair_key]
    pip_size = pair_info["pip_size"]

    # Ensure timezone-aware index in US/Eastern
    idx = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True))
    df_est = df.copy()
    df_est.index = idx.tz_convert("US/Eastern")

    now_est = datetime.now(EST)
    ny_close_time = dtime(17, 0)  # 5:00 PM EST

    # Previous NY session: yesterday 5 PM → today 5 PM
    # If it's before 5 PM today, "current session" started yesterday at 5 PM
    # If it's after 5 PM today, "current session" started today at 5 PM
    if now_est.time() < ny_close_time:
        session_end = now_est.replace(hour=17, minute=0, second=0, microsecond=0)
        session_start = session_end - timedelta(days=1)
        prev_session_end = session_start
        prev_session_start = prev_session_end - timedelta(days=1)
    else:
        session_start = now_est.replace(hour=17, minute=0, second=0, microsecond=0)
        session_end = session_start + timedelta(days=1)
        prev_session_end = session_start
        prev_session_start = prev_session_end - timedelta(days=1)

    # Skip weekends — walk back to Friday 5 PM if needed
    # Sunday/Saturday → prev session is Thu 5PM → Fri 5PM
    while prev_session_start.weekday() >= 5:  # Sat=5, Sun=6
        prev_session_start -= timedelta(days=1)
        prev_session_end -= timedelta(days=1)
    # Also handle Monday before 5 PM — prev session is Thu→Fri
    if prev_session_end.weekday() >= 5:
        while prev_session_end.weekday() >= 5:
            prev_session_end -= timedelta(days=1)
            prev_session_start -= timedelta(days=1)

    # Filter bars to previous NY session
    mask = (df_est.index >= prev_session_start) & (df_est.index < prev_session_end)
    prev_session = df_est[mask]

    if prev_session.empty or len(prev_session) < 3:
        print(f"  [warn] Not enough bars for prev NY session "
              f"({prev_session_start.strftime('%m/%d %H:%M')} → "
              f"{prev_session_end.strftime('%m/%d %H:%M')})")
        return None

    prev_high = float(prev_session["High"].max())
    prev_low = float(prev_session["Low"].min())
    dealing_range = prev_high - prev_low

    if dealing_range < pip_size * 5:
        print(f"  [warn] Dealing range too small: {dealing_range / pip_size:.1f} pips")
        return None

    # Current price
    current_price = float(df_est["Close"].iloc[-1])

    # PO3 levels (33.3% from low)
    po3_1 = prev_low + dealing_range * (1 / 3)
    po3_2 = prev_low + dealing_range * (2 / 3)
    equilibrium = prev_low + dealing_range * 0.5

    # PO9 levels (11.1% increments)
    po9 = [prev_low + dealing_range * (i / 9) for i in range(1, 10)]

    # Range in pips
    range_pips = dealing_range / pip_size

    # Zone classification
    pct_in_range = (current_price - prev_low) / dealing_range if dealing_range > 0 else 0.5
    if pct_in_range <= 0.111:
        zone = "deep_discount"
    elif pct_in_range <= 0.333:
        zone = "discount"
    elif pct_in_range <= 0.667:
        zone = "neutral"
    elif pct_in_range <= 0.889:
        zone = "premium"
    else:
        zone = "deep_premium"

    return GoldbachLevels(
        pair=pair_key,
        date=prev_session_start.strftime("%Y-%m-%d"),
        prev_high=prev_high,
        prev_low=prev_low,
        dealing_range=dealing_range,
        equilibrium=equilibrium,
        po3_1=po3_1,
        po3_2=po3_2,
        po9=po9,
        range_pips=range_pips,
        pip_size=pip_size,
        current_price=current_price,
        zone=zone,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL LOGIC — Goldbach Bounce / Rejection
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ForexSignal:
    pair: str
    direction: str              # "buy" or "sell"
    model: str                  # "goldbach_bounce" or "goldbach_rejection"
    entry: float
    stop_loss: float
    take_profit: float
    sl_pips: float
    tp_pips: float
    rr_ratio: float
    zone: str
    reason: str
    levels: GoldbachLevels
    lot_size: float
    risk_amount: float


def scan_for_goldbach_signals(levels: GoldbachLevels) -> List[ForexSignal]:
    """
    Scan for Goldbach setups based on price zone.

    DISCOUNT BOUNCE (buy):
        Price is in discount zone (below PO3-1 / 33% of range)
        Entry at or near PO9-1 or PO9-2 support
        SL below prev low (or 1.5x dealing range PO9 increment below entry)
        TP at equilibrium (50%) or PO3-2 (66%)

    PREMIUM REJECTION (sell):
        Price is in premium zone (above PO3-2 / 66% of range)
        Entry at or near PO9-7 or PO9-8 resistance
        SL above prev high
        TP at equilibrium or PO3-1

    NO TRADE in neutral zone (33%-66%).
    """
    signals = []
    pair_info = PAIRS[levels.pair]
    pip_size = pair_info["pip_size"]
    po9_increment = levels.dealing_range / 9

    price = levels.current_price

    MIN_SL_PIPS = 5    # minimum pips clearance between entry and SL
    FALLBACK_SL_PIPS = 15  # fallback SL distance when static SL is inverted/too close

    # ── DISCOUNT BOUNCE (BUY) ──
    if levels.zone in ("deep_discount", "discount"):
        # Entry: current price (already in discount)
        entry = price
        # SL: below prev low by 1 PO9 increment (buffer)
        stop_loss = levels.prev_low - po9_increment

        # Dynamic SL check: if static SL is NOT at least MIN_SL_PIPS below entry,
        # override to entry - FALLBACK_SL_PIPS (handles inverted/too-close levels
        # on parabolic days where price has blown below the prev dealing range).
        if (entry - stop_loss) / pip_size < MIN_SL_PIPS:
            stop_loss = entry - FALLBACK_SL_PIPS * pip_size

        # TP: equilibrium (50% of range) — conservative target
        take_profit = levels.equilibrium

        sl_pips = abs(entry - stop_loss) / pip_size
        tp_pips = abs(take_profit - entry) / pip_size
        rr = tp_pips / sl_pips if sl_pips > 0 else 0

        # Only take if RR >= 1.5
        if rr >= 1.5 and sl_pips <= 50:  # max 50 pip SL
            # Hard cap: $100.00 risk regardless of pip math
            risk_amount = RISK_PER_TRADE
            lot_size = risk_amount / (sl_pips * pair_info["pip_value"]) if sl_pips > 0 else 0

            # Nearest PO9 support level
            nearest_po9 = min(levels.po9, key=lambda x: abs(x - price))
            nearest_po9_idx = levels.po9.index(nearest_po9) + 1

            signals.append(ForexSignal(
                pair=levels.pair,
                direction="buy",
                model="goldbach_bounce",
                entry=round(entry, 5),
                stop_loss=round(stop_loss, 5),
                take_profit=round(take_profit, 5),
                sl_pips=round(sl_pips, 1),
                tp_pips=round(tp_pips, 1),
                rr_ratio=round(rr, 2),
                zone=levels.zone,
                reason=(f"Discount bounce — price at PO9-{nearest_po9_idx} "
                        f"({price:.5f}), target equilibrium ({levels.equilibrium:.5f})"),
                levels=levels,
                lot_size=round(lot_size, 2),
                risk_amount=round(risk_amount, 2),  # always $100.00
            ))

    # ── PREMIUM REJECTION (SELL) ──
    elif levels.zone in ("deep_premium", "premium"):
        entry = price
        stop_loss = levels.prev_high + po9_increment
        take_profit = levels.equilibrium

        # Dynamic SL check: if static SL is NOT at least MIN_SL_PIPS above entry,
        # override to entry + FALLBACK_SL_PIPS (handles inverted/too-close levels
        # on parabolic days where price has blown far above the prev dealing range).
        if (stop_loss - entry) / pip_size < MIN_SL_PIPS:
            stop_loss = entry + FALLBACK_SL_PIPS * pip_size

        sl_pips = abs(stop_loss - entry) / pip_size
        tp_pips = abs(entry - take_profit) / pip_size
        rr = tp_pips / sl_pips if sl_pips > 0 else 0

        if rr >= 1.5 and sl_pips <= 50:
            # Hard cap: $100.00 risk regardless of pip math
            risk_amount = RISK_PER_TRADE
            lot_size = risk_amount / (sl_pips * pair_info["pip_value"]) if sl_pips > 0 else 0

            nearest_po9 = min(levels.po9, key=lambda x: abs(x - price))
            nearest_po9_idx = levels.po9.index(nearest_po9) + 1

            signals.append(ForexSignal(
                pair=levels.pair,
                direction="sell",
                model="goldbach_rejection",
                entry=round(entry, 5),
                stop_loss=round(stop_loss, 5),
                take_profit=round(take_profit, 5),
                sl_pips=round(sl_pips, 1),
                tp_pips=round(tp_pips, 1),
                rr_ratio=round(rr, 2),
                zone=levels.zone,
                reason=(f"Premium rejection — price at PO9-{nearest_po9_idx} "
                        f"({price:.5f}), target equilibrium ({levels.equilibrium:.5f})"),
                levels=levels,
                lot_size=round(lot_size, 2),
                risk_amount=round(risk_amount, 2),
            ))

    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# LEDGER — Write to shared trade_state.json (fcntl locked)
# ═══════════════════════════════════════════════════════════════════════════════

def _forex_signal_id(sig: ForexSignal) -> str:
    """Unique fingerprint for a forex signal."""
    now = datetime.now(EST)
    return f"{sig.pair}_{now.strftime('%Y%m%d')}_{sig.entry:.5f}_{sig.direction}"


def write_signal_to_ledger(sig: ForexSignal, dry_run: bool = False) -> str:
    """
    Write a forex signal to trade_state.json in the same format as the
    equity FVG engine. Uses fcntl file locks for safe parallel access.

    Returns the signal_id.
    """
    now = datetime.now(EST)
    signal_id = _forex_signal_id(sig)
    trade_id = f"{sig.pair}_goldbach_{now.strftime('%Y%m%d_%H%M%S')}"

    trade = {
        "id": trade_id,
        "timestamp": now.isoformat(),
        "symbol": sig.pair,
        "timeframe": "daily_goldbach",
        "direction": sig.direction,
        "entry_price": sig.entry,
        "stop_loss": sig.stop_loss,
        "take_profit": sig.take_profit,
        "current_price": sig.levels.current_price,
        "sl_distance": round(sig.sl_pips * sig.levels.pip_size, 5),
        "rr_ratio": sig.rr_ratio,
        "position_size": sig.lot_size,
        "risk_amount": sig.risk_amount,
        "gross_potential": round(sig.tp_pips * sig.lot_size * PAIRS[sig.pair]["pip_value"] / 100, 2),
        "net_potential": 0,
        "reason": sig.reason,
        "model": sig.model,
        "sl_pips": sig.sl_pips,
        "tp_pips": sig.tp_pips,
        "lot_size": sig.lot_size,
        # Goldbach metadata
        "dealing_range_pips": round(sig.levels.range_pips, 1),
        "prev_high": sig.levels.prev_high,
        "prev_low": sig.levels.prev_low,
        "equilibrium": sig.levels.equilibrium,
        "zone": sig.zone,
        "po3_1": sig.levels.po3_1,
        "po3_2": sig.levels.po3_2,
        # Status
        "status": "pending",
        "signal_id": signal_id,
        "exit_price": None,
        "exit_timestamp": None,
        "gross_pnl": None,
        "net_pnl": None,
        "commission": 0,
        "slippage": None,
    }

    if dry_run:
        print(f"  [dry-run] Would write to ledger: {signal_id}")
        return signal_id

    # ── Write to shared trade_state.json (parallel-safe with equity engine) ──
    today = now.strftime("%Y-%m-%d")
    ledger = _safe_read_json(STATE_LEDGER_FILE, {})
    if ledger.get("date") != today:
        ledger = {"date": today, "pending_orders": [], "alerted_signal_ids": []}

    # Dedup check
    existing_ids = [p["signal_id"] for p in ledger.get("pending_orders", [])]
    if signal_id in existing_ids:
        print(f"  [ledger] Already in ledger: {signal_id[:40]}")
        return signal_id

    # Compute expiry — forex signals expire at next NY close (5 PM EST)
    if now.time() < dtime(17, 0):
        expires = now.replace(hour=17, minute=0, second=0, microsecond=0)
    else:
        expires = (now + timedelta(days=1)).replace(hour=17, minute=0, second=0, microsecond=0)

    order = {
        "signal_id": signal_id,
        "trade_id": trade_id,
        "direction": sig.direction,
        "entry_price": sig.entry,
        "stop_loss": sig.stop_loss,
        "take_profit": sig.take_profit,
        "armed_at": now.isoformat(),
        "armed_bar_time": now.isoformat(),
        "expires_after": expires.isoformat(),
        "status": "pending",
        "model": "goldbach_forex",
    }
    ledger["pending_orders"].append(order)
    if signal_id not in ledger.get("alerted_signal_ids", []):
        ledger.setdefault("alerted_signal_ids", []).append(signal_id)
    _safe_write_json(STATE_LEDGER_FILE, ledger)
    print(f"  [ledger] ✅ Written to trade_state.json: {signal_id[:50]}")

    # ── Also log to forex-specific trade file ──
    forex_trades = _safe_read_json(FOREX_LOG_FILE, [])
    forex_trades.append(trade)
    _safe_write_json(FOREX_LOG_FILE, forex_trades)

    return signal_id


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def print_goldbach_levels(levels: GoldbachLevels) -> None:
    """Pretty-print the Goldbach levels for a pair."""
    pip = levels.pip_size
    pair_info = PAIRS[levels.pair]

    zone_emoji = {
        "deep_discount": "🟢🟢",
        "discount": "🟢",
        "neutral": "⚪",
        "premium": "🔴",
        "deep_premium": "🔴🔴",
    }

    print(f"\n  {'─'*50}")
    print(f"  📊 {pair_info['name']} — Goldbach Analysis")
    print(f"  {'─'*50}")
    print(f"  Prev NY Session ({levels.date}):")
    print(f"     High:  {levels.prev_high:.5f}")
    print(f"     Low:   {levels.prev_low:.5f}")
    print(f"     Range: {levels.range_pips:.1f} pips")
    print()
    print(f"  PO3 Levels:")
    print(f"     PO3-1 (33%): {levels.po3_1:.5f}  ── discount ceiling")
    print(f"     Equil (50%): {levels.equilibrium:.5f}  ── midpoint")
    print(f"     PO3-2 (67%): {levels.po3_2:.5f}  ── premium floor")
    print()
    print(f"  PO9 Levels (11.1% increments):")
    for i, level in enumerate(levels.po9):
        marker = " ◄── PRICE" if abs(level - levels.current_price) < (levels.dealing_range / 18) else ""
        zone_marker = ""
        if i == 2:
            zone_marker = "  [PO3-1]"
        elif i == 5:
            zone_marker = "  [PO3-2]"
        elif i == 4:
            zone_marker = "  [equil]"
        print(f"     PO9-{i+1}: {level:.5f}{zone_marker}{marker}")
    print()
    pct = (levels.current_price - levels.prev_low) / levels.dealing_range * 100
    emoji = zone_emoji.get(levels.zone, "")
    print(f"  Current: {levels.current_price:.5f} ({pct:.1f}% of range)")
    print(f"  Zone:    {emoji} {levels.zone.upper().replace('_', ' ')}")


def print_signal(sig: ForexSignal) -> None:
    """Pretty-print a forex signal."""
    emoji = "🟢" if sig.direction == "buy" else "🔴"
    pair_info = PAIRS[sig.pair]
    print(f"\n  {'═'*50}")
    print(f"  {emoji} SIGNAL: {pair_info['name']} {sig.direction.upper()}")
    print(f"  Model:  {sig.model}")
    print(f"  {'═'*50}")
    print(f"  Entry:       {sig.entry:.5f}")
    print(f"  Stop Loss:   {sig.stop_loss:.5f}  ({sig.sl_pips:.1f} pips)")
    print(f"  Take Profit: {sig.take_profit:.5f}  ({sig.tp_pips:.1f} pips)")
    print(f"  R:R:         {sig.rr_ratio:.1f}:1")
    print(f"  Lot Size:    {sig.lot_size:.2f}")
    print(f"  Risk:        ${sig.risk_amount:.2f}")
    print(f"  Zone:        {sig.zone}")
    print(f"  Reason:      {sig.reason}")


# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM ALERTS
# ═══════════════════════════════════════════════════════════════════════════════

def alert_forex_signal(sig: ForexSignal) -> None:
    """Send a Telegram alert for a forex Goldbach signal."""
    try:
        from notifier import _send_telegram
        pair_info = PAIRS[sig.pair]
        emoji = "🟢" if sig.direction == "buy" else "🔴"

        msg = (
            f"{emoji} <b>FOREX SIGNAL: {pair_info['name']} {sig.direction.upper()}</b>\n"
            f"Model: Goldbach {sig.model.replace('goldbach_', '').title()}\n"
            f"─────────────────────────\n"
            f"Entry:       {sig.entry:.5f}\n"
            f"Stop Loss:   {sig.stop_loss:.5f} ({sig.sl_pips:.0f} pips)\n"
            f"Take Profit: {sig.take_profit:.5f} ({sig.tp_pips:.0f} pips)\n"
            f"R:R:         {sig.rr_ratio:.1f}:1\n"
            f"Lot Size:    {sig.lot_size:.2f}\n"
            f"Risk:        ${sig.risk_amount:.2f}\n"
            f"─────────────────────────\n"
            f"Zone: {sig.zone.upper().replace('_', ' ')}\n"
            f"Range: {sig.levels.range_pips:.0f} pips\n"
            f"{sig.reason}"
        )
        _send_telegram(msg)
    except Exception as e:
        # Fall back to openclaw CLI
        try:
            import subprocess
            pair_info = PAIRS[sig.pair]
            short_msg = (
                f"{'🟢' if sig.direction == 'buy' else '🔴'} "
                f"FOREX: {pair_info['name']} {sig.direction.upper()} @ {sig.entry:.5f} | "
                f"SL {sig.sl_pips:.0f}p | TP {sig.tp_pips:.0f}p | RR {sig.rr_ratio:.1f}:1"
            )
            subprocess.run(
                ["openclaw", "message", "send", "-t", "6515146575",
                 "--channel", "telegram", "-m", short_msg],
                capture_output=True, timeout=10
            )
            print(f"  [notifier] Forex alert sent via CLI ✅")
        except Exception as e2:
            print(f"  [notifier] Forex alert failed: {e2}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def scan_pair(pair_key: str, dry_run: bool = False) -> List[ForexSignal]:
    """Full scan pipeline for a single forex pair."""
    pair_info = PAIRS[pair_key]
    print(f"\n  📡 Fetching {pair_info['name']} data...")
    df = fetch_forex_data(pair_key, days=7)
    if df.empty:
        print(f"  ❌ No data for {pair_info['name']}")
        return []

    print(f"  📐 Computing NY session dealing range...")
    levels = compute_ny_dealing_range(df, pair_key)
    if levels is None:
        return []

    print_goldbach_levels(levels)

    signals = scan_for_goldbach_signals(levels)
    if not signals:
        print(f"\n  📭 No Goldbach setup for {pair_info['name']} — price in {levels.zone} zone")
        if levels.zone == "neutral":
            print(f"     Price at equilibrium — no edge for bounce or rejection")
        return []

    for sig in signals:
        print_signal(sig)
        signal_id = write_signal_to_ledger(sig, dry_run=dry_run)

        if not dry_run:
            try:
                alert_forex_signal(sig)
            except Exception as e:
                print(f"  [alert] Failed: {e}")

    return signals


def main():
    parser = argparse.ArgumentParser(
        description="Goldbach Forex Scanner — PO3/PO9 Discount/Premium Model"
    )
    parser.add_argument("--pair", help="Single pair to scan (EURUSD or GBPUSD)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to ledger or alert")
    args = parser.parse_args()

    now = datetime.now(EST)
    print("\n" + "=" * 55)
    print("  📊 GOLDBACH FOREX SCANNER")
    print(f"  {now.strftime('%I:%M %p ET · %a %b %d, %Y')}")
    print("=" * 55)

    pairs_to_scan = [args.pair.upper()] if args.pair else list(PAIRS.keys())
    all_signals = []

    for pair_key in pairs_to_scan:
        if pair_key not in PAIRS:
            print(f"  ❌ Unknown pair: {pair_key}. Options: {list(PAIRS.keys())}")
            continue
        sigs = scan_pair(pair_key, dry_run=args.dry_run)
        all_signals.extend(sigs)

    # Summary
    print(f"\n{'='*55}")
    print(f"  📋 SUMMARY")
    print(f"{'='*55}")
    print(f"  Pairs scanned:  {len(pairs_to_scan)}")
    print(f"  Signals found:  {len(all_signals)}")
    for sig in all_signals:
        emoji = "🟢" if sig.direction == "buy" else "🔴"
        print(f"  {emoji} {PAIRS[sig.pair]['name']} {sig.direction.upper()} | "
              f"RR {sig.rr_ratio:.1f}:1 | {sig.zone}")
    if not all_signals:
        print(f"  📭 No setups — all pairs in neutral zone or insufficient RR")

    if args.dry_run:
        print(f"\n  ⚠️  DRY RUN — nothing written to ledger")
    print()


if __name__ == "__main__":
    main()
