#!/usr/bin/env python3
"""
FOREX FVG KILLZONE SCANNER — ICT Continuation Model for Live Forex
===================================================================
Scans 10 forex pairs for FVG continuation setups during London/NY killzones.

Model: 15m FVG detection → displacement filter (1.0x ATR body) → retest entry
       → Break-Even at IRL → TP at 2.0x risk

Killzones:
  London Open:        03:00–05:00 ET
  NY Open + Follow:   08:00–12:00 ET
  Late Day:           14:00–15:00 ET

Writes signals to shared trade_state.json with model tag "fvg_killzone".
Conflict resolution vs Goldbach handled by forex_manager.py.

Usage:
    python3 forex_fvg_scanner.py             # scan all 10 pairs
    python3 forex_fvg_scanner.py --pair EURUSD
    python3 forex_fvg_scanner.py --dry-run   # no ledger writes
"""

import os
import sys
import json
import fcntl
import argparse
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Any, Optional, Tuple
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

EST = ZoneInfo("US/Eastern")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
STATE_LEDGER_FILE = os.path.join(RESULTS_DIR, "trade_state.json")
FOREX_LOG_FILE = os.path.join(RESULTS_DIR, "forex_trades.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Import pair definitions from goldbach_forex.py
from goldbach_forex import PAIRS, RISK_PER_TRADE, _safe_read_json, _safe_write_json

# ── Model parameters ──
ATR_PERIOD      = 14
ATR_SL_MULT     = 0.5      # SL = 0.5x ATR (tight, same as equity engine)
RR_RATIO        = 2.0      # TP = 2.0x risk
DISPLACEMENT_MULT = 1.0    # FVG candle body must be >= 1.0x ATR
RETEST_MAX_BARS = 20       # max bars to wait for FVG retest
IRL_PIVOT_BARS  = 5        # local pivot lookback

# ── Killzones (minutes since midnight ET) ──
# London Open
LONDON_START = 180    # 03:00 ET
LONDON_END   = 300    # 05:00 ET
# NY Open + Follow-Through (expanded from 11:00 to 12:00 per TOD analysis)
NY_START     = 480    # 08:00 ET
NY_END       = 720    # 12:00 ET  ← was 11:00, +$4,589 P&L at 74% WR
# Late Day micro-killzone (14:00-15:00 ET — 83% WR, PF 5.37 in backtest)
LATE_START   = 840    # 14:00 ET
LATE_END     = 900    # 15:00 ET

# ── Realistic costs ──
SPREAD_PIPS = 1.5     # spread penalty at entry

# ── Risk limits ──
MAX_SIGNALS_PER_PAIR = 2   # max FVG signals per pair per scan
MAX_TOTAL_SIGNALS    = 6   # max total signals per scan


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCH
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_15m_data(pair_key: str, days: int = 5) -> pd.DataFrame:
    """Fetch 15m forex data for FVG detection."""
    import yfinance as yf
    pair_info = PAIRS[pair_key]
    ticker = pair_info["ticker"]

    try:
        df = yf.download(ticker, period=f"{days}d", interval="15m",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        if not df.empty and len(df) > 30:
            return df
    except Exception as e:
        print(f"  [error] {pair_key}: {e}")

    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# FVG DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_fvgs(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    3-candle FVG detection.
    Bullish FVG: candle[i-2].High < candle[i].Low
    Bearish FVG: candle[i-2].Low > candle[i].High
    
    Returns arrays indexed by the MIDDLE candle (the displacement bar).
    """
    high = df["High"].values
    low = df["Low"].values
    n = len(df)

    bull_top = np.full(n, np.nan)
    bull_bot = np.full(n, np.nan)
    bear_top = np.full(n, np.nan)
    bear_bot = np.full(n, np.nan)

    for i in range(2, n):
        if high[i-2] < low[i]:
            bull_top[i-1] = low[i]
            bull_bot[i-1] = high[i-2]
        if low[i-2] > high[i]:
            bear_top[i-1] = low[i-2]
            bear_bot[i-1] = high[i]

    return bull_top, bull_bot, bear_top, bear_bot


# ═══════════════════════════════════════════════════════════════════════════════
# IRL PIVOT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def find_local_pivots(high: np.ndarray, low: np.ndarray,
                      lookback: int = IRL_PIVOT_BARS) -> Tuple[np.ndarray, np.ndarray]:
    n = len(high)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        window_h = high[max(0, i-lookback):i+lookback+1]
        if high[i] == window_h.max():
            pivot_highs[i] = high[i]
        window_l = low[max(0, i-lookback):i+lookback+1]
        if low[i] == window_l.min():
            pivot_lows[i] = low[i]

    return pivot_highs, pivot_lows


def find_irl_target(direction: str, entry: float, tp: float,
                    bar_idx: int, pivot_highs: np.ndarray,
                    pivot_lows: np.ndarray) -> Optional[float]:
    """Find nearest IRL target between entry and TP."""
    look_start = max(0, bar_idx - 100)
    look_end = bar_idx

    if direction == "buy":
        candidates = [pivot_highs[j] for j in range(look_start, look_end)
                      if not np.isnan(pivot_highs[j]) and entry < pivot_highs[j] < tp]
        return min(candidates) if candidates else None
    else:
        candidates = [pivot_lows[j] for j in range(look_start, look_end)
                      if not np.isnan(pivot_lows[j]) and tp < pivot_lows[j] < entry]
        return max(candidates) if candidates else None


# ═══════════════════════════════════════════════════════════════════════════════
# KILLZONE CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def is_in_killzone(ts: pd.Timestamp) -> Tuple[bool, str]:
    """Check if a timestamp falls in a forex killzone."""
    try:
        et = ts.tz_convert("US/Eastern")
    except TypeError:
        et = ts.tz_localize("UTC").tz_convert("US/Eastern")

    mins = et.hour * 60 + et.minute

    if LONDON_START <= mins < LONDON_END:
        return True, "London"
    if NY_START <= mins < NY_END:
        return True, "NY"
    if LATE_START <= mins < LATE_END:
        return True, "Late-NY"
    return False, "none"


def is_killzone_now() -> Tuple[bool, str]:
    """Check if current time is in a killzone."""
    now = datetime.now(EST)
    mins = now.hour * 60 + now.minute
    if LONDON_START <= mins < LONDON_END:
        return True, "London"
    if NY_START <= mins < NY_END:
        return True, "NY"
    if LATE_START <= mins < LATE_END:
        return True, "Late-NY"
    return False, "none"


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL SCANNING
# ═══════════════════════════════════════════════════════════════════════════════

def scan_pair_fvg(pair_key: str, dry_run: bool = False) -> List[Dict]:
    """
    Scan a single forex pair for FVG continuation setups.
    Only fires during London/NY killzones.
    
    Returns list of signal dicts ready for ledger.
    """
    pair_info = PAIRS[pair_key]
    pip_size = pair_info["pip_size"]
    signals = []

    # Fetch data
    df = fetch_15m_data(pair_key, days=5)
    if df.empty or len(df) < 50:
        print(f"  ❌ {pair_info['name']}: insufficient data ({len(df)} bars)")
        return []

    print(f"  [ok] {pair_info['name']}: {len(df)} bars (15m)")

    # Current bar must be in killzone
    last_ts = df.index[-1]
    in_kz, kz_name = is_in_killzone(last_ts)
    if not in_kz:
        # Also check current time (data might be slightly delayed)
        in_kz, kz_name = is_killzone_now()

    if not in_kz:
        now_et = datetime.now(EST)
        print(f"  ⏰ {pair_info['name']}: Outside killzone ({now_et.strftime('%H:%M ET')})")
        return []

    print(f"  🎯 {pair_info['name']}: In {kz_name} killzone")

    # Compute indicators
    atr = compute_atr(df)
    bull_top, bull_bot, bear_top, bear_bot = detect_fvgs(df)
    pivot_highs, pivot_lows = find_local_pivots(df["High"].values, df["Low"].values)

    high = df["High"].values
    low = df["Low"].values
    open_ = df["Open"].values
    close = df["Close"].values

    # Track armed FVGs
    armed: Dict[int, tuple] = {}
    spread_penalty = SPREAD_PIPS * pip_size

    # Scan recent bars (last 40) for FVG creation + retest
    scan_start = max(30, len(df) - 40)

    for i in range(scan_start, len(df)):
        atr_val = atr.iloc[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        # Displacement check: FVG candle body must be >= 1.0x ATR
        body = abs(close[i] - open_[i])

        # Check killzone on FVG-creating bar
        bar_in_kz, _ = is_in_killzone(df.index[i])

        # Arm new FVGs (only if displacement + killzone)
        if body >= atr_val * DISPLACEMENT_MULT and bar_in_kz:
            if not np.isnan(bull_top[i]):
                armed[i] = ("buy", float(bull_top[i]), float(bull_bot[i]),
                            float(atr_val), i)
            if not np.isnan(bear_top[i]):
                armed[i] = ("sell", float(bear_bot[i]), float(bear_top[i]),
                            float(atr_val), i)

        # Check for retests on armed FVGs
        to_remove = []
        for fvg_bar, (direction, limit_price, other_edge, atr_arm, armed_bar) in armed.items():
            bars_elapsed = i - armed_bar
            if bars_elapsed > RETEST_MAX_BARS:
                to_remove.append(fvg_bar)
                continue
            if bars_elapsed == 0:
                continue

            # Invalidation
            if direction == "buy" and low[i] < other_edge:
                to_remove.append(fvg_bar)
                continue
            if direction == "sell" and high[i] > other_edge:
                to_remove.append(fvg_bar)
                continue

            # Retest check
            if not (low[i] <= limit_price <= high[i]):
                continue

            # Build trade with spread penalty
            if direction == "buy":
                entry = limit_price + spread_penalty
                sl = entry - atr_arm * ATR_SL_MULT
                risk = entry - sl
                if risk <= 0:
                    continue
                tp = entry + risk * RR_RATIO
            else:
                entry = limit_price - spread_penalty
                sl = entry + atr_arm * ATR_SL_MULT
                risk = sl - entry
                if risk <= 0:
                    continue
                tp = entry - risk * RR_RATIO

            # Find IRL target for break-even
            irl = find_irl_target(direction, entry, tp, i, pivot_highs, pivot_lows)

            # Calculate lot size from risk
            sl_pips = risk / pip_size
            tp_pips = abs(tp - entry) / pip_size
            rr = tp_pips / sl_pips if sl_pips > 0 else 0

            if rr < 1.5 or sl_pips > 50 or sl_pips < 3:
                continue

            lot_size = RISK_PER_TRADE / (sl_pips * pair_info["pip_value"])

            signals.append({
                "pair": pair_key,
                "direction": direction,
                "entry": round(entry, 5),
                "stop_loss": round(sl, 5),
                "take_profit": round(tp, 5),
                "sl_pips": round(sl_pips, 1),
                "tp_pips": round(tp_pips, 1),
                "rr_ratio": round(rr, 2),
                "lot_size": round(lot_size, 3),
                "risk_amount": round(RISK_PER_TRADE, 2),
                "irl_target": round(irl, 5) if irl else None,
                "killzone": kz_name,
                "fvg_bar_time": df.index[fvg_bar].isoformat(),
                "retest_bar_time": df.index[i].isoformat(),
                "atr": round(atr_arm, 6),
                "model": "fvg_killzone",
            })
            to_remove.append(fvg_bar)

            if len(signals) >= MAX_SIGNALS_PER_PAIR:
                break

        for k in to_remove:
            armed.pop(k, None)

        if len(signals) >= MAX_SIGNALS_PER_PAIR:
            break

    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# CONFLICT RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

def check_conflicts(fvg_signal: Dict, ledger: Dict) -> Tuple[bool, str]:
    """
    Conflict Resolution Matrix:
    
    Check if a Goldbach signal exists for the same pair in the current ledger.
    
    SAME DIRECTION:  Allow both (high confluence) → return (True, "confluence")
    OPPOSITE DIRECTION: FVG takes priority, block Goldbach → return (True, "fvg_priority")
    NO CONFLICT:     Allow → return (True, "no_conflict")
    
    Returns: (allow_fvg, reason)
    """
    pair = fvg_signal["pair"]
    fvg_dir = fvg_signal["direction"]

    active_orders = ledger.get("pending_orders", [])
    goldbach_orders = [
        o for o in active_orders
        if o.get("model", "").startswith("goldbach")
        and o.get("status") in ("pending", "open")
        and o.get("trade_id", "").startswith(pair)
    ]

    if not goldbach_orders:
        return True, "no_conflict"

    for gb_order in goldbach_orders:
        gb_dir = gb_order.get("direction", "")
        if gb_dir == fvg_dir:
            print(f"  ✅ CONFLUENCE: Goldbach {gb_dir.upper()} + FVG {fvg_dir.upper()} "
                  f"on {pair} — both allowed")
            return True, "confluence"
        else:
            # Opposite direction — FVG priority, mark Goldbach as blocked
            print(f"  ⚠️  CONFLICT: Goldbach {gb_dir.upper()} vs FVG {fvg_dir.upper()} "
                  f"on {pair} — FVG takes priority")
            gb_order["status"] = "blocked_by_fvg"
            gb_order["blocked_reason"] = (
                f"FVG Killzone {fvg_dir.upper()} signal conflicts — "
                f"momentum model takes priority"
            )
            gb_order["blocked_timestamp"] = datetime.now(EST).isoformat()
            return True, "fvg_priority"

    return True, "no_conflict"


# ═══════════════════════════════════════════════════════════════════════════════
# LEDGER WRITE
# ═══════════════════════════════════════════════════════════════════════════════

def write_fvg_to_ledger(signal: Dict, dry_run: bool = False) -> Optional[str]:
    """Write FVG signal to shared trade_state.json."""
    now = datetime.now(EST)
    today = now.strftime("%Y-%m-%d")
    pair = signal["pair"]
    direction = signal["direction"]

    signal_id = f"{pair}_{today.replace('-','')}_{signal['entry']:.5f}_{direction}_fvg"
    trade_id = f"{pair}_fvg_{now.strftime('%Y%m%d_%H%M%S')}"

    if dry_run:
        print(f"  [dry-run] Would write: {signal_id}")
        return signal_id

    # Load ledger
    ledger = _safe_read_json(STATE_LEDGER_FILE, {})
    if ledger.get("date") != today:
        ledger = {"date": today, "pending_orders": [], "alerted_signal_ids": []}

    # Dedup
    existing_ids = [p.get("signal_id", "") for p in ledger.get("pending_orders", [])]
    if signal_id in existing_ids:
        print(f"  [ledger] Already exists: {signal_id[:50]}")
        return None

    # Conflict resolution
    allow, conflict_reason = check_conflicts(signal, ledger)
    if not allow:
        print(f"  [ledger] Blocked: {conflict_reason}")
        return None

    # Expiry: next NY close (5 PM ET)
    if now.time() < dtime(17, 0):
        expires = now.replace(hour=17, minute=0, second=0, microsecond=0)
    else:
        expires = (now + timedelta(days=1)).replace(hour=17, minute=0, second=0, microsecond=0)

    order = {
        "signal_id": signal_id,
        "trade_id": trade_id,
        "direction": direction,
        "entry_price": signal["entry"],
        "stop_loss": signal["stop_loss"],
        "take_profit": signal["take_profit"],
        "armed_at": now.isoformat(),
        "armed_bar_time": signal.get("retest_bar_time", now.isoformat()),
        "expires_after": expires.isoformat(),
        "status": "pending",
        "model": "fvg_killzone",
        "lot_size": signal["lot_size"],
        "risk_amount": signal["risk_amount"],
        "sl_pips": signal["sl_pips"],
        "tp_pips": signal["tp_pips"],
        "rr_ratio": signal["rr_ratio"],
        "killzone": signal["killzone"],
        "irl_target": signal.get("irl_target"),
        "conflict_resolution": conflict_reason,
    }

    ledger["pending_orders"].append(order)
    if signal_id not in ledger.get("alerted_signal_ids", []):
        ledger.setdefault("alerted_signal_ids", []).append(signal_id)

    _safe_write_json(STATE_LEDGER_FILE, ledger)
    print(f"  [ledger] ✅ Written: {signal_id[:60]}")

    # Also log to forex trades file
    forex_trades = _safe_read_json(FOREX_LOG_FILE, [])
    log_entry = {**signal, "signal_id": signal_id, "trade_id": trade_id,
                 "timestamp": now.isoformat(), "conflict_resolution": conflict_reason}
    forex_trades.append(log_entry)
    _safe_write_json(FOREX_LOG_FILE, forex_trades)

    return signal_id


# ═══════════════════════════════════════════════════════════════════════════════
# ALERTS
# ═══════════════════════════════════════════════════════════════════════════════

def alert_fvg_signal(signal: Dict) -> None:
    """Send Telegram alert for FVG killzone signal."""
    pair_info = PAIRS[signal["pair"]]
    emoji = "🟢" if signal["direction"] == "buy" else "🔴"
    irl_line = ""
    if signal.get("irl_target"):
        irl_line = f"\nIRL (BE):    {signal['irl_target']:.5f}"

    msg = (
        f"{emoji} <b>FVG KILLZONE: {pair_info['name']} {signal['direction'].upper()}</b>\n"
        f"Model: ICT Continuation (PF 4.65)\n"
        f"Killzone: {signal['killzone']} Open\n"
        f"─────────────────────────\n"
        f"Entry:       {signal['entry']:.5f}\n"
        f"Stop Loss:   {signal['stop_loss']:.5f} ({signal['sl_pips']:.0f} pips)\n"
        f"Take Profit: {signal['take_profit']:.5f} ({signal['tp_pips']:.0f} pips)\n"
        f"R:R:         {signal['rr_ratio']:.1f}:1{irl_line}\n"
        f"─────────────────────────\n"
        f"Risk:        ${signal['risk_amount']:.2f}\n"
        f"Lot Size:    {signal['lot_size']:.3f}"
    )

    conflict = signal.get("conflict_resolution", "no_conflict")
    if conflict == "confluence":
        msg += f"\n\n✅ CONFLUENCE: Goldbach agrees — high conviction"
    elif conflict == "fvg_priority":
        msg += f"\n\n⚠️ Goldbach signal overridden — FVG momentum priority"

    try:
        subprocess.run(
            ["openclaw", "message", "send", "-t", "6515146575",
             "--channel", "telegram", "-m", msg],
            capture_output=True, timeout=10
        )
        print(f"  [alert] Telegram sent ✅")
    except Exception as e:
        print(f"  [alert] Failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def print_signal(signal: Dict) -> None:
    pair_info = PAIRS[signal["pair"]]
    emoji = "🟢" if signal["direction"] == "buy" else "🔴"
    print(f"\n  {'═'*55}")
    print(f"  {emoji} FVG KILLZONE: {pair_info['name']} {signal['direction'].upper()}")
    print(f"  {'═'*55}")
    print(f"  Entry:       {signal['entry']:.5f}")
    print(f"  Stop Loss:   {signal['stop_loss']:.5f}  ({signal['sl_pips']:.1f} pips)")
    print(f"  Take Profit: {signal['take_profit']:.5f}  ({signal['tp_pips']:.1f} pips)")
    print(f"  R:R:         {signal['rr_ratio']:.1f}:1")
    if signal.get("irl_target"):
        print(f"  IRL Target:  {signal['irl_target']:.5f}  (BE trigger)")
    print(f"  Killzone:    {signal['killzone']} Open")
    print(f"  Lot Size:    {signal['lot_size']:.3f}")
    print(f"  Risk:        ${signal['risk_amount']:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Forex FVG Killzone Scanner — ICT Continuation Model"
    )
    parser.add_argument("--pair", help="Single pair to scan (e.g., EURUSD)")
    parser.add_argument("--dry-run", action="store_true", help="No ledger writes or alerts")
    parser.add_argument("--force", action="store_true", help="Scan even outside killzone")
    args = parser.parse_args()

    now = datetime.now(EST)
    in_kz, kz_name = is_killzone_now()

    print(f"\n{'='*60}")
    print(f"  🎯 FOREX FVG KILLZONE SCANNER")
    print(f"  {now.strftime('%I:%M %p ET · %a %b %d, %Y')}")
    print(f"  Killzone: {'✅ ' + kz_name + ' Open' if in_kz else '❌ Outside killzone'}")
    print(f"{'='*60}")

    if not in_kz and not args.force:
        print(f"\n  ⏰ Not in killzone — scanner idle.")
        print(f"     London: 03:00–05:00 ET | NY: 08:00–12:00 ET | Late: 14:00–15:00 ET")
        print(f"     Use --force to scan anyway.")
        return

    pairs_to_scan = [args.pair.upper()] if args.pair else list(PAIRS.keys())
    all_signals = []

    for pair_key in pairs_to_scan:
        if pair_key not in PAIRS:
            print(f"  ❌ Unknown pair: {pair_key}")
            continue
        print(f"\n  📡 Scanning {PAIRS[pair_key]['name']}...")
        sigs = scan_pair_fvg(pair_key, dry_run=args.dry_run)
        all_signals.extend(sigs)

        if len(all_signals) >= MAX_TOTAL_SIGNALS:
            print(f"\n  ⚠️  Max total signals ({MAX_TOTAL_SIGNALS}) reached — stopping scan")
            break

    # Display and write signals
    for sig in all_signals:
        print_signal(sig)
        sig_id = write_fvg_to_ledger(sig, dry_run=args.dry_run)
        if sig_id and not args.dry_run:
            alert_fvg_signal(sig)

    # Summary
    print(f"\n{'='*60}")
    print(f"  📋 FVG SCAN SUMMARY")
    print(f"{'='*60}")
    print(f"  Pairs scanned:  {len(pairs_to_scan)}")
    print(f"  Signals found:  {len(all_signals)}")
    for sig in all_signals:
        emoji = "🟢" if sig["direction"] == "buy" else "🔴"
        print(f"  {emoji} {PAIRS[sig['pair']]['name']} {sig['direction'].upper()} | "
              f"RR {sig['rr_ratio']:.1f}:1 | {sig['killzone']}")
    if not all_signals:
        print(f"  📭 No FVG setups — no valid retests in killzone window")
    if args.dry_run:
        print(f"\n  ⚠️  DRY RUN — nothing written")
    print()


if __name__ == "__main__":
    main()
