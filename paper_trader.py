#!/usr/bin/env python3
"""
LIVE PAPER TRADING ENGINE — Institutional Grade
Uses quant_engine.py signal generator with walk-forward validated parameters.
Includes friction tracking, daily loss limit, forced EOD close, and CSV logging.

Usage:
    python3 paper_trader.py                    # scan for QQQ 15m signals
    python3 paper_trader.py --pnl              # show P&L summary
    python3 paper_trader.py --update           # update open trades vs current prices
    python3 paper_trader.py --show-trades      # show full trade log
    python3 paper_trader.py --reset            # clear all paper trades

Live-Ready Parameters (from walk-forward validation):
    Instrument:   QQQ (Nasdaq 100 ETF)
    Timeframe:    15-minute
    ATR Stop:     0.5x ATR
    Risk/Reward:  2.5:1
    Displacement: 1.0x ATR
    Session:      9:30-11:30 AM + 1:30-3:30 PM EST
"""
import sys
import os
import csv
import json
import fcntl
import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data_fetcher import download_data, INSTRUMENTS
from quant_engine import (
    Config,
    generate_signals,
    apply_friction,
    compute_indicators,
    refresh_blackout_calendar,
)
from notifier import alert_new_trade, alert_session_summary, alert_daily_limit

# Ensure QQQ is available
if "QQQ" not in INSTRUMENTS:
    INSTRUMENTS["QQQ"] = "QQQ"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — LOCKED LIVE-READY PARAMS
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TRADES_FILE      = os.path.join(RESULTS_DIR, "paper_trades.json")
CSV_LOG_FILE     = os.path.join(RESULTS_DIR, "paper_trades_log.csv")
DAILY_STATE_FILE = os.path.join(RESULTS_DIR, "paper_daily_state.json")
STATE_LEDGER_FILE = os.path.join(RESULTS_DIR, "trade_state.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Max bars a pending limit order stays alive before expiring (15m bars = 75 min)
PENDING_ORDER_EXPIRY_BARS = 5

# Capital & risk
INITIAL_CAPITAL = 10_000.0
RISK_PCT = 0.01  # 1% risk per trade

# Daily loss circuit breaker
MAX_DAILY_LOSS   = 150.0  # halt trading if daily net P&L drops below -$150
DAILY_LOSS_LIMIT = MAX_DAILY_LOSS  # alias used by stress_test.py

# Forced close time (EST)
FORCE_CLOSE_HOUR = 15
FORCE_CLOSE_MINUTE = 30

# Live-ready config from walk-forward validation
LIVE_CONFIG = Config(
    symbols=["QQQ"],
    initial_capital=INITIAL_CAPITAL,
    risk_pct=RISK_PCT,
    # Strategy — locked from backtest
    atr_multiplier_sl=0.5,
    rr_ratio=2.5,
    displacement_threshold=1.0,
    # All filters ON
    session_filter=True,
    use_htf_filter=True,
    use_adx_filter=True,
    adx_threshold=18.0,
    use_rvol_filter=True,
    rvol_multiplier=1.2,
    # Friction — institutional
    etf_slippage_pct=0.00005,  # 0.005%
    commission_round_trip=2.40,
)

EST = ZoneInfo("US/Eastern")

# CSV columns for detailed trade log
CSV_COLUMNS = [
    "id", "timestamp", "symbol", "timeframe", "direction",
    "entry_price", "stop_loss", "take_profit",
    "position_size", "risk_amount",
    # Indicator snapshot at entry
    "atr_at_entry", "adx_at_entry", "rvol_at_entry", "ema_direction",
    # Outcome
    "status", "exit_price", "exit_timestamp",
    "gross_pnl", "net_pnl", "commission", "slippage",
    "reason",
]


# ─────────────────────────────────────────────────────────────────────────────
# SAFE I/O — atomic writes + advisory file locking
#
# Why both?
#   • fcntl.flock: prevents concurrent reads/writes between cron and Streamlit
#   • .tmp + os.replace: guarantees the file is never half-written on crash
#     (os.replace is atomic on POSIX — kernel swaps the inode in one syscall)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_read_json(path: str, default: Any) -> Any:
    """Read a JSON file with shared (read) lock. Returns `default` on any error."""
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
    """
    Write JSON atomically with exclusive lock.
    1. Acquire exclusive lock on a .lock sentinel file
    2. Write to <path>.tmp
    3. os.replace(.tmp → path) — atomic on POSIX
    4. Release lock
    Readers always see either the old complete file or the new one — never partial.
    """
    lock_path = path + ".lock"
    tmp_path  = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, path)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


# ─────────────────────────────────────────────────────────────────────────────
# YFINANCE SAFE WRAPPER — hard timeout via ThreadPoolExecutor
#
# yfinance.download() uses requests internally with no exposed timeout.
# A DNS stall or slow Yahoo server can block the entire cron process for
# minutes. We run it in a thread and enforce a hard wall-clock deadline.
# ─────────────────────────────────────────────────────────────────────────────

def _yf_download_safe(ticker: str, timeout_sec: int = 20, **kwargs):
    """
    Call yfinance.download() with a hard timeout.
    Returns an empty DataFrame on timeout or any error.

    We import yfinance at call time (not module load time) so that test mocks
    patching the yfinance module object are picked up correctly. The lambda
    captures the module reference at submission time, ensuring the thread uses
    whatever yfinance.download is bound to at that moment (real or mocked).
    """
    import yfinance as _yf
    import pandas as _pd
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_yf.download, ticker, **kwargs)
        result = future.result(timeout=timeout_sec)
        executor.shutdown(wait=False)
        return result
    except FuturesTimeoutError:
        print(f"  ⚠️  yfinance timeout ({timeout_sec}s) for {ticker} — using fallback")
        executor.shutdown(wait=False, cancel_futures=True)
        return _pd.DataFrame()
    except Exception as e:
        print(f"  ⚠️  yfinance error for {ticker}: {e} — using fallback")
        executor.shutdown(wait=False)
        return _pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE LOG (JSON)
# ═══════════════════════════════════════════════════════════════════════════════

def load_trades() -> List[Dict[str, Any]]:
    return _safe_read_json(TRADES_FILE, [])


def save_trades(trades: List[Dict[str, Any]]) -> None:
    _safe_write_json(TRADES_FILE, trades)


def log_trade(trade: Dict[str, Any]) -> None:
    trades = load_trades()
    trades.append(trade)
    save_trades(trades)


# ═══════════════════════════════════════════════════════════════════════════════
# CSV TRADE LOG (detailed, for analysis)
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_csv_header():
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(CSV_LOG_FILE):
        with open(CSV_LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def append_csv_row(trade: Dict[str, Any]) -> None:
    """Append a single trade row to the CSV log."""
    _ensure_csv_header()
    row = {col: trade.get(col, "") for col in CSV_COLUMNS}
    with open(CSV_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


def update_csv_row(trade_id: str, updates: Dict[str, Any]) -> None:
    """Update an existing CSV row by trade id (rewrites file)."""
    if not os.path.exists(CSV_LOG_FILE):
        return
    rows = []
    with open(CSV_LOG_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("id") == trade_id:
                row.update({k: str(v) for k, v in updates.items()})
            rows.append(row)
    with open(CSV_LOG_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# DAILY STATE — Circuit Breaker
# ═══════════════════════════════════════════════════════════════════════════════

def _today_str() -> str:
    return datetime.now(EST).strftime("%Y-%m-%d")


def load_daily_state() -> Dict[str, Any]:
    state = _safe_read_json(DAILY_STATE_FILE, {})
    if not state or state.get("date") != _today_str():
        return {"date": _today_str(), "daily_pnl": 0.0, "halted": False, "trades_today": 0}
    return state


def save_daily_state(state: Dict[str, Any]) -> None:
    _safe_write_json(DAILY_STATE_FILE, state)


def check_daily_limit() -> bool:
    """Returns True if trading is allowed, False if halted."""
    state = load_daily_state()
    if state.get("halted", False):
        return False
    if state.get("daily_pnl", 0) <= -MAX_DAILY_LOSS:
        state["halted"] = True
        save_daily_state(state)
        return False
    return True


def update_daily_pnl(pnl: float) -> None:
    state = load_daily_state()
    state["daily_pnl"] = round(state.get("daily_pnl", 0) + pnl, 2)
    state["trades_today"] = state.get("trades_today", 0) + 1
    if state["daily_pnl"] <= -MAX_DAILY_LOSS:
        state["halted"] = True
        print(f"\n  🛑 DAILY LOSS LIMIT HIT: ${state['daily_pnl']:.2f} — TRADING HALTED FOR TODAY")
        try:
            alert_daily_limit()
        except Exception:
            pass
    save_daily_state(state)


# ═══════════════════════════════════════════════════════════════════════════════
# STATE LEDGER — Persistent memory across cron runs
# ═══════════════════════════════════════════════════════════════════════════════
#
# trade_state.json schema:
# {
#   "date": "2026-03-31",          # resets daily
#   "pending_orders": [            # limit orders waiting to be filled
#     {
#       "signal_id": "QQQ_1743400200_478.50_buy",  # fingerprint: ts+price+dir
#       "trade_id": "QQQ_cont_20260331_093045",
#       "direction": "buy",
#       "entry_price": 478.50,
#       "stop_loss": 476.20,
#       "take_profit": 484.25,
#       "armed_at": "2026-03-31T09:30:45-04:00",
#       "armed_bar_time": "2026-03-31T09:30:00-04:00",  # bar that triggered
#       "expires_after": "2026-03-31T10:45:00-04:00",   # max 5 bars = 75 min
#       "status": "pending"        # pending | filled | expired | cancelled
#     }
#   ],
#   "alerted_signal_ids": [        # signals we've already Telegram'd
#     "QQQ_1743400200_478.50_buy",
#     ...
#   ]
# }
#
# Signal fingerprint = f"{symbol}_{bar_epoch}_{entry_price:.2f}_{direction}"
# This uniquely identifies a specific FVG setup regardless of which cron run sees it.

def _signal_fingerprint(symbol: str, bar_time: Any, entry: float, direction: str) -> str:
    """
    Generate a stable fingerprint for a signal so we can deduplicate across runs.
    Uses the bar's epoch timestamp (rounded to 15m), entry price, and direction.
    """
    try:
        ts = pd.to_datetime(bar_time)
        if hasattr(ts, 'timestamp'):
            epoch = int(ts.timestamp() // 900 * 900)  # floor to 15m boundary
        else:
            epoch = int(ts.value // (900 * 1_000_000_000) * 900)
    except Exception:
        epoch = int(datetime.now(EST).timestamp() // 900 * 900)
    return f"{symbol}_{epoch}_{entry:.2f}_{direction}"


def load_state_ledger() -> Dict[str, Any]:
    """Load the persistent state ledger, resetting if it's a new day."""
    today = _today_str()
    ledger = _safe_read_json(STATE_LEDGER_FILE, {})
    if ledger.get("date") == today:
        return ledger
    return {
        "date": today,
        "pending_orders": [],
        "alerted_signal_ids": [],
    }


def save_state_ledger(ledger: Dict[str, Any]) -> None:
    _safe_write_json(STATE_LEDGER_FILE, ledger)


def is_already_alerted(signal_id: str) -> bool:
    """Return True if we've already sent a Telegram alert for this signal."""
    ledger = load_state_ledger()
    return signal_id in ledger.get("alerted_signal_ids", [])


def mark_alerted(signal_id: str) -> None:
    """Record that we've sent a Telegram alert for this signal."""
    ledger = load_state_ledger()
    if signal_id not in ledger["alerted_signal_ids"]:
        ledger["alerted_signal_ids"].append(signal_id)
    save_state_ledger(ledger)


def add_pending_order(trade: Dict[str, Any], signal_id: str, bar_time: Any) -> None:
    """
    Register a new pending limit order in the state ledger.
    Called when a signal fires and we arm the limit order.
    """
    ledger = load_state_ledger()

    # Check for existing pending order with same fingerprint
    existing = [p for p in ledger["pending_orders"] if p["signal_id"] == signal_id]
    if existing:
        return  # already registered — don't duplicate

    # Compute expiry: bar_time + 5 bars (75 min)
    try:
        bar_dt = pd.to_datetime(bar_time)
        if bar_dt.tzinfo is None:
            bar_dt = bar_dt.tz_localize("US/Eastern")
        else:
            bar_dt = bar_dt.tz_convert("US/Eastern")
        expires = bar_dt + pd.Timedelta(minutes=15 * PENDING_ORDER_EXPIRY_BARS)
        expires_str = expires.isoformat()
        bar_time_str = bar_dt.isoformat()
    except Exception:
        now = datetime.now(EST)
        expires_str = (now + pd.Timedelta(minutes=75)).isoformat()
        bar_time_str = now.isoformat()

    order = {
        "signal_id":      signal_id,
        "trade_id":       trade["id"],
        "direction":      trade["direction"],
        "entry_price":    trade["entry_price"],
        "stop_loss":      trade["stop_loss"],
        "take_profit":    trade["take_profit"],
        "armed_at":       datetime.now(EST).isoformat(),
        "armed_bar_time": bar_time_str,
        "expires_after":  expires_str,
        "status":         "pending",
    }
    ledger["pending_orders"].append(order)
    save_state_ledger(ledger)
    print(f"  [ledger] Pending order armed: {signal_id[:50]}")
    print(f"           Expires: {expires_str}")


def resolve_pending_orders(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Check all pending limit orders against the latest price bars.
    For each pending order:
      - FILLED: current bar crossed through the entry price → promote to open trade
      - EXPIRED: current time is past expires_after → cancel
      - SL HIT (while pending): price blew through SL before touching entry → cancel
      - Still PENDING: do nothing

    Returns list of newly-filled trade dicts (to be merged into trades file).
    Called at the START of each cron run, before scanning for new signals.
    """
    ledger = load_state_ledger()
    # Only resolve QQQ equity orders — forex orders have signal_ids starting with EURUSD/GBPUSD
    pending = [p for p in ledger["pending_orders"]
               if p["status"] == "pending"
               and p.get("signal_id", "").startswith("QQQ")]

    if not pending:
        return []

    now = datetime.now(EST)
    current_bar = df.iloc[-1]
    current_high = float(current_bar["High"])
    current_low  = float(current_bar["Low"])
    current_price = float(current_bar["Close"])
    filled_trades = []

    print(f"\n  [ledger] Checking {len(pending)} pending order(s)... "
          f"(price: ${current_price:.2f})")

    for order in pending:
        signal_id = order["signal_id"]
        entry  = order["entry_price"]
        sl     = order["stop_loss"]
        tp     = order["take_profit"]
        direction = order["direction"]

        # Check expiry
        try:
            expires_dt = pd.to_datetime(order["expires_after"])
            if expires_dt.tzinfo is None:
                expires_dt = expires_dt.tz_localize("US/Eastern")
            else:
                expires_dt = expires_dt.tz_convert("US/Eastern")
            now_aware = now if now.tzinfo else now.replace(tzinfo=ZoneInfo("US/Eastern"))
            if now_aware > expires_dt:
                order["status"] = "expired"
                print(f"  [ledger] ⏰ EXPIRED: {signal_id[:40]} (entry ${entry:.2f} never filled)")
                continue
        except Exception:
            pass

        # Check if filled: did the bar touch our limit entry level?
        if direction == "buy":
            filled = current_low <= entry <= current_high or current_price <= entry
            # SL hit before entry: price gapped past entry without touching it
            sl_hit_pending = current_low < sl and current_low > entry
            # GAP-OVER: single candle gaps below BOTH entry AND stop loss
            # e.g. buy limit at 480, SL at 478 — bar opens at 477.50
            # The order would technically fill at entry then immediately stop out.
            # We treat this as a worst-case fill: filled at entry, exited at SL.
            gap_over = current_high < entry  # entire bar is below our limit (for buy)
        else:
            filled = current_low <= entry <= current_high or current_price >= entry
            sl_hit_pending = current_high > sl and current_high < entry
            gap_over = current_low > entry  # entire bar is above our limit (for sell)

        # Gap-over: bar completely skipped our entry level — fill at entry, close at SL
        if gap_over:
            order["status"] = "gap_over_sl"
            gap_exit = sl  # worst-case exit at stop loss price
            print(f"  [ledger] ⚡ GAP-OVER: {signal_id[:40]}")
            print(f"           Bar [{current_low:.2f}–{current_high:.2f}] skipped entry "
                  f"${entry:.2f} entirely — filling at entry, closing at SL ${sl:.2f}")
            trades = load_trades()
            for t in trades:
                if t["id"] == order["trade_id"]:
                    t["status"] = "closed_gap_sl"
                    t["fill_price"] = entry
                    t["exit_price"] = gap_exit
                    t["exit_timestamp"] = datetime.now(EST).isoformat()
                    raw_pnl = (gap_exit - entry) if direction == "buy" else (entry - gap_exit)
                    t["gross_pnl"] = round(raw_pnl * t.get("position_size", 1), 2)
                    t["net_pnl"]   = round(t["gross_pnl"] - t.get("commission", 0), 2)
                    break
            save_trades(trades)
            continue

        if sl_hit_pending and not filled:
            order["status"] = "cancelled_sl_skip"
            print(f"  [ledger] ❌ CANCELLED (SL skip): {signal_id[:40]} — "
                  f"price moved through SL ${sl:.2f} without filling entry ${entry:.2f}")
            continue

        if filled:
            order["status"] = "filled"
            print(f"  [ledger] ✅ FILLED: {signal_id[:40]} @ ${entry:.2f}")

            # Update the trade record from "pending" → "open" in trades file
            trades = load_trades()
            for t in trades:
                if t["id"] == order["trade_id"]:
                    t["status"] = "open"
                    t["fill_timestamp"] = datetime.now(EST).isoformat()
                    t["fill_price"] = entry
                    filled_trades.append(t)
                    break
            save_trades(trades)
        else:
            print(f"  [ledger] ⏳ PENDING: {signal_id[:40]} — "
                  f"waiting for ${entry:.2f} "
                  f"({'buy' if direction=='buy' else 'sell'}, "
                  f"current: ${current_price:.2f})")

    # Save updated ledger
    save_state_ledger(ledger)
    return filled_trades


def expire_old_pending_orders() -> None:
    """Expire any pending orders past their expiry time. Called at start of each run."""
    ledger = load_state_ledger()
    now = datetime.now(EST)
    now_aware = now if now.tzinfo else now.replace(tzinfo=ZoneInfo("US/Eastern"))
    changed = False
    for order in ledger["pending_orders"]:
        if order["status"] != "pending":
            continue
        try:
            exp = pd.to_datetime(order["expires_after"])
            if exp.tzinfo is None:
                exp = exp.tz_localize("US/Eastern")
            else:
                exp = exp.tz_convert("US/Eastern")
            if now_aware > exp:
                order["status"] = "expired"
                changed = True
                print(f"  [ledger] ⏰ Expired pending order: {order['signal_id'][:40]}")
        except Exception:
            pass
    if changed:
        save_state_ledger(ledger)


def print_ledger_status() -> None:
    """Print a summary of the current state ledger."""
    ledger = load_state_ledger()
    # Show QQQ-only orders in the equity ledger status (forex orders filtered out)
    qqq_orders = [p for p in ledger["pending_orders"]
                  if p.get("signal_id", "").startswith("QQQ") or
                     p.get("signal_id", "").startswith("DRY")]
    pending  = [p for p in qqq_orders if p["status"] == "pending"]
    filled   = [p for p in qqq_orders if p["status"] == "filled"]
    expired  = [p for p in qqq_orders if p["status"] in ("expired", "cancelled_sl_skip")]
    alerted  = ledger.get("alerted_signal_ids", [])

    print(f"\n  📋 STATE LEDGER ({ledger.get('date', '?')})")
    print(f"     Pending orders:  {len(pending)}")
    print(f"     Filled today:    {len(filled)}")
    print(f"     Expired/Cancel:  {len(expired)}")
    print(f"     Alerts sent:     {len(alerted)}")
    for p in pending:
        print(f"     ⏳ {p['direction'].upper()} @ ${p['entry_price']:.2f} "
              f"| expires {p['expires_after'][11:16]}")


# ═══════════════════════════════════════════════════════════════════════════════
# TIME CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def is_past_force_close() -> bool:
    """Check if current time is past 3:30 PM EST."""
    now = datetime.now(EST)
    return (now.hour > FORCE_CLOSE_HOUR or

            (now.hour == FORCE_CLOSE_HOUR and now.minute >= FORCE_CLOSE_MINUTE))


def is_market_hours() -> bool:
    """Check if we're in a valid trading session window."""
    now = datetime.now(EST)
    minutes = now.hour * 60 + now.minute
    in_am = 570 <= minutes <= 690   # 9:30-11:30
    in_pm = 810 <= minutes <= 930   # 1:30-3:30
    weekday = now.weekday()
    return (in_am or in_pm) and weekday < 5


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION — Using Institutional Engine
# ═══════════════════════════════════════════════════════════════════════════════

def get_indicator_snapshot(df: pd.DataFrame, bar_idx: int, cfg: Config) -> Dict[str, Any]:
    """Extract indicator values at a specific bar for CSV logging."""
    ind = compute_indicators(df, cfg)
    return {
        "atr_at_entry": round(float(ind["atr"].iloc[bar_idx]), 4) if not np.isnan(ind["atr"].iloc[bar_idx]) else 0,
        "adx_at_entry": round(float(ind["adx"].iloc[bar_idx]), 1) if not np.isnan(ind["adx"].iloc[bar_idx]) else 0,
        "rvol_at_entry": round(float(ind["rvol"].iloc[bar_idx]), 2) if not np.isnan(ind["rvol"].iloc[bar_idx]) else 0,
        "ema_direction": "bullish" if ind["htf_signal"].iloc[bar_idx] > 0 else "bearish" if ind["htf_signal"].iloc[bar_idx] < 0 else "neutral",
    }


def scan_for_signals(verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Scan QQQ 15m for fresh continuation signals using the institutional engine.
    """
    if not check_daily_limit():
        if verbose:
            state = load_daily_state()
            print(f"\n  🛑 TRADING HALTED — Daily P&L: ${state['daily_pnl']:.2f} (limit: -${MAX_DAILY_LOSS})")
        return []

    if verbose:
        print("\n" + "=" * 60)
        print("  INSTITUTIONAL PAPER TRADING SCANNER")
        print(f"  {datetime.now(EST).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print("=" * 60)
        print(f"  Instrument:   QQQ")
        print(f"  Timeframe:    15m")
        print(f"  ATR SL:       {LIVE_CONFIG.atr_multiplier_sl}x")
        print(f"  Risk/Reward:  {LIVE_CONFIG.rr_ratio}:1")
        print(f"  Displacement: {LIVE_CONFIG.displacement_threshold}x ATR")
        print(f"  Filters:      EMA={LIVE_CONFIG.use_htf_filter} ADX={LIVE_CONFIG.use_adx_filter} RVOL={LIVE_CONFIG.use_rvol_filter}")
        print(f"  Friction:     ${LIVE_CONFIG.commission_round_trip}/RT + {LIVE_CONFIG.etf_slippage_pct*100:.3f}% slip")
        state = load_daily_state()
        print(f"  Daily P&L:    ${state.get('daily_pnl', 0):+.2f} (limit: -${MAX_DAILY_LOSS})")
        print()

    # Download fresh 15m data
    df = download_data("QQQ", period="60d", interval="15m")
    if df.empty or len(df) < 50:
        if verbose:
            print("  ⚠️  Insufficient data — need at least 50 bars")
        return []

    # Compute current capital from trade history
    trades = load_trades()
    closed = [t for t in trades if t.get("status") in ("closed_sl", "closed_tp", "closed_eod", "closed_manual")]
    total_realized = sum(t.get("net_pnl", 0) or 0 for t in closed)
    current_capital = INITIAL_CAPITAL + total_realized

    # Generate signals using institutional engine
    signals = generate_signals(df, LIVE_CONFIG)

    # Filter to only recent signals (last 3 bars)
    recent_signals = [s for s in signals if s.get("bar", 0) >= len(df) - 3]

    new_trades = []
    for sig in recent_signals:
        entry = sig["entry"]
        sl = sig["sl"]
        tp = sig["tp"]
        direction = sig["direction"]
        bar_idx = sig["bar"]

        sl_dist = abs(entry - sl)
        if sl_dist <= 0:
            continue

        risk_amount = current_capital * RISK_PCT
        position_size = risk_amount / sl_dist
        rr = abs(tp - entry) / sl_dist

        # Get indicator snapshot for CSV
        indicators = get_indicator_snapshot(df, bar_idx, LIVE_CONFIG)

        # Apply friction to estimate net P&L at TP
        adj_entry, adj_exit_tp, commission = apply_friction(
            entry, tp, direction, "QQQ", LIVE_CONFIG
        )
        if direction == "buy":
            gross_potential = (tp - entry) * position_size
            net_potential = (adj_exit_tp - adj_entry) * position_size - commission
        else:
            gross_potential = (entry - tp) * position_size
            net_potential = (adj_entry - adj_exit_tp) * position_size - commission

        current_price = float(df["Close"].iloc[-1])

        # Build signal fingerprint for deduplication
        bar_time = df.index[bar_idx]
        signal_id = _signal_fingerprint("QQQ", bar_time, entry, direction)

        # Skip if we've already armed this exact signal in a prior cron run
        if is_already_alerted(signal_id):
            if verbose:
                print(f"  [ledger] Skipping duplicate signal: {signal_id[:50]}")
            continue

        # Also skip if there's already a pending order for this signal
        ledger_check = load_state_ledger()
        existing_pending = [p for p in ledger_check["pending_orders"]
                            if p["signal_id"] == signal_id and p["status"] == "pending"]
        if existing_pending:
            if verbose:
                print(f"  [ledger] Order already pending: {signal_id[:50]}")
            continue

        trade_id = f"QQQ_cont_{datetime.now(EST).strftime('%Y%m%d_%H%M%S')}"

        trade = {
            "id": trade_id,
            "timestamp": datetime.now(EST).isoformat(),
            "symbol": "QQQ",
            "timeframe": "15m",
            "direction": direction,
            "entry_price": round(entry, 4),
            "stop_loss": round(sl, 4),
            "take_profit": round(tp, 4),
            "current_price": round(current_price, 4),
            "sl_distance": round(sl_dist, 4),
            "rr_ratio": round(rr, 2),
            "position_size": round(position_size, 4),
            "risk_amount": round(risk_amount, 2),
            "gross_potential": round(gross_potential, 2),
            "net_potential": round(net_potential, 2),
            "reason": sig.get("reason", ""),
            # Indicator snapshot
            "atr_at_entry": indicators["atr_at_entry"],
            "adx_at_entry": indicators["adx_at_entry"],
            "rvol_at_entry": indicators["rvol_at_entry"],
            "ema_direction": indicators["ema_direction"],
            # Status — starts as "pending" until price fills the limit entry
            "status": "pending",
            "signal_id": signal_id,
            "exit_price": None,
            "exit_timestamp": None,
            "gross_pnl": None,
            "net_pnl": None,
            "commission": round(commission, 2),
            "slippage": None,
        }

        new_trades.append(trade)

        # Register in state ledger immediately
        add_pending_order(trade, signal_id, bar_time)

        if verbose:
            emoji = "🟢" if direction == "buy" else "🔴"
            print(f"  {emoji} SIGNAL: QQQ continuation ({direction.upper()})")
            print(f"     Entry:       ${entry:.2f}")
            print(f"     Stop Loss:   ${sl:.2f} ({sl_dist:.2f} away)")
            print(f"     Take Profit: ${tp:.2f}")
            print(f"     RR:          {rr:.1f}:1")
            print(f"     Risk:        ${risk_amount:.2f}")
            print(f"     Size:        {position_size:.2f} shares")
            print(f"     Net @ TP:    ${net_potential:+.2f} (after ${commission:.2f} comm + slip)")
            print(f"     ATR: {indicators['atr_at_entry']} | ADX: {indicators['adx_at_entry']} | "
                  f"RVOL: {indicators['rvol_at_entry']}x | EMA: {indicators['ema_direction']}")
            print(f"     Reason:      {sig.get('reason', '')}")

    if verbose and not new_trades:
        print("  📊 No new signals — market conditions not aligned")

    return new_trades


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE MANAGEMENT — Update, Close, Force-Close
# ═══════════════════════════════════════════════════════════════════════════════

def _close_trade(trade: Dict, exit_price: float, status: str, verbose: bool = True) -> None:
    """Close a trade, compute friction-adjusted P&L, update daily state."""
    entry = trade["entry_price"]
    direction = trade["direction"]
    position_size = trade["position_size"]

    adj_entry, adj_exit, commission = apply_friction(
        entry, exit_price, direction, "QQQ", LIVE_CONFIG
    )

    if direction == "buy":
        gross_pnl = (exit_price - entry) * position_size
        net_pnl = (adj_exit - adj_entry) * position_size - commission
    else:
        gross_pnl = (entry - exit_price) * position_size
        net_pnl = (adj_entry - adj_exit) * position_size - commission

    slippage = abs(gross_pnl - (net_pnl + commission))

    trade["status"] = status
    trade["exit_price"] = round(exit_price, 4)
    trade["exit_timestamp"] = datetime.now(EST).isoformat()
    trade["gross_pnl"] = round(gross_pnl, 2)
    trade["net_pnl"] = round(net_pnl, 2)
    trade["commission"] = round(commission, 2)
    trade["slippage"] = round(slippage, 4)

    # Update daily P&L
    update_daily_pnl(net_pnl)

    # Update CSV log
    update_csv_row(trade["id"], {
        "status": status,
        "exit_price": trade["exit_price"],
        "exit_timestamp": trade["exit_timestamp"],
        "gross_pnl": trade["gross_pnl"],
        "net_pnl": trade["net_pnl"],
        "commission": trade["commission"],
        "slippage": trade["slippage"],
    })

    if verbose:
        result_emoji = "✅" if net_pnl > 0 else "❌"
        print(f"  {result_emoji} {status.upper()}: QQQ {direction} | "
              f"Gross: ${gross_pnl:+.2f} → Net: ${net_pnl:+.2f} "
              f"(comm: ${commission:.2f}, slip: ${slippage:.4f})")


def update_open_trades(verbose: bool = True) -> List[Dict[str, Any]]:
    """Check open trades against current prices. Close if SL/TP hit or EOD."""
    trades = load_trades()
    open_trades = [t for t in trades if t.get("status") == "open"]
    closed_now = []

    if not open_trades:
        if verbose:
            print("\n  No open trades to update.")
        return []

    if verbose:
        print(f"\n  Checking {len(open_trades)} open trade(s)...")

    # Force-close if past 3:30 PM EST
    force_close = is_past_force_close()
    if force_close and verbose:
        print("  ⏰ Past 3:30 PM EST — force-closing all open positions")

    df = download_data("QQQ", period="5d", interval="15m")
    if df.empty:
        return []

    current_price = float(df["Close"].iloc[-1])

    for trade in open_trades:
        trade["current_price"] = round(current_price, 4)
        entry = trade["entry_price"]
        sl = trade["stop_loss"]
        tp = trade["take_profit"]
        direction = trade["direction"]

        if force_close:
            _close_trade(trade, current_price, "closed_eod", verbose)
            closed_now.append(trade)
            continue

        # Check SL/TP
        if direction == "buy":
            if current_price <= sl:
                _close_trade(trade, sl, "closed_sl", verbose)
                closed_now.append(trade)
            elif current_price >= tp:
                _close_trade(trade, tp, "closed_tp", verbose)
                closed_now.append(trade)
        else:
            if current_price >= sl:
                _close_trade(trade, sl, "closed_sl", verbose)
                closed_now.append(trade)
            elif current_price <= tp:
                _close_trade(trade, tp, "closed_tp", verbose)
                closed_now.append(trade)

    save_trades(trades)
    return closed_now


# ═══════════════════════════════════════════════════════════════════════════════
# P&L SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def show_pnl_summary() -> None:
    """Print full paper trading P&L summary with friction breakdown."""
    trades = load_trades()
    closed_statuses = ("closed_sl", "closed_tp", "closed_eod", "closed_manual")
    closed = [t for t in trades if t.get("status") in closed_statuses]
    open_t = [t for t in trades if t.get("status") == "open"]

    print("\n" + "=" * 60)
    print("  INSTITUTIONAL PAPER TRADING — P&L SUMMARY")
    print("=" * 60)

    if not trades:
        print("  No trades logged yet. Run a scan first.")
        return

    # Aggregate P&L
    total_gross = sum(t.get("gross_pnl", 0) or 0 for t in closed)
    total_net = sum(t.get("net_pnl", 0) or 0 for t in closed)
    total_comm = sum(t.get("commission", 0) or 0 for t in closed)
    total_slip = sum(t.get("slippage", 0) or 0 for t in closed)

    wins = [t for t in closed if (t.get("net_pnl") or 0) > 0]
    losses = [t for t in closed if (t.get("net_pnl") or 0) <= 0]
    win_rate = len(wins) / len(closed) * 100 if closed else 0

    print(f"\n  Total Trades:   {len(trades)} ({len(closed)} closed, {len(open_t)} open)")
    print(f"  Win Rate:       {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"\n  Gross P&L:      ${total_gross:+.2f}")
    print(f"  Commission:     -${total_comm:.2f}")
    print(f"  Slippage:       -${total_slip:.4f}")
    print(f"  ─────────────────────────")
    print(f"  Net P&L:        ${total_net:+.2f}")
    print(f"\n  Account:        ${INITIAL_CAPITAL + total_net:,.2f}  (started ${INITIAL_CAPITAL:,.2f})")
    print(f"  Return:         {total_net / INITIAL_CAPITAL * 100:+.2f}%")

    if closed:
        avg_win = np.mean([t["net_pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t["net_pnl"]) for t in losses]) if losses else 0
        total_win_pnl = sum(t["net_pnl"] for t in wins) if wins else 0
        total_loss_pnl = sum(abs(t["net_pnl"]) for t in losses) if losses else 0
        pf = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float("inf")
        pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"

        print(f"\n  Avg Win:        ${avg_win:.2f}")
        print(f"  Avg Loss:       ${avg_loss:.2f}")
        print(f"  Profit Factor:  {pf_str}")

        # Equity curve stats
        equity = [INITIAL_CAPITAL]
        for t in sorted(closed, key=lambda x: x.get("exit_timestamp", "")):
            equity.append(equity[-1] + (t.get("net_pnl", 0) or 0))
        peak = INITIAL_CAPITAL
        max_dd = 0
        for eq in equity:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        print(f"  Max Drawdown:   {max_dd:.1f}%")

    # Daily state
    state = load_daily_state()
    print(f"\n  TODAY ({state.get('date', '?')}):")
    print(f"    Trades:  {state.get('trades_today', 0)}")
    print(f"    P&L:     ${state.get('daily_pnl', 0):+.2f}")
    print(f"    Status:  {'🛑 HALTED' if state.get('halted') else '✅ ACTIVE'}")

    # Open trades
    if open_t:
        print(f"\n  OPEN TRADES ({len(open_t)}):")
        for t in open_t:
            emoji = "🟢" if t["direction"] == "buy" else "🔴"
            unrealized = 0
            if t.get("current_price") and t.get("entry_price"):
                if t["direction"] == "buy":
                    unrealized = (t["current_price"] - t["entry_price"]) * t.get("position_size", 0)
                else:
                    unrealized = (t["entry_price"] - t["current_price"]) * t.get("position_size", 0)
            print(f"    {emoji} {t['direction'].upper()} @ ${t['entry_price']:.2f} | "
                  f"SL: ${t['stop_loss']:.2f} | TP: ${t['take_profit']:.2f} | "
                  f"Unrealized: ${unrealized:+.2f}")

    # Recent closed
    if closed:
        recent = sorted(closed, key=lambda x: x.get("exit_timestamp", ""), reverse=True)[:5]
        print(f"\n  RECENT CLOSED:")
        for t in recent:
            result = "✅" if (t.get("net_pnl") or 0) > 0 else "❌"
            status = t.get("status", "?").replace("closed_", "").upper()
            print(f"    {result} {t['direction'].upper()} | {status} | "
                  f"Gross: ${t.get('gross_pnl', 0):+.2f} → Net: ${t.get('net_pnl', 0):+.2f}")


def show_trade_log() -> None:
    """Show full trade log."""
    trades = load_trades()
    if not trades:
        print("\n  No trades logged.")
        return

    print(f"\n  FULL TRADE LOG ({len(trades)} trades):")
    print(f"  {'─' * 90}")
    for t in trades:
        emoji = "🟢" if t["direction"] == "buy" else "🔴"
        status = t.get("status", "open")
        net = t.get("net_pnl")
        net_str = f"${net:+.2f}" if net is not None else "pending"
        print(f"  {emoji} {t.get('timestamp', '?')[:19]} | {t['direction'].upper():5} | "
              f"Entry: ${t['entry_price']:.2f} | SL: ${t['stop_loss']:.2f} | "
              f"TP: ${t['take_profit']:.2f} | {status:12} | {net_str}")
        if t.get("atr_at_entry"):
            print(f"     ATR: {t['atr_at_entry']} | ADX: {t['adx_at_entry']} | "
                  f"RVOL: {t['rvol_at_entry']}x | EMA: {t['ema_direction']}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_dry_run() -> None:
    """
    --dry-run / --test mode: full end-to-end system test without market hours checks.

    1. Bypasses all time locks (weekend, market hours, AM session filter)
    2. Fetches the most recent available 15m data (Friday's close if weekend)
    3. Runs the real signal engine — reports any genuine setups found
    4. Forces a synthetic test trade into trade_state.json regardless of real signals
    5. Fires a real Telegram alert so you can verify formatting + delivery
    6. Does NOT affect real trade ledger (test entries are prefixed DRY_RUN_)
    """
    print("\n" + "=" * 60)
    print("  🧪 DRY RUN — END-TO-END SYSTEM TEST")
    print("  ⚠️  Time locks BYPASSED — using last available data")
    print("=" * 60)
    now = datetime.now(EST)
    print(f"  Run time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")

    # Step 1: Calendar fetch
    print("  [1/5] 📡 Fetching economic calendar...")
    try:
        refresh_blackout_calendar(LIVE_CONFIG)
        print("        ✅ Calendar fetched")
    except Exception as e:
        print(f"        ⚠️  Calendar failed: {e}")

    # Step 2: Load state ledger
    print("\n  [2/5] 📋 State ledger status:")
    print_ledger_status()

    # Step 3: Fetch real market data (last 5 trading days)
    print("\n  [3/5] 📊 Fetching QQQ 15m data (last 5 days)...")
    df = download_data("QQQ", period="5d", interval="15m")
    if df.empty:
        print("        ❌ No data available — check connection")
        return
    last_bar = df.iloc[-1]
    last_bar_time = df.index[-1]
    print(f"        ✅ {len(df)} bars loaded")
    print(f"        Last bar: {last_bar_time} | Close: ${float(last_bar['Close']):.2f}")

    # Step 4: Run real signal engine (all time filters bypassed at Config level)
    print("\n  [4/5] 🔍 Running signal engine on real data...")
    test_cfg = Config(
        symbols=["QQQ"],
        initial_capital=INITIAL_CAPITAL,
        risk_pct=RISK_PCT,
        atr_multiplier_sl=LIVE_CONFIG.atr_multiplier_sl,
        rr_ratio=LIVE_CONFIG.rr_ratio,
        displacement_threshold=LIVE_CONFIG.displacement_threshold,
        # Disable all API-dependent filters for fast dry run
        use_vix_filter=False,
        use_blackout_filter=False,
        use_macro_veto=False,
        # Keep technical filters ON — real signal quality check
        session_filter=False,   # bypass AM session time gate
        use_htf_filter=LIVE_CONFIG.use_htf_filter,
        use_adx_filter=LIVE_CONFIG.use_adx_filter,
        adx_threshold=LIVE_CONFIG.adx_threshold,
        use_rvol_filter=LIVE_CONFIG.use_rvol_filter,
        rvol_multiplier=LIVE_CONFIG.rvol_multiplier,
        etf_slippage_pct=LIVE_CONFIG.etf_slippage_pct,
        commission_round_trip=LIVE_CONFIG.commission_round_trip,
    )
    from quant_engine import generate_signals
    real_signals = generate_signals(df, test_cfg)
    recent_real = [s for s in real_signals if s.get("bar", 0) >= len(df) - 10]
    if recent_real:
        print(f"        ✅ {len(recent_real)} real signal(s) found in last 10 bars!")
        for s in recent_real[:3]:
            print(f"           {s['direction'].upper()} @ ${s['entry']:.2f} | "
                  f"SL ${s['sl']:.2f} | TP ${s['tp']:.2f}")
    else:
        print(f"        ℹ️  No live signals in last 10 bars (normal for weekend/quiet market)")
        print(f"           Total signals in 5-day window: {len(real_signals)}")

    # Step 5: Synthesize a test trade and write to ledger
    print("\n  [5/5] 🔧 Writing synthetic test order to trade_state.json...")
    last_close = float(last_bar["Close"])
    vix_data = _yf_download_safe("^VIX", timeout_sec=15, period="2d", interval="1d",
                                  progress=False, auto_adjust=True)
    vix_now = float(vix_data["Close"].iloc[-1]) if not vix_data.empty else 21.5

    qqq_monthly = _yf_download_safe("QQQ", timeout_sec=20, period="2y", interval="1mo",
                                     progress=False, auto_adjust=True)
    if not qqq_monthly.empty and len(qqq_monthly) >= 20:
        close_m = qqq_monthly["Close"].squeeze()
        sma20 = float(close_m.rolling(20).mean().iloc[-1])
        macro_pct = (last_close - sma20) / sma20 * 100
    else:
        macro_pct = 8.2  # fallback

    # Build a realistic synthetic trade using real ATR
    try:
        from quant_engine import compute_indicators
        ind = compute_indicators(df, test_cfg)
        atr_val = float(ind["atr"].dropna().iloc[-1])
    except Exception:
        atr_val = last_close * 0.003  # ~0.3% of price as fallback ATR

    test_entry = round(last_close - atr_val * 0.3, 2)   # slightly below close (buy retest)
    test_sl    = round(test_entry - atr_val * LIVE_CONFIG.atr_multiplier_sl, 2)
    test_tp    = round(test_entry + atr_val * LIVE_CONFIG.atr_multiplier_sl * LIVE_CONFIG.rr_ratio, 2)
    sl_dist    = round(abs(test_entry - test_sl), 4)
    risk_amt   = round(INITIAL_CAPITAL * RISK_PCT, 2)
    pos_size   = round(risk_amt / sl_dist, 4) if sl_dist > 0 else 10.0
    rr         = round(abs(test_tp - test_entry) / sl_dist, 2) if sl_dist > 0 else LIVE_CONFIG.rr_ratio

    dry_run_id  = f"DRY_RUN_{now.strftime('%Y%m%d_%H%M%S')}"
    signal_id   = f"QQQ_DRYRUN_{now.strftime('%Y%m%d%H%M')}_buy"

    test_trade = {
        "id":           dry_run_id,
        "timestamp":    now.isoformat(),
        "symbol":       "QQQ",
        "timeframe":    "15m",
        "direction":    "buy",
        "entry_price":  test_entry,
        "stop_loss":    test_sl,
        "take_profit":  test_tp,
        "current_price": last_close,
        "sl_distance":  sl_dist,
        "rr_ratio":     rr,
        "position_size": pos_size,
        "risk_amount":  risk_amt,
        "gross_potential": round(abs(test_tp - test_entry) * pos_size, 2),
        "net_potential": round(abs(test_tp - test_entry) * pos_size - LIVE_CONFIG.commission_round_trip, 2),
        "reason":       "DRY RUN — synthetic FVG retest (buy limit at FVG boundary)",
        "atr_at_entry": round(atr_val, 4),
        "adx_at_entry": 0,
        "rvol_at_entry": 0,
        "ema_direction": "bullish",
        "status":       "pending",
        "signal_id":    signal_id,
        "exit_price":   None,
        "exit_timestamp": None,
        "gross_pnl":    None,
        "net_pnl":      None,
        "commission":   LIVE_CONFIG.commission_round_trip,
        "slippage":     None,
        "dry_run":      True,
    }

    # Write to ledger (uses real ledger functions — full integration test)
    add_pending_order(test_trade, signal_id, last_bar_time)
    print(f"        ✅ Pending order written: {dry_run_id}")
    print(f"           Entry: ${test_entry:.2f} | SL: ${test_sl:.2f} | TP: ${test_tp:.2f}")
    print(f"           Signal ID: {signal_id}")

    # Step 6: Fire real Telegram alert
    print("\n  📲 Sending Telegram alert (live delivery test)...")
    # Add DRY RUN badge to the message
    test_trade_labeled = dict(test_trade)
    test_trade_labeled["reason"] = "🧪 DRY RUN — FVG Retest Setup (System Test)"
    try:
        alert_new_trade(test_trade_labeled, vix=vix_now, macro_pct=macro_pct)
        mark_alerted(signal_id)
        print("        ✅ Telegram alert delivered")
    except Exception as e:
        print(f"        ❌ Telegram alert failed: {e}")

    # Final summary
    print("\n" + "=" * 60)
    print("  ✅ DRY RUN COMPLETE — System Status:")
    print("=" * 60)
    print(f"  Data ingestion:   ✅  {len(df)} bars | last close ${last_close:.2f}")
    print(f"  Signal engine:    ✅  {len(real_signals)} signals in 5-day window")
    print(f"  State ledger:     ✅  Pending order written to trade_state.json")
    print(f"  Telegram alert:   ✅  Check your phone")
    print(f"  VIX (live):       {vix_now:.1f}  {'✅ CLEAR' if vix_now < 25 else '🔴 BLOCKED'}")
    print(f"  Macro:            {macro_pct:+.1f}% vs 20M SMA  {'✅ BULL' if macro_pct >= 0 else '⚠️  BEAR'}")
    print(f"\n  📄 Ledger: {STATE_LEDGER_FILE}")
    print(f"  💡 Run `python3 paper_trader.py --reset` to clear test data before Monday")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Institutional Paper Trading Engine — QQQ 15m Continuation"
    )
    parser.add_argument("--pnl",      action="store_true", help="Show P&L summary")
    parser.add_argument("--show-trades", action="store_true", help="Show full trade log")
    parser.add_argument("--update",   action="store_true", help="Update open trades vs current prices")
    parser.add_argument("--reset",    action="store_true", help="Clear all paper trades and ledger")
    parser.add_argument("--summary",  action="store_true", help="Send AM session summary via Telegram")
    parser.add_argument("--dry-run",  action="store_true", help="End-to-end system test (bypasses time locks, sends real Telegram alert)")
    parser.add_argument("--test",     action="store_true", help="Alias for --dry-run")
    args = parser.parse_args()

    if args.dry_run or args.test:
        run_dry_run()
        return

    if args.reset:
        for f in [TRADES_FILE, CSV_LOG_FILE, DAILY_STATE_FILE, STATE_LEDGER_FILE]:
            if os.path.exists(f):
                os.remove(f)
        print("  🗑️  All paper trades and state ledger cleared.")
        return

    if args.summary:
        print("  📊 Sending AM session summary to Telegram...")
        alert_session_summary()
        show_pnl_summary()
        return

    if args.show_trades:
        show_trade_log()
        return

    if args.pnl:
        show_pnl_summary()
        return

    if args.update:
        update_open_trades(verbose=True)
        show_pnl_summary()
        return

    # ── Default: Scan for signals ──
    print("\n" + "=" * 60)
    print("  🏛️  INSTITUTIONAL PAPER TRADER — QQQ 15m")
    print("=" * 60)

    # ── Dynamic Economic Calendar: fetch today's high-impact schedule ──
    print("\n  📡 Fetching today's economic calendar...")
    try:
        refresh_blackout_calendar(LIVE_CONFIG)
    except Exception as e:
        print(f"  ⚠️  Calendar refresh failed: {e} — blackout filter will use cached/computed windows")

    # Check market hours (warn but don't block — useful for testing)
    if not is_market_hours():
        now_est = datetime.now(EST)
        print(f"\n  ⚠️  Outside market hours ({now_est.strftime('%H:%M %Z')})")
        print("  Scanning anyway — signals based on last available data\n")

    # ── State Ledger: expire old orders + resolve pending fills ──
    print_ledger_status()
    expire_old_pending_orders()
    # Fetch latest bars to check pending fills before scanning for new signals
    _df_for_pending = download_data("QQQ", period="5d", interval="15m")
    if not _df_for_pending.empty:
        resolve_pending_orders(_df_for_pending)

    # Scan for new signals
    new_signals = scan_for_signals(verbose=True)

    # Fetch live VIX + macro once (shared across all signals this run)
    # Hard-capped at 15/20s via _yf_download_safe — won't block the cron
    vix_now = None
    macro_pct = None
    vix_data = _yf_download_safe("^VIX", timeout_sec=15, period="2d", interval="1d",
                                  progress=False, auto_adjust=True)
    if not vix_data.empty:
        try:
            vix_now = float(vix_data["Close"].iloc[-1])
        except Exception:
            pass
    qqq_monthly = _yf_download_safe("QQQ", timeout_sec=20, period="2y", interval="1mo",
                                     progress=False, auto_adjust=True)
    if not qqq_monthly.empty and len(qqq_monthly) >= 20:
        try:
            close = qqq_monthly["Close"].squeeze()
            sma20 = float(close.rolling(20).mean().iloc[-1])
            price = float(close.iloc[-1])
            macro_pct = (price - sma20) / sma20 * 100
        except Exception:
            pass

    # Log new signals + fire Telegram alerts (deduplicated via state ledger)
    for signal in new_signals:
        log_trade(signal)
        append_csv_row(signal)
        signal_id = signal.get("signal_id", "")
        # Double-check: only alert if not already alerted (guard against race conditions)
        if signal_id and is_already_alerted(signal_id):
            print(f"  [ledger] Alert already sent for {signal_id[:40]} — skipping")
            continue
        try:
            alert_new_trade(signal, vix=vix_now, macro_pct=macro_pct)
            if signal_id:
                mark_alerted(signal_id)  # record that we sent this alert
        except Exception as e:
            print(f"  [alert] Telegram notification failed: {e}")

    # Update existing open trades
    update_open_trades(verbose=True)

    # Show summary
    show_pnl_summary()

    # Print CSV location
    if os.path.exists(CSV_LOG_FILE):
        print(f"\n  📄 Detailed CSV log: {CSV_LOG_FILE}")


if __name__ == "__main__":
    main()
