#!/usr/bin/env python3
"""
FOREX TRADE MANAGER — Lifecycle management for Goldbach Forex positions
Reads trade_state.json, checks fills/SL/TP against live prices, updates ledger.

Usage:
    python3 forex_manager.py          # run check (called by cron every 15 min)
    python3 forex_manager.py --status # print current open forex trades
    python3 forex_manager.py --dry-run # check fills without writing to ledger

Pairs supported: EURUSD, GBPUSD (extendable via FOREX_TICKERS)
"""

import os
import sys
import json
import fcntl
import argparse
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

EST       = ZoneInfo("US/Eastern")
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR       = os.path.join(BASE_DIR, "results")
STATE_LEDGER_FILE = os.path.join(RESULTS_DIR, "trade_state.json")
FOREX_LOG_FILE    = os.path.join(RESULTS_DIR, "forex_trades.json")
FOREX_CLOSED_FILE = os.path.join(RESULTS_DIR, "forex_closed.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# yfinance ticker mapping
FOREX_TICKERS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCHF": "CHF=X",
    "USDCAD": "CAD=X",
    "NZDUSD": "NZDUSD=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "EURGBP": "EURGBP=X",
}

PIP_SIZES = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "AUDUSD": 0.0001,
    "USDCHF": 0.0001,
    "USDCAD": 0.0001,
    "NZDUSD": 0.0001,
    "EURGBP": 0.0001,
    "USDJPY": 0.01,
    "EURJPY": 0.01,
    "GBPJPY": 0.01,
}

PIP_VALUES = {
    "EURUSD": 10.0,   # $10 per pip per standard lot
    "GBPUSD": 10.0,
    "AUDUSD": 10.0,
    "NZDUSD": 10.0,
    "USDCHF": 11.0,
    "USDCAD": 7.30,
    "EURGBP": 12.50,
    "USDJPY": 6.67,
    "EURJPY": 6.67,
    "GBPJPY": 6.67,
}

COMMISSION_PER_LOT = 0.0   # spread-only model for forex (no fixed commission)


# ═══════════════════════════════════════════════════════════════════════════════
# SAFE I/O (shared fcntl pattern with paper_trader.py)
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


def _today_str() -> str:
    return datetime.now(EST).strftime("%Y-%m-%d")


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE PRICE FETCH
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_live_price(pair: str) -> Optional[Dict[str, float]]:
    """
    Fetch live bid/ask/OHLC for a forex pair via yfinance.
    Returns dict with: price, high, low, open, prev_close
    Returns None on failure.
    """
    ticker_sym = FOREX_TICKERS.get(pair.upper())
    if not ticker_sym:
        print(f"  [price] Unknown pair: {pair}")
        return None
    try:
        import yfinance as yf
        ticker = yf.Ticker(ticker_sym)
        # Use 2-day 15m to get today's intraday OHLC
        hist = ticker.history(period="2d", interval="15m")
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = [col[0] for col in hist.columns]
        if hist.empty:
            return None

        last = hist.iloc[-1]
        today_bars = hist[hist.index.date == hist.index[-1].date()]

        return {
            "price":      float(last["Close"]),
            "high":       float(today_bars["High"].max()),
            "low":        float(today_bars["Low"].min()),
            "open":       float(today_bars["Open"].iloc[0]),
            "prev_close": float(hist.iloc[-2]["Close"]) if len(hist) > 1 else float(last["Close"]),
            "timestamp":  hist.index[-1].isoformat(),
        }
    except Exception as e:
        print(f"  [price] Failed to fetch {pair}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════

def _is_forex_trade(order: Dict[str, Any]) -> bool:
    """Return True if this ledger entry is a Forex trade (not an equity trade)."""
    symbol = order.get("signal_id", "") + order.get("trade_id", "")
    pair = order.get("trade_id", "").split("_")[0].upper()
    return pair in FOREX_TICKERS or order.get("model", "").startswith("goldbach_forex")


def _pips(price_diff: float, pair: str) -> float:
    pip = PIP_SIZES.get(pair.upper(), 0.0001)
    return abs(price_diff) / pip


def _gross_pnl(entry: float, exit_price: float, direction: str,
               lot_size: float, pair: str) -> float:
    pip = PIP_SIZES.get(pair.upper(), 0.0001)
    pip_val = PIP_VALUES.get(pair.upper(), 10.0)
    if direction == "buy":
        pips = (exit_price - entry) / pip
    else:
        pips = (entry - exit_price) / pip
    return round(pips * lot_size * pip_val, 2)


def validate_order_levels(order: Dict[str, Any]) -> Optional[str]:
    """
    Sanity-check SL/TP direction vs entry for the given trade side.
    Returns an error string if levels are inverted, None if valid.

    BUY:  sl < entry < tp
    SELL: tp < entry < sl
    """
    direction = order["direction"]
    entry = float(order["entry_price"])
    sl    = float(order["stop_loss"])
    tp    = float(order["take_profit"])

    if direction == "buy":
        if sl >= entry:
            return (f"BUY has SL ({sl:.5f}) >= entry ({entry:.5f}) — "
                    f"stale/inverted level, skipping to avoid ghost trigger")
        if tp <= entry:
            return (f"BUY has TP ({tp:.5f}) <= entry ({entry:.5f}) — "
                    f"inverted level, skipping")
    elif direction == "sell":
        if sl <= entry:
            return (f"SELL has SL ({sl:.5f}) <= entry ({entry:.5f}) — "
                    f"stale/inverted level, skipping to avoid ghost trigger")
        if tp >= entry:
            return (f"SELL has TP ({tp:.5f}) >= entry ({entry:.5f}) — "
                    f"inverted level, skipping")
    return None


def check_order_fills(
    order: Dict[str, Any],
    quote: Dict[str, float],
    dry_run: bool = False,
) -> Tuple[str, Optional[float]]:
    """
    Check if a pending/open forex order has been filled, stopped out, or TP'd.

    Returns: (new_status, exit_price)
      new_status: "pending" | "open" | "closed_tp" | "closed_sl" | "expired"
    """
    direction   = order["direction"]
    entry       = float(order["entry_price"])
    sl          = float(order["stop_loss"])
    tp          = float(order["take_profit"])
    status      = order.get("status", "pending")
    current     = quote["price"]
    bar_high    = quote["high"]
    bar_low     = quote["low"]

    # ── Check expiry ──
    try:
        expires = pd.to_datetime(order.get("expires_after", "2099-01-01"), utc=True)
        now_utc = pd.Timestamp.now(tz="UTC")
        if now_utc > expires:
            return "expired", None
    except Exception:
        pass

    # ── PENDING → check if entry filled ──
    if status == "pending":
        if direction == "buy":
            filled = bar_low <= entry <= bar_high or current <= entry
        else:
            filled = bar_low <= entry <= bar_high or current >= entry

        if not filled:
            return "pending", None
        # Filled — now check if SL/TP also hit same bar
        status = "open"
        # Fall through to open logic

    # ── OPEN → check SL and TP ──
    if status == "open":
        if direction == "buy":
            tp_hit = bar_high >= tp or current >= tp
            sl_hit = bar_low <= sl or current <= sl
        else:
            tp_hit = bar_low <= tp or current <= tp
            sl_hit = bar_high >= sl or current >= sl

        # Priority: if both hit same bar, assume SL (conservative / worst-case)
        if sl_hit and tp_hit:
            return "closed_sl", sl
        if tp_hit:
            return "closed_tp", tp
        if sl_hit:
            return "closed_sl", sl

        return "open", None

    return status, None


# ═══════════════════════════════════════════════════════════════════════════════
# ALERT
# ═══════════════════════════════════════════════════════════════════════════════

def _send_alert(msg: str) -> None:
    """Send Telegram alert via openclaw CLI."""
    try:
        subprocess.run(
            ["openclaw", "message", "send", "-t", "6515146575",
             "--channel", "telegram", "-m", msg],
            capture_output=True, timeout=10
        )
        print(f"  [alert] Telegram sent ✅")
    except Exception as e:
        print(f"  [alert] Failed: {e}")


def alert_filled(order: Dict[str, Any], quote: Dict[str, float]) -> None:
    pair = order.get("trade_id", "???").split("_")[0]
    direction = order["direction"].upper()
    entry = float(order["entry_price"])
    sl = float(order["stop_loss"])
    tp = float(order["take_profit"])
    sl_pips = round(_pips(entry - sl, pair), 1)
    tp_pips = round(_pips(tp - entry, pair), 1)
    emoji = "🟢" if direction == "BUY" else "🔴"
    _send_alert(
        f"{emoji} FOREX FILLED: {pair} {direction}\n"
        f"Entry:  {entry:.5f}\n"
        f"SL:     {sl:.5f}  ({sl_pips:.0f} pips)\n"
        f"TP:     {tp:.5f}  ({tp_pips:.0f} pips)\n"
        f"Current: {quote['price']:.5f}"
    )


FAT_TAIL_REMINDER = (
    "\n[System Note: Fat-tail regime active. "
    "6-loss streaks are within normal backtest parameters. "
    "Do not intervene.]"
)


def alert_closed(order: Dict[str, Any], exit_price: float,
                 new_status: str, gross_pnl: float) -> None:
    pair = order.get("trade_id", "???").split("_")[0]
    direction = order["direction"].upper()
    entry = float(order["entry_price"])
    result = "WIN ✅" if new_status == "closed_tp" else "LOSS ❌"
    pnl_sign = "+" if gross_pnl >= 0 else ""
    pnl_pips = round(_pips(abs(exit_price - entry), pair), 1)
    emoji = "💚" if new_status == "closed_tp" else "🔴"
    msg = (
        f"{emoji} FOREX CLOSED: {pair} {direction} — {result}\n"
        f"Entry:  {entry:.5f}\n"
        f"Exit:   {exit_price:.5f}  ({pnl_pips:.0f} pips)\n"
        f"P&L:    {pnl_sign}${gross_pnl:.2f}"
    )
    # Append fat-tail reminder on every stop loss
    if new_status == "closed_sl":
        msg += FAT_TAIL_REMINDER
    _send_alert(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN MANAGER LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_manager(dry_run: bool = False) -> None:
    now = datetime.now(EST)
    print(f"\n  {'═'*50}")
    print(f"  📊 FOREX MANAGER — {now.strftime('%I:%M %p ET · %a %b %d')}")
    print(f"  {'═'*50}")

    # Load ledger
    ledger = _safe_read_json(STATE_LEDGER_FILE, {})
    today = _today_str()
    if ledger.get("date") != today:
        print(f"  ℹ️  Ledger is from {ledger.get('date','?')} — no open trades today")
        return

    # Find forex orders
    all_orders = ledger.get("pending_orders", [])
    forex_orders = [o for o in all_orders
                    if _is_forex_trade(o) and o.get("status") in ("pending", "open")]

    if not forex_orders:
        print(f"  📭 No active forex orders in ledger")
        return

    print(f"  Found {len(forex_orders)} active forex order(s)")

    # Fetch live prices (cache per pair to avoid redundant calls)
    price_cache: Dict[str, Optional[Dict]] = {}
    pairs_needed = set()
    for o in forex_orders:
        pair = o.get("trade_id", "").split("_")[0].upper()
        if pair in FOREX_TICKERS:
            pairs_needed.add(pair)

    for pair in pairs_needed:
        print(f"  📡 Fetching {pair} live price...")
        price_cache[pair] = fetch_live_price(pair)
        if price_cache[pair]:
            q = price_cache[pair]
            print(f"     {pair}: {q['price']:.5f} "
                  f"(H: {q['high']:.5f} / L: {q['low']:.5f})")
        else:
            print(f"     {pair}: ❌ Price unavailable")

    # Process each order
    closed_log = _safe_read_json(FOREX_CLOSED_FILE, [])
    changed = False

    for order in all_orders:
        if not _is_forex_trade(order):
            continue
        if order.get("status") not in ("pending", "open"):
            continue

        pair = order.get("trade_id", "").split("_")[0].upper()
        quote = price_cache.get(pair)
        old_status = order["status"]

        if not quote:
            print(f"\n  ⚠️  {pair}: No price data — leaving order as {old_status}")
            continue

        # ── Direction sanity check — reject ghost/stale levels ──
        level_err = validate_order_levels(order)
        if level_err:
            print(f"\n  ⚠️  {pair} LEVEL SANITY FAIL: {level_err}")
            print(f"     Skipping this order — mark as 'invalid_levels' in ledger")
            if not dry_run:
                order["status"] = "invalid_levels"
                order["invalid_reason"] = level_err
                order["invalid_timestamp"] = now.isoformat()
                changed = True
            continue

        new_status, exit_price = check_order_fills(order, quote, dry_run=dry_run)

        print(f"\n  {pair} {order['direction'].upper()} @ {order['entry_price']}")
        print(f"     Was: {old_status} → Now: {new_status}")
        print(f"     Current price: {quote['price']:.5f}")

        if new_status == old_status and new_status == "pending":
            entry = float(order["entry_price"])
            pips_away = _pips(abs(quote['price'] - entry), pair)
            print(f"     Entry {entry:.5f} not yet hit ({pips_away:.1f} pips away)")
            continue

        if dry_run:
            print(f"     [dry-run] Would update status to: {new_status}")
            if exit_price:
                gross = _gross_pnl(float(order["entry_price"]), exit_price,
                                   order["direction"], float(order.get("lot_size", 0.1)), pair)
                print(f"     [dry-run] P&L would be: ${gross:+.2f}")
            continue

        # Apply state changes
        changed = True
        order["status"] = new_status

        if new_status == "open" and old_status == "pending":
            order["fill_price"] = float(order["entry_price"])
            order["fill_timestamp"] = now.isoformat()
            print(f"     ✅ FILLED at {order['entry_price']}")
            alert_filled(order, quote)

        elif new_status in ("closed_tp", "closed_sl", "expired"):
            actual_exit = exit_price or quote["price"]
            order["exit_price"] = actual_exit
            order["exit_timestamp"] = now.isoformat()
            lot_size = float(order.get("lot_size", 0.1))
            gross = _gross_pnl(float(order["entry_price"]), actual_exit,
                                order["direction"], lot_size, pair)
            order["gross_pnl"] = gross
            order["net_pnl"]   = gross  # no fixed commission on forex (spread model)

            if new_status == "closed_tp":
                pips = round(_pips(abs(actual_exit - float(order["entry_price"])), pair), 1)
                print(f"     🏆 TP HIT @ {actual_exit:.5f} (+{pips:.0f} pips | ${gross:+.2f})")
                alert_closed(order, actual_exit, new_status, gross)
            elif new_status == "closed_sl":
                pips = round(_pips(abs(actual_exit - float(order["entry_price"])), pair), 1)
                print(f"     🛑 SL HIT @ {actual_exit:.5f} (-{pips:.0f} pips | ${gross:+.2f})")
                alert_closed(order, actual_exit, new_status, gross)
            elif new_status == "expired":
                print(f"     ⏰ EXPIRED — entry never filled")

            # Archive to closed log
            closed_log.append(dict(order))

    # Save updated ledger
    if changed:
        _safe_write_json(STATE_LEDGER_FILE, ledger)
        _safe_write_json(FOREX_CLOSED_FILE, closed_log)
        print(f"\n  ✅ Ledger updated")
    else:
        print(f"\n  ℹ️  No changes to ledger")


# ═══════════════════════════════════════════════════════════════════════════════
# STATUS DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def print_status() -> None:
    """Print a summary of all open forex trades."""
    now = datetime.now(EST)
    ledger = _safe_read_json(STATE_LEDGER_FILE, {})
    closed = _safe_read_json(FOREX_CLOSED_FILE, [])

    all_orders = ledger.get("pending_orders", [])
    forex_open   = [o for o in all_orders if _is_forex_trade(o) and o.get("status") in ("pending", "open")]
    forex_closed_today = [o for o in all_orders if _is_forex_trade(o) and o.get("status") not in ("pending", "open")]
    all_closed = closed  # historical

    print(f"\n  {'═'*50}")
    print(f"  📊 FOREX MANAGER STATUS — {now.strftime('%I:%M %p ET')}")
    print(f"  {'═'*50}")
    print(f"  Open/Pending:   {len(forex_open)}")
    print(f"  Closed today:   {len(forex_closed_today)}")
    print(f"  Total closed:   {len(all_closed)}")

    if forex_open:
        print(f"\n  ACTIVE ORDERS:")
        for o in forex_open:
            pair = o.get("trade_id", "?").split("_")[0]
            d = o["direction"].upper()
            status = o.get("status", "?")
            entry = float(o["entry_price"])
            sl = float(o["stop_loss"])
            tp = float(o["take_profit"])
            sl_pips = round(_pips(abs(entry - sl), pair), 0)
            tp_pips = round(_pips(abs(tp - entry), pair), 0)
            emoji = "🟢" if d == "BUY" else "🔴"
            print(f"  {emoji} [{status}] {pair} {d} @ {entry:.5f} | "
                  f"SL {sl_pips:.0f}p | TP {tp_pips:.0f}p")

    if all_closed:
        wins  = [t for t in all_closed if t.get("status") == "closed_tp"]
        losses = [t for t in all_closed if t.get("status") == "closed_sl"]
        total_pnl = sum(t.get("net_pnl", 0) or 0 for t in all_closed)
        wr = len(wins) / len(all_closed) * 100 if all_closed else 0
        sign = "+" if total_pnl >= 0 else ""
        print(f"\n  HISTORICAL:")
        print(f"  Win Rate:  {wr:.0f}%  ({len(wins)}W / {len(losses)}L)")
        print(f"  Total P&L: {sign}${total_pnl:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Forex Trade Manager — lifecycle management for Goldbach positions"
    )
    parser.add_argument("--status",  action="store_true", help="Print open trade status")
    parser.add_argument("--dry-run", action="store_true", help="Check fills without writing")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    run_manager(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
