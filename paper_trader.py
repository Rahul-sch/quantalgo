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
import argparse
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

# Ensure QQQ is available
if "QQQ" not in INSTRUMENTS:
    INSTRUMENTS["QQQ"] = "QQQ"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — LOCKED LIVE-READY PARAMS
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TRADES_FILE = os.path.join(RESULTS_DIR, "paper_trades.json")
CSV_LOG_FILE = os.path.join(RESULTS_DIR, "paper_trades_log.csv")
DAILY_STATE_FILE = os.path.join(RESULTS_DIR, "paper_daily_state.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Capital & risk
INITIAL_CAPITAL = 10_000.0
RISK_PCT = 0.01  # 1% risk per trade

# Daily loss circuit breaker
MAX_DAILY_LOSS = 150.0  # halt trading if daily net P&L drops below -$150

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


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE LOG (JSON)
# ═══════════════════════════════════════════════════════════════════════════════

def load_trades() -> List[Dict[str, Any]]:
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE) as f:
            return json.load(f)
    return []


def save_trades(trades: List[Dict[str, Any]]) -> None:
    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2, default=str)


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
    if os.path.exists(DAILY_STATE_FILE):
        with open(DAILY_STATE_FILE) as f:
            state = json.load(f)
        # Reset if it's a new day
        if state.get("date") != _today_str():
            return {"date": _today_str(), "daily_pnl": 0.0, "halted": False, "trades_today": 0}
        return state
    return {"date": _today_str(), "daily_pnl": 0.0, "halted": False, "trades_today": 0}


def save_daily_state(state: Dict[str, Any]) -> None:
    with open(DAILY_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


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
    save_daily_state(state)


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
            # Status
            "status": "open",
            "exit_price": None,
            "exit_timestamp": None,
            "gross_pnl": None,
            "net_pnl": None,
            "commission": round(commission, 2),
            "slippage": None,
        }

        new_trades.append(trade)

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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Institutional Paper Trading Engine — QQQ 15m Continuation"
    )
    parser.add_argument("--pnl", action="store_true", help="Show P&L summary")
    parser.add_argument("--show-trades", action="store_true", help="Show full trade log")
    parser.add_argument("--update", action="store_true", help="Update open trades vs current prices")
    parser.add_argument("--reset", action="store_true", help="Clear all paper trades")
    args = parser.parse_args()

    if args.reset:
        for f in [TRADES_FILE, CSV_LOG_FILE, DAILY_STATE_FILE]:
            if os.path.exists(f):
                os.remove(f)
        print("  🗑️  All paper trades cleared.")
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

    # Scan for new signals
    new_signals = scan_for_signals(verbose=True)

    # Log new signals
    for signal in new_signals:
        log_trade(signal)
        append_csv_row(signal)

    # Update existing open trades
    update_open_trades(verbose=True)

    # Show summary
    show_pnl_summary()

    # Print CSV location
    if os.path.exists(CSV_LOG_FILE):
        print(f"\n  📄 Detailed CSV log: {CSV_LOG_FILE}")


if __name__ == "__main__":
    main()
