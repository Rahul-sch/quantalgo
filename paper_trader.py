#!/usr/bin/env python3
"""
Live Paper Trading Signal Generator
Fetches latest market data, runs strategies, generates trade signals,
logs paper trades to JSON. Runs on-demand or can be called by a cron job.

Usage:
    python3 paper_trader.py                    # scan all symbols
    python3 paper_trader.py --symbol TSLA      # single symbol
    python3 paper_trader.py --show-trades      # show trade log
    python3 paper_trader.py --pnl              # show P&L summary
"""
import sys
import os
import json
import argparse
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data_fetcher import download_data, is_forex
from strategies import strategy_goldbach_bounce, strategy_goldbach_momentum
from continuation import strategy_continuation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TRADES_FILE = os.path.join(RESULTS_DIR, "paper_trades.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

INITIAL_CAPITAL = 10000.0
RISK_PCT = 0.01   # 1% risk per trade

# Best strategies from backtest + walk-forward
SCAN_CONFIG = [
    {"symbol": "TSLA",  "strategy": "goldbach_bounce",    "timeframe": "1d", "period": "5y"},
    {"symbol": "NVDA",  "strategy": "goldbach_momentum",  "timeframe": "1h", "period": "2y"},
    {"symbol": "SPY",   "strategy": "goldbach_bounce",    "timeframe": "1h", "period": "2y"},
    {"symbol": "TSLA",  "strategy": "continuation",       "timeframe": "1h", "period": "2y"},
    {"symbol": "SPY",   "strategy": "continuation",       "timeframe": "5m", "period": "60d"},
    {"symbol": "NVDA",  "strategy": "goldbach_bounce",    "timeframe": "5m", "period": "60d"},
]

STRATEGY_MAP = {
    "goldbach_bounce": strategy_goldbach_bounce,
    "goldbach_momentum": strategy_goldbach_momentum,
    "continuation": strategy_continuation,
}


# ─── Trade Log ─────────────────────────────────────────────────────────────────

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


# ─── Signal Generation ─────────────────────────────────────────────────────────

def get_latest_price(df: pd.DataFrame) -> float:
    return float(df["Close"].iloc[-1])


def get_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def signal_to_paper_trade(
    signal: Dict[str, Any],
    symbol: str,
    strategy: str,
    timeframe: str,
    df: pd.DataFrame,
    capital: float,
) -> Optional[Dict[str, Any]]:
    """Convert a strategy signal into a paper trade entry."""
    # Only use the most recent signal (last bar)
    if signal.get("bar", 0) < len(df) - 3:
        return None  # Signal is too old

    entry = signal["entry"]
    sl = signal["sl"]
    tp = signal["tp"]
    direction = signal["direction"]

    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return None

    risk_amount = capital * RISK_PCT
    pip_value = 0.01 if "JPY" in symbol else 0.0001 if is_forex(symbol) else 1.0

    if is_forex(symbol):
        position_size = risk_amount / (sl_dist / pip_value)
    else:
        position_size = risk_amount / sl_dist

    rr = abs(tp - entry) / sl_dist if sl_dist > 0 else 0
    current_price = get_latest_price(df)

    return {
        "id": f"{symbol}_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "strategy": strategy,
        "timeframe": timeframe,
        "direction": direction,
        "entry_price": round(entry, 5),
        "stop_loss": round(sl, 5),
        "take_profit": round(tp, 5),
        "current_price": round(current_price, 5),
        "sl_distance": round(sl_dist, 5),
        "rr_ratio": round(rr, 2),
        "position_size": round(position_size, 4),
        "risk_amount": round(risk_amount, 2),
        "potential_profit": round(risk_amount * rr, 2),
        "reason": signal.get("reason", ""),
        "status": "open",
        "exit_price": None,
        "actual_pnl": None,
    }


def scan_for_signals(
    symbols: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Scan all configured symbols/strategies for fresh signals."""
    new_signals = []
    config = SCAN_CONFIG
    if symbols:
        config = [c for c in SCAN_CONFIG if c["symbol"] in symbols]

    if verbose:
        print("\n" + "=" * 60)
        print("  LIVE PAPER TRADING SCANNER")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

    capital = INITIAL_CAPITAL  # Use fixed starting capital for signals

    for cfg in config:
        sym = cfg["symbol"]
        strat_name = cfg["strategy"]
        tf = cfg["timeframe"]
        period = cfg["period"]
        strat_fn = STRATEGY_MAP.get(strat_name)

        if strat_fn is None:
            continue

        df = download_data(sym, period=period, interval=tf)
        if df.empty or len(df) < 50:
            continue

        try:
            signals = strat_fn(df)
        except Exception as e:
            if verbose:
                print(f"  {sym} {strat_name}: error — {e}")
            continue

        # Find signals from the last 3 bars
        recent_signals = [s for s in signals if s.get("bar", 0) >= len(df) - 3]

        if recent_signals:
            latest = recent_signals[-1]
            trade = signal_to_paper_trade(latest, sym, strat_name, tf, df, capital)
            if trade:
                new_signals.append(trade)
                if verbose:
                    direction_emoji = "🟢" if trade["direction"] == "buy" else "🔴"
                    print(f"\n  {direction_emoji} SIGNAL: {sym} {strat_name} ({tf})")
                    print(f"     Direction: {trade['direction'].upper()}")
                    print(f"     Entry:     {trade['entry_price']}")
                    print(f"     Stop:      {trade['stop_loss']}")
                    print(f"     Target:    {trade['take_profit']}")
                    print(f"     RR:        {trade['rr_ratio']}:1")
                    print(f"     Risk:      ${trade['risk_amount']:.2f}")
                    print(f"     Reason:    {trade['reason']}")
        else:
            if verbose:
                print(f"  · {sym} {strat_name} ({tf}): no signal")

    if verbose:
        if new_signals:
            print(f"\n  📊 {len(new_signals)} new signal(s) logged to paper_trades.json")
        else:
            print(f"\n  📊 No new signals — market conditions not aligned")

    return new_signals


# ─── Trade Management ──────────────────────────────────────────────────────────

def update_open_trades(verbose: bool = True) -> List[Dict[str, Any]]:
    """Check open paper trades against current prices. Close if SL/TP hit."""
    trades = load_trades()
    open_trades = [t for t in trades if t.get("status") == "open"]
    closed_now = []

    if not open_trades:
        return []

    if verbose:
        print(f"\n  Checking {len(open_trades)} open trade(s)...")

    for trade in open_trades:
        sym = trade["symbol"]
        df = download_data(sym, period="5d", interval="1h")
        if df.empty:
            continue

        current_price = get_latest_price(df)
        trade["current_price"] = round(current_price, 5)

        entry = trade["entry_price"]
        sl = trade["stop_loss"]
        tp = trade["take_profit"]
        direction = trade["direction"]

        # Check if SL or TP hit
        if direction == "buy":
            if current_price <= sl:
                trade["status"] = "closed_sl"
                trade["exit_price"] = sl
                trade["actual_pnl"] = round(-trade["risk_amount"], 2)
            elif current_price >= tp:
                trade["status"] = "closed_tp"
                trade["exit_price"] = tp
                trade["actual_pnl"] = round(trade["risk_amount"] * trade["rr_ratio"], 2)
        else:
            if current_price >= sl:
                trade["status"] = "closed_sl"
                trade["exit_price"] = sl
                trade["actual_pnl"] = round(-trade["risk_amount"], 2)
            elif current_price <= tp:
                trade["status"] = "closed_tp"
                trade["exit_price"] = tp
                trade["actual_pnl"] = round(trade["risk_amount"] * trade["rr_ratio"], 2)

        if trade["status"] != "open":
            trade["closed_at"] = datetime.now().isoformat()
            closed_now.append(trade)
            if verbose:
                result = "✅ TP HIT" if "tp" in trade["status"] else "❌ SL HIT"
                print(f"  {result}: {sym} | P&L: ${trade['actual_pnl']:+.2f}")

    save_trades(trades)
    return closed_now


# ─── P&L Summary ───────────────────────────────────────────────────────────────

def show_pnl_summary() -> None:
    """Print full paper trading P&L summary."""
    trades = load_trades()
    closed = [t for t in trades if t.get("status") in ("closed_sl", "closed_tp")]
    open_t = [t for t in trades if t.get("status") == "open"]

    print("\n" + "=" * 60)
    print("  PAPER TRADING P&L SUMMARY")
    print("=" * 60)

    if not trades:
        print("  No trades logged yet. Run a scan first.")
        return

    total_pnl = sum(t.get("actual_pnl", 0) or 0 for t in closed)
    wins = [t for t in closed if (t.get("actual_pnl") or 0) > 0]
    losses = [t for t in closed if (t.get("actual_pnl") or 0) <= 0]
    win_rate = len(wins) / len(closed) * 100 if closed else 0

    print(f"\n  Total Trades:  {len(trades)} ({len(closed)} closed, {len(open_t)} open)")
    print(f"  Win Rate:      {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Total P&L:     ${total_pnl:+.2f}")
    print(f"  Account:       ${INITIAL_CAPITAL + total_pnl:,.2f}  (started ${INITIAL_CAPITAL:,.2f})")
    print(f"  Return:        {total_pnl / INITIAL_CAPITAL * 100:+.2f}%")

    if closed:
        avg_win = np.mean([t["actual_pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t["actual_pnl"]) for t in losses]) if losses else 0
        pf = sum(t["actual_pnl"] for t in wins) / sum(abs(t["actual_pnl"]) for t in losses) if losses else float("inf")
        print(f"\n  Avg Win:       ${avg_win:.2f}")
        print(f"  Avg Loss:      ${avg_loss:.2f}")
        print(f"  Profit Factor: {pf:.2f}")

    if open_t:
        print(f"\n  OPEN TRADES ({len(open_t)}):")
        for t in open_t:
            direction_emoji = "🟢" if t["direction"] == "buy" else "🔴"
            print(f"    {direction_emoji} {t['symbol']} {t['strategy']} | "
                  f"Entry: {t['entry_price']} | SL: {t['stop_loss']} | TP: {t['take_profit']}")

    if closed:
        print(f"\n  RECENT CLOSED TRADES:")
        for t in sorted(closed, key=lambda x: x.get("closed_at", ""), reverse=True)[:5]:
            result = "✅" if (t.get("actual_pnl") or 0) > 0 else "❌"
            print(f"    {result} {t['symbol']} {t['strategy']} ({t['timeframe']}) | "
                  f"P&L: ${t.get('actual_pnl', 0):+.2f}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Goldbach Paper Trading Signal Generator")
    parser.add_argument("--symbol", type=str, default=None, help="Scan specific symbol only")
    parser.add_argument("--show-trades", action="store_true", help="Show trade log")
    parser.add_argument("--pnl", action="store_true", help="Show P&L summary")
    parser.add_argument("--update", action="store_true", help="Update open trades vs current prices")
    args = parser.parse_args()

    if args.pnl or args.show_trades:
        show_pnl_summary()
        return

    if args.update:
        update_open_trades(verbose=True)
        show_pnl_summary()
        return

    # Scan for new signals
    symbols = [args.symbol] if args.symbol else None
    new_signals = scan_for_signals(symbols=symbols, verbose=True)

    # Log new signals
    for signal in new_signals:
        log_trade(signal)

    # Update existing open trades
    update_open_trades(verbose=True)

    # Always show summary
    show_pnl_summary()


if __name__ == "__main__":
    main()
