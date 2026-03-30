#!/usr/bin/env python3
"""
GOLDBACH FOREX BACKTESTER — PO3/PO9 Discount/Premium Strategy
Historical validation over max available 15m data (~60 days via yfinance).

No lookahead: dealing range computed from CLOSED previous NY session only.
Entry logic mirrors goldbach_forex.py exactly.

Usage:
    python3 forex_backtester.py                  # EUR/USD default
    python3 forex_backtester.py --pair GBPUSD    # GBP/USD
    python3 forex_backtester.py --pair EURUSD --min-rr 2.0  # stricter RR filter
    python3 forex_backtester.py --pair EURUSD --max-sl 30   # tighter SL cap
"""

import argparse
import sys
from datetime import datetime, timedelta, date, time as dtime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

EST = ZoneInfo("US/Eastern")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — mirrors goldbach_forex.py exactly
# ═══════════════════════════════════════════════════════════════════════════════

PAIRS = {
    "EURUSD": {"ticker": "EURUSD=X", "pip_size": 0.0001, "pip_value": 10.0, "name": "EUR/USD"},
    "GBPUSD": {"ticker": "GBPUSD=X", "pip_size": 0.0001, "pip_value": 10.0, "name": "GBP/USD"},
    "AUDUSD": {"ticker": "AUDUSD=X", "pip_size": 0.0001, "pip_value": 10.0, "name": "AUD/USD"},
    "USDJPY": {"ticker": "USDJPY=X", "pip_size": 0.01,   "pip_value": 10.0, "name": "USD/JPY"},
}

RISK_PER_TRADE  = 100.0   # $ risk per trade (fixed)
MIN_RR          = 1.5     # minimum risk:reward ratio
MAX_SL_PIPS     = 50      # max SL in pips (no wide-stop trades)
MIN_RANGE_PIPS  = 20      # skip sessions with tiny dealing ranges


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def download_max_history(pair_key: str) -> pd.DataFrame:
    """Download max available 15m bars via yfinance (~60 days)."""
    import yfinance as yf
    ticker_sym = PAIRS[pair_key]["ticker"]
    print(f"  Downloading {PAIRS[pair_key]['name']} 15m history (max 60 days)...")
    df = yf.download(ticker_sym, period="60d", interval="15m",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    # Localize to EST
    if df.index.tz is None:
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True)).tz_convert("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")
    print(f"  Got {len(df)} bars: {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION SEGMENTATION — NY close = 5 PM EST
# ═══════════════════════════════════════════════════════════════════════════════

def build_ny_sessions(df: pd.DataFrame) -> Dict[date, pd.DataFrame]:
    """
    Split data into NY sessions (5 PM → 5 PM).
    Key = the DATE of the session START (the "previous" day in dealing range terms).
    Returns dict: session_start_date → bars DataFrame
    """
    sessions: Dict[date, List] = {}
    ny_close = dtime(17, 0)

    for ts, row in df.iterrows():
        # Which session does this bar belong to?
        if ts.time() >= ny_close:
            session_date = ts.date()
        else:
            session_date = (ts - timedelta(days=1)).date()
        sessions.setdefault(session_date, []).append((ts, row))

    # Convert to DataFrames, skip weekends with < 4 bars
    result = {}
    for d, rows in sessions.items():
        if d.weekday() >= 5:
            continue
        idx = pd.DatetimeIndex([r[0] for r in rows])
        data = pd.DataFrame([r[1] for r in rows], index=idx)
        if len(data) >= 4:
            result[d] = data
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDBACH LEVEL COMPUTATION — zero lookahead, uses closed prev session
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Levels:
    prev_high:      float
    prev_low:       float
    dealing_range:  float
    range_pips:     float
    equilibrium:    float
    po3_1:          float   # 33.3% from low
    po3_2:          float   # 66.7% from low
    po9:            List[float]
    pip_size:       float
    po9_increment:  float


def compute_levels(prev_session: pd.DataFrame, pip_size: float) -> Optional[Levels]:
    h = float(prev_session["High"].max())
    l = float(prev_session["Low"].min())
    r = h - l
    pips = r / pip_size
    if pips < MIN_RANGE_PIPS:
        return None
    equil = l + r * 0.5
    po3_1 = l + r / 3
    po3_2 = l + r * 2 / 3
    po9   = [l + r * i / 9 for i in range(1, 10)]
    return Levels(
        prev_high=h, prev_low=l, dealing_range=r,
        range_pips=pips, equilibrium=equil,
        po3_1=po3_1, po3_2=po3_2, po9=po9,
        pip_size=pip_size, po9_increment=r / 9,
    )


def price_zone(price: float, lvl: Levels) -> str:
    pct = (price - lvl.prev_low) / lvl.dealing_range
    if pct <= 0.111:   return "deep_discount"
    if pct <= 0.333:   return "discount"
    if pct <= 0.667:   return "neutral"
    if pct <= 0.889:   return "premium"
    return "deep_premium"


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    session_date:   date
    entry_time:     pd.Timestamp
    exit_time:      Optional[pd.Timestamp]
    pair:           str
    direction:      str         # "buy" | "sell"
    model:          str         # "discount_bounce" | "premium_rejection"
    entry:          float
    stop_loss:      float
    take_profit:    float
    sl_pips:        float
    tp_pips:        float
    rr:             float
    lot_size:       float
    zone:           str
    outcome:        str         # "win" | "loss" | "open" | "expired"
    exit_price:     Optional[float] = None
    pips_captured:  float = 0.0
    gross_pnl:      float = 0.0
    net_pnl:        float = 0.0
    dealing_range_pips: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION — mirrors goldbach_forex.py logic
# ═══════════════════════════════════════════════════════════════════════════════

def build_signal(
    bar_time: pd.Timestamp,
    open_price: float,
    lvl: Levels,
    pip_value: float,
    min_rr: float,
    max_sl_pips: int,
    pair: str,
    session_date: date,
) -> Optional[Trade]:
    """
    Generate a signal from a single bar's open price.
    No lookahead — uses only the bar's open (first tradeable price of that bar).
    """
    pip = lvl.pip_size
    zone = price_zone(open_price, lvl)

    if zone in ("deep_discount", "discount"):
        direction  = "buy"
        model      = "discount_bounce"
        entry      = open_price
        stop_loss  = lvl.prev_low - lvl.po9_increment
        take_profit = lvl.equilibrium
    elif zone in ("premium", "deep_premium"):
        direction  = "sell"
        model      = "premium_rejection"
        entry      = open_price
        stop_loss  = lvl.prev_high + lvl.po9_increment
        take_profit = lvl.equilibrium
    else:
        return None  # neutral — no trade

    sl_pips = abs(entry - stop_loss) / pip
    tp_pips = abs(take_profit - entry) / pip
    rr = tp_pips / sl_pips if sl_pips > 0 else 0

    if rr < min_rr:
        return None
    if sl_pips > max_sl_pips:
        return None
    if tp_pips < 5:
        return None  # degenerate TP

    lot_size = RISK_PER_TRADE / (sl_pips * pip_value) if sl_pips > 0 else 0

    return Trade(
        session_date=session_date,
        entry_time=bar_time,
        exit_time=None,
        pair=pair,
        direction=direction,
        model=model,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        sl_pips=round(sl_pips, 1),
        tp_pips=round(tp_pips, 1),
        rr=round(rr, 2),
        lot_size=round(lot_size, 4),
        zone=zone,
        outcome="open",
        dealing_range_pips=round(lvl.range_pips, 1),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE RESOLUTION — walk bars forward from entry, check SL/TP
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_trade(trade: Trade, session_bars: pd.DataFrame, pip_value: float) -> Trade:
    """
    Walk bars forward from entry bar until SL, TP, or session end.
    Uses bar High/Low for worst-case fill (no optimistic intra-bar fills).
    If both SL and TP hit same bar → SL wins (conservative).
    """
    pip = trade.stop_loss  # not pip, get from pip_size
    # recalculate pip_size
    pip_size = abs(trade.entry - trade.stop_loss) / trade.sl_pips if trade.sl_pips > 0 else 0.0001

    entry_ts = trade.entry_time
    future_bars = session_bars[session_bars.index > entry_ts]

    for ts, bar in future_bars.iterrows():
        bar_high = float(bar["High"])
        bar_low  = float(bar["Low"])

        if trade.direction == "buy":
            tp_hit = bar_high >= trade.take_profit
            sl_hit = bar_low  <= trade.stop_loss
        else:
            tp_hit = bar_low  <= trade.take_profit
            sl_hit = bar_high >= trade.stop_loss

        # Same-bar: conservative → SL
        if sl_hit and tp_hit:
            exit_p = trade.stop_loss
            outcome = "loss"
        elif tp_hit:
            exit_p = trade.take_profit
            outcome = "win"
        elif sl_hit:
            exit_p = trade.stop_loss
            outcome = "loss"
        else:
            continue

        if trade.direction == "buy":
            pips = (exit_p - trade.entry) / pip_size
        else:
            pips = (trade.entry - exit_p) / pip_size

        gross = pips * trade.lot_size * pip_value

        trade.exit_time    = ts
        trade.exit_price   = round(exit_p, 5)
        trade.pips_captured = round(pips, 1)
        trade.gross_pnl    = round(gross, 2)
        trade.net_pnl      = round(gross, 2)  # no fixed commission
        trade.outcome      = outcome
        return trade

    # Session ended without resolution
    last_bar = future_bars.iloc[-1] if not future_bars.empty else None
    if last_bar is not None:
        exit_p = float(last_bar["Close"])
        if trade.direction == "buy":
            pips = (exit_p - trade.entry) / pip_size
        else:
            pips = (trade.entry - exit_p) / pip_size
        gross = pips * trade.lot_size * pip_value
        trade.exit_time    = future_bars.index[-1]
        trade.exit_price   = round(exit_p, 5)
        trade.pips_captured = round(pips, 1)
        trade.gross_pnl    = round(gross, 2)
        trade.net_pnl      = round(gross, 2)
        trade.outcome      = "win" if gross > 0 else "loss"
    else:
        trade.outcome = "expired"

    return trade


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    df: pd.DataFrame,
    pair_key: str,
    min_rr: float = MIN_RR,
    max_sl_pips: int = MAX_SL_PIPS,
) -> List[Trade]:
    pip_size  = PAIRS[pair_key]["pip_size"]
    pip_value = PAIRS[pair_key]["pip_value"]
    pair_name = PAIRS[pair_key]["name"]

    sessions = build_ny_sessions(df)
    sorted_dates = sorted(sessions.keys())
    trades: List[Trade] = []

    print(f"\n  Running backtest on {len(sorted_dates)} NY sessions...")

    for i, session_date in enumerate(sorted_dates):
        if i == 0:
            continue  # need previous session for dealing range

        prev_date    = sorted_dates[i - 1]
        prev_session = sessions[prev_date]
        curr_session = sessions[session_date]

        lvl = compute_levels(prev_session, pip_size)
        if lvl is None:
            continue

        # One trade per session — use FIRST bar of current session as entry signal
        # This is the open bar after NY close (5 PM ET)
        first_bar = curr_session.iloc[0]
        bar_time  = curr_session.index[0]
        open_price = float(first_bar["Open"])

        # Skip weekends
        if session_date.weekday() >= 5:
            continue

        sig = build_signal(
            bar_time, open_price, lvl,
            pip_value, min_rr, max_sl_pips,
            pair_key, session_date,
        )
        if sig is None:
            continue

        sig = resolve_trade(sig, curr_session, pip_value)
        trades.append(sig)

    return trades


# ═══════════════════════════════════════════════════════════════════════════════
# DRAWDOWN
# ═══════════════════════════════════════════════════════════════════════════════

def compute_max_drawdown(trades: List[Trade]) -> Tuple[float, float]:
    """Returns (max_drawdown_dollars, max_drawdown_pct_of_peak)."""
    if not trades:
        return 0.0, 0.0
    equity = 0.0
    peak   = 0.0
    max_dd = 0.0
    max_dd_pct = 0.0
    for t in trades:
        equity += t.net_pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        dd_pct = dd / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct
    return round(max_dd, 2), round(max_dd_pct, 1)


def compute_equity_curve(trades: List[Trade]) -> List[float]:
    equity = 0.0
    curve = []
    for t in trades:
        equity += t.net_pnl
        curve.append(round(equity, 2))
    return curve


# ═══════════════════════════════════════════════════════════════════════════════
# TEAR SHEET
# ═══════════════════════════════════════════════════════════════════════════════

def print_tearsheet(trades: List[Trade], pair_key: str, min_rr: float, max_sl_pips: int) -> None:
    pair_name = PAIRS[pair_key]["name"]
    if not trades:
        print("\n  ⚠️  No trades generated — check data or loosen filters")
        return

    wins   = [t for t in trades if t.outcome == "win"]
    losses = [t for t in trades if t.outcome == "loss"]
    total  = len(trades)
    wr     = len(wins) / total * 100 if total > 0 else 0

    total_pips = sum(t.pips_captured for t in trades)
    total_pnl  = sum(t.net_pnl for t in trades)
    avg_win_pips  = np.mean([t.pips_captured for t in wins])   if wins   else 0
    avg_loss_pips = np.mean([t.pips_captured for t in losses]) if losses else 0

    max_dd, max_dd_pct = compute_max_drawdown(trades)
    equity_curve = compute_equity_curve(trades)

    # Consecutive stats
    max_con_wins = max_con_losses = cur_w = cur_l = 0
    for t in trades:
        if t.outcome == "win":
            cur_w += 1; cur_l = 0
            max_con_wins = max(max_con_wins, cur_w)
        else:
            cur_l += 1; cur_w = 0
            max_con_losses = max(max_con_losses, cur_l)

    # By model
    bounces = [t for t in trades if t.model == "discount_bounce"]
    rejections = [t for t in trades if t.model == "premium_rejection"]
    bounce_wr = len([t for t in bounces if t.outcome == "win"]) / len(bounces) * 100 if bounces else 0
    reject_wr = len([t for t in rejections if t.outcome == "win"]) / len(rejections) * 100 if rejections else 0

    pnl_sign = "+" if total_pnl >= 0 else ""
    pips_sign = "+" if total_pips >= 0 else ""
    dd_color = "⚠️ " if max_dd > 300 else ""

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║   GOLDBACH FOREX BACKTEST — {pair_name:<28} ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Min RR: {min_rr}x  |  Max SL: {max_sl_pips} pips  |  Risk/trade: $100    ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  PERFORMANCE SUMMARY                                     ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Total Trades:      {total:<38} ║")
    print(f"║  Win Rate:          {wr:.1f}%  ({len(wins)}W / {len(losses)}L){'':<26} ║")
    print(f"║  Total Pips:        {pips_sign}{total_pips:.1f} pips{'':<35} ║")
    print(f"║  Net P&L:           {pnl_sign}${total_pnl:.2f} (@ $100 risk/trade){'':<16} ║")
    print(f"║  Max Drawdown:      {dd_color}${max_dd:.2f}  ({max_dd_pct:.1f}% of peak){'':<18} ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  TRADE QUALITY                                           ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Avg Win:           +{avg_win_pips:.1f} pips{'':<37} ║")
    print(f"║  Avg Loss:          {avg_loss_pips:.1f} pips{'':<37} ║")
    print(f"║  Avg R:R:           {np.mean([t.rr for t in trades]):.2f}:1{'':<38} ║")
    print(f"║  Max Con. Wins:     {max_con_wins:<38} ║")
    print(f"║  Max Con. Losses:   {max_con_losses:<38} ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  BY MODEL                                                ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Discount Bounce:   {len(bounces)} trades | {bounce_wr:.0f}% WR{'':<28} ║")
    print(f"║  Premium Rejection: {len(rejections)} trades | {reject_wr:.0f}% WR{'':<28} ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  EQUITY CURVE (last 10 trades)                          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    last10 = equity_curve[-10:]
    for i, (t, eq) in enumerate(zip(trades[-10:], last10)):
        icon = "✅" if t.outcome == "win" else "❌"
        pnl = f"{'+' if t.net_pnl >= 0 else ''}{t.net_pnl:.0f}"
        eq_str = f"${eq:+.0f}"
        print(f"║  {icon} {t.session_date} | {t.pair} {t.direction:<4} | {pnl:>6} | eq: {eq_str:<12} ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Trade log
    print()
    print(f"  {'DATE':<12} {'DIR':<5} {'MODEL':<20} {'ENTRY':>8} {'EXIT':>8} "
          f"{'PIPS':>7} {'P&L':>8} {'OUT'}")
    print(f"  {'─'*90}")
    for t in trades:
        icon = "✅" if t.outcome == "win" else "❌" if t.outcome == "loss" else "⏰"
        exit_str = f"{t.exit_price:.5f}" if t.exit_price else "–"
        pips_str = f"{t.pips_captured:+.1f}"
        pnl_str  = f"${t.net_pnl:+.2f}"
        print(f"  {str(t.session_date):<12} {t.direction:<5} {t.model:<20} "
              f"{t.entry:>8.5f} {exit_str:>8} {pips_str:>7} {pnl_str:>8}  {icon}")

    print()
    verdict = "✅ EDGE DETECTED" if wr >= 50 and total_pnl > 0 else \
              "⚠️  MARGINAL EDGE" if wr >= 45 or total_pnl > 0 else \
              "❌ NO EDGE — review parameters"
    print(f"  VERDICT: {verdict}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Goldbach Forex Backtester — PO3/PO9 discount/premium strategy"
    )
    parser.add_argument("--pair",      default="EURUSD",
                        help="Pair to backtest (EURUSD, GBPUSD, AUDUSD, USDJPY)")
    parser.add_argument("--min-rr",   type=float, default=MIN_RR,
                        help=f"Minimum R:R ratio (default: {MIN_RR})")
    parser.add_argument("--max-sl",   type=int,   default=MAX_SL_PIPS,
                        help=f"Max SL in pips (default: {MAX_SL_PIPS})")
    args = parser.parse_args()

    pair_key = args.pair.upper()
    if pair_key not in PAIRS:
        print(f"❌ Unknown pair: {pair_key}. Options: {list(PAIRS.keys())}")
        sys.exit(1)

    now = datetime.now(EST)
    print()
    print("═" * 60)
    print(f"  GOLDBACH FOREX BACKTESTER")
    print(f"  {now.strftime('%I:%M %p ET · %a %b %d, %Y')}")
    print(f"  Pair: {PAIRS[pair_key]['name']}  |  Min RR: {args.min_rr}  |  Max SL: {args.max_sl}p")
    print("═" * 60)

    df = download_max_history(pair_key)
    if df.empty or len(df) < 50:
        print("❌ Not enough data — try again or check yfinance availability")
        sys.exit(1)

    trades = run_backtest(df, pair_key, min_rr=args.min_rr, max_sl_pips=args.max_sl)
    print_tearsheet(trades, pair_key, args.min_rr, args.max_sl)


if __name__ == "__main__":
    main()
