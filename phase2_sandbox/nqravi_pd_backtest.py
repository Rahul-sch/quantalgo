#!/usr/bin/env python3
"""
NQravi Premium/Discount Filter Backtest
========================================
Standalone backtest: applies the NQravi continuation model's premium/discount
zone filter to QQQ 15m FVG signals over the last 30 days.

Outputs:
  - Total FVG signals BEFORE the PD filter
  - Total FVG signals AFTER the PD filter
  - Isolated win rate of filtered setups

No modifications to any live engine files.
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

TICKER = "QQQ"
PERIOD = "1mo"            # ~30 days of 15m data (yfinance max for 15m)
INTERVAL = "15m"
SWING_LOOKBACK = 20       # bars each side for swing detection
ATR_PERIOD = 14
ATR_SL_MULT = 1.0         # 1x ATR stop loss
RR_RATIO = 2.0            # 2:1 reward/risk
COMMISSION_RT = 2.40      # round-trip commission
SLIPPAGE_PCT = 0.005      # 0.005% per trade (entry + exit)
POSITION_SIZE = 10_000    # $10k notional per trade

# Session windows (ET minutes from midnight)
AM_START = 570   # 9:30
AM_END   = 690   # 11:30
PM_START = 810   # 13:30
PM_END   = 930   # 15:30


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCH
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_data() -> pd.DataFrame:
    """Fetch QQQ 15m data for the last 30 days."""
    print(f"  📡 Fetching {TICKER} {INTERVAL} data ({PERIOD})...")
    df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    print(f"  ✅ {len(df)} bars loaded ({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """True Range → ATR."""
    h = df["High"]
    l = df["Low"]
    c = df["Close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def detect_fvgs(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Vectorized 3-candle FVG detection.
    Bull FVG: candle[i-2].High < candle[i].Low  (gap up)
    Bear FVG: candle[i-2].Low  > candle[i].High (gap down)
    """
    h = df["High"].values
    l = df["Low"].values
    n = len(df)

    bull_top = np.full(n, np.nan)
    bull_bot = np.full(n, np.nan)
    bear_top = np.full(n, np.nan)
    bear_bot = np.full(n, np.nan)

    for i in range(2, n):
        # Bullish FVG
        if h[i - 2] < l[i]:
            bull_top[i] = l[i]       # FVG top = candle[i] low
            bull_bot[i] = h[i - 2]   # FVG bot = candle[i-2] high
        # Bearish FVG
        if l[i - 2] > h[i]:
            bear_top[i] = l[i - 2]   # FVG top = candle[i-2] low
            bear_bot[i] = h[i]       # FVG bot = candle[i] high

    return (
        pd.Series(bull_top, index=df.index),
        pd.Series(bull_bot, index=df.index),
        pd.Series(bear_top, index=df.index),
        pd.Series(bear_bot, index=df.index),
    )


def find_swings(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> Tuple[pd.Series, pd.Series]:
    """Rolling window swing high/low detection."""
    window = 2 * lookback + 1
    roll_max = df["High"].rolling(window, center=True).max()
    roll_min = df["Low"].rolling(window, center=True).min()

    swing_highs = pd.Series(np.nan, index=df.index)
    swing_lows = pd.Series(np.nan, index=df.index)

    swing_highs[df["High"] == roll_max] = df["High"][df["High"] == roll_max]
    swing_lows[df["Low"] == roll_min] = df["Low"][df["Low"] == roll_min]

    return swing_highs, swing_lows


def compute_premium_discount(df: pd.DataFrame, swing_highs: pd.Series,
                              swing_lows: pd.Series) -> pd.Series:
    """
    NQravi Premium/Discount filter.
    For each bar, find the most recent confirmed swing high and swing low
    (using .shift(1) to avoid lookahead). Compute the 50% fib level.
    Return a Series: price position as % of swing range (0=swing low, 1=swing high).
    Values > 0.5 = premium, < 0.5 = discount.
    """
    # Forward-fill the most recent swing high and swing low (shifted to avoid lookahead)
    recent_sh = swing_highs.shift(SWING_LOOKBACK + 1).ffill()
    recent_sl = swing_lows.shift(SWING_LOOKBACK + 1).ffill()

    swing_range = recent_sh - recent_sl
    # Avoid division by zero
    swing_range = swing_range.replace(0, np.nan)

    position_in_range = (df["Close"] - recent_sl) / swing_range
    return position_in_range


def is_in_session(timestamps: pd.DatetimeIndex) -> pd.Series:
    """Check if each bar is within AM or PM session (ET)."""
    # Convert to ET
    et = timestamps.tz_convert("US/Eastern") if timestamps.tz else timestamps.tz_localize("UTC").tz_convert("US/Eastern")
    minutes = et.hour * 60 + et.minute
    in_am = (minutes >= AM_START) & (minutes < AM_END)
    in_pm = (minutes >= PM_START) & (minutes < PM_END)
    return pd.Series(in_am | in_pm, index=timestamps)


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION & BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

def generate_signals(df: pd.DataFrame) -> List[Dict]:
    """
    Generate FVG continuation signals with retest entry.
    Returns list of signal dicts with 'pd_filtered' flag.
    """
    atr = compute_atr(df)
    bull_top, bull_bot, bear_top, bear_bot = detect_fvgs(df)
    swing_highs, swing_lows = find_swings(df)
    pd_position = compute_premium_discount(df, swing_highs, swing_lows)
    session_mask = is_in_session(df.index)

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values

    signals = []

    # Track armed FVGs waiting for retest
    # armed[fvg_bar_idx] = (direction, limit_price, fvg_other_edge, atr_val, armed_bar)
    armed = {}
    RETEST_MAX_BARS = 5

    for i in range(30, len(df)):
        if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
            continue

        # ── ARM new bullish FVG ──
        if not np.isnan(bull_top.iloc[i]):
            fvg_top_val = float(bull_top.iloc[i])
            fvg_bot_val = float(bull_bot.iloc[i])
            armed[i] = ("long", fvg_top_val, fvg_bot_val, float(atr.iloc[i]), i)

        # ── ARM new bearish FVG ──
        if not np.isnan(bear_top.iloc[i]):
            fvg_top_val = float(bear_top.iloc[i])
            fvg_bot_val = float(bear_bot.iloc[i])
            armed[i] = ("short", fvg_bot_val, fvg_top_val, float(atr.iloc[i]), i)

        # ── Check armed FVGs for retest fill ──
        to_remove = []
        for fvg_bar, (direction, limit_price, other_edge, atr_val, armed_bar) in armed.items():
            bars_elapsed = i - armed_bar
            if bars_elapsed > RETEST_MAX_BARS:
                to_remove.append(fvg_bar)
                continue
            if bars_elapsed == 0:
                continue

            # Check structure invalidation
            if direction == "long" and low[i] < other_edge:
                to_remove.append(fvg_bar)
                continue
            if direction == "short" and high[i] > other_edge:
                to_remove.append(fvg_bar)
                continue

            # Check retest
            if not (low[i] <= limit_price <= high[i]):
                continue

            # Must be in session
            if not session_mask.iloc[i]:
                continue

            # Build trade
            entry = limit_price
            if direction == "long":
                sl = entry - atr_val * ATR_SL_MULT
                risk = entry - sl
                if risk <= 0:
                    continue
                tp = entry + risk * RR_RATIO
            else:
                sl = entry + atr_val * ATR_SL_MULT
                risk = sl - entry
                if risk <= 0:
                    continue
                tp = entry - risk * RR_RATIO

            # ── Premium/Discount check ──
            pd_val = pd_position.iloc[i]
            pd_pass = True
            if not np.isnan(pd_val):
                if direction == "long" and pd_val > 0.5:
                    pd_pass = False   # longing in premium → blocked
                elif direction == "short" and pd_val < 0.5:
                    pd_pass = False   # shorting in discount → blocked

            signals.append({
                "bar_idx": i,
                "time": str(df.index[i]),
                "direction": direction,
                "entry": round(entry, 2),
                "sl": round(sl, 2),
                "tp": round(tp, 2),
                "risk": round(risk, 2),
                "atr": round(atr_val, 2),
                "pd_position": round(pd_val, 4) if not np.isnan(pd_val) else None,
                "pd_zone": "premium" if (not np.isnan(pd_val) and pd_val > 0.5) else "discount",
                "pd_pass": pd_pass,
            })

            to_remove.append(fvg_bar)

        for k in to_remove:
            armed.pop(k, None)

    return signals


def simulate_trades(signals: List[Dict], df: pd.DataFrame) -> List[Dict]:
    """
    Walk forward from each signal's entry bar to determine outcome.
    Uses candle High/Low for SL/TP checks (not Close).
    """
    high = df["High"].values
    low = df["Low"].values
    n = len(df)

    results = []
    for sig in signals:
        entry = sig["entry"]
        sl = sig["sl"]
        tp = sig["tp"]
        direction = sig["direction"]
        start_bar = sig["bar_idx"] + 1  # next bar after entry

        outcome = "open"
        exit_price = entry
        exit_bar = None

        for j in range(start_bar, min(start_bar + 40, n)):  # max 40 bars hold
            if direction == "long":
                # Check SL first (conservative)
                if low[j] <= sl:
                    outcome = "loss"
                    exit_price = sl
                    exit_bar = j
                    break
                if high[j] >= tp:
                    outcome = "win"
                    exit_price = tp
                    exit_bar = j
                    break
            else:  # short
                if high[j] >= sl:
                    outcome = "loss"
                    exit_price = sl
                    exit_bar = j
                    break
                if low[j] <= tp:
                    outcome = "win"
                    exit_price = tp
                    exit_bar = j
                    break

        # If still open after 40 bars, close at market
        if outcome == "open":
            last_bar = min(start_bar + 39, n - 1)
            exit_price = float(df["Close"].iloc[last_bar])
            exit_bar = last_bar
            if direction == "long":
                outcome = "win" if exit_price > entry else "loss"
            else:
                outcome = "win" if exit_price < entry else "loss"

        # P&L calculation with friction
        shares = POSITION_SIZE / entry
        if direction == "long":
            gross_pnl = (exit_price - entry) * shares
        else:
            gross_pnl = (entry - exit_price) * shares

        slippage = POSITION_SIZE * (SLIPPAGE_PCT / 100) * 2  # entry + exit
        net_pnl = gross_pnl - COMMISSION_RT - slippage

        results.append({
            **sig,
            "outcome": outcome,
            "exit_price": round(exit_price, 2),
            "exit_bar": exit_bar,
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(results: List[Dict]) -> None:
    """Print the backtest report."""
    all_trades = results
    pd_filtered = [r for r in results if r["pd_pass"]]
    pd_blocked = [r for r in results if not r["pd_pass"]]

    def stats(trades, label):
        if not trades:
            print(f"\n  {label}: 0 trades")
            return
        wins = [t for t in trades if t["outcome"] == "win"]
        losses = [t for t in trades if t["outcome"] == "loss"]
        wr = len(wins) / len(trades) * 100
        total_pnl = sum(t["net_pnl"] for t in trades)
        avg_win = np.mean([t["net_pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["net_pnl"] for t in losses]) if losses else 0
        pf = abs(sum(t["net_pnl"] for t in wins) / sum(t["net_pnl"] for t in losses)) if losses and sum(t["net_pnl"] for t in losses) != 0 else float("inf")
        longs = [t for t in trades if t["direction"] == "long"]
        shorts = [t for t in trades if t["direction"] == "short"]
        long_wr = len([t for t in longs if t["outcome"] == "win"]) / len(longs) * 100 if longs else 0
        short_wr = len([t for t in shorts if t["outcome"] == "win"]) / len(shorts) * 100 if shorts else 0

        print(f"\n  {label}")
        print(f"  {'─' * 50}")
        print(f"  Total trades:    {len(trades)}")
        print(f"  Wins / Losses:   {len(wins)}W / {len(losses)}L")
        print(f"  Win Rate:        {wr:.1f}%")
        print(f"  Long WR:         {long_wr:.1f}% ({len(longs)} trades)")
        print(f"  Short WR:        {short_wr:.1f}% ({len(shorts)} trades)")
        print(f"  Net P&L:         ${total_pnl:,.2f}")
        print(f"  Avg Win:         ${avg_win:,.2f}")
        print(f"  Avg Loss:        ${avg_loss:,.2f}")
        print(f"  Profit Factor:   {pf:.2f}" if pf != float("inf") else f"  Profit Factor:   ∞")

    print(f"\n  {'═' * 60}")
    print(f"  📊 NQravi PREMIUM/DISCOUNT FILTER BACKTEST")
    print(f"  {'═' * 60}")
    print(f"  Ticker:          {TICKER}")
    print(f"  Period:          {PERIOD} ({INTERVAL})")
    print(f"  ATR SL:          {ATR_SL_MULT}x | RR: {RR_RATIO}:1")
    print(f"  Commission:      ${COMMISSION_RT} RT | Slippage: {SLIPPAGE_PCT}%")
    print(f"  Swing Lookback:  {SWING_LOOKBACK} bars")

    stats(all_trades, "ALL SIGNALS (no PD filter)")
    stats(pd_filtered, "AFTER PD FILTER (NQravi model)")

    # Show what the filter blocked
    if pd_blocked:
        blocked_wins = len([t for t in pd_blocked if t["outcome"] == "win"])
        blocked_losses = len([t for t in pd_blocked if t["outcome"] == "loss"])
        blocked_wr = blocked_wins / len(pd_blocked) * 100
        blocked_pnl = sum(t["net_pnl"] for t in pd_blocked)
        print(f"\n  BLOCKED BY PD FILTER")
        print(f"  {'─' * 50}")
        print(f"  Blocked trades:  {len(pd_blocked)}")
        print(f"  Blocked WR:      {blocked_wr:.1f}% ({blocked_wins}W / {blocked_losses}L)")
        print(f"  Blocked P&L:     ${blocked_pnl:,.2f}")
        print(f"  → Filter saved:  ${abs(blocked_pnl):,.2f}" if blocked_pnl < 0 else f"  → Filter cost:   ${blocked_pnl:,.2f} (missed profits)")

    # Premium/Discount zone breakdown
    premium_trades = [t for t in all_trades if t["pd_zone"] == "premium"]
    discount_trades = [t for t in all_trades if t["pd_zone"] == "discount"]
    if premium_trades:
        p_wr = len([t for t in premium_trades if t["outcome"] == "win"]) / len(premium_trades) * 100
        print(f"\n  ZONE BREAKDOWN")
        print(f"  {'─' * 50}")
        print(f"  Premium zone:    {len(premium_trades)} trades, WR {p_wr:.1f}%")
    if discount_trades:
        d_wr = len([t for t in discount_trades if t["outcome"] == "win"]) / len(discount_trades) * 100
        print(f"  Discount zone:   {len(discount_trades)} trades, WR {d_wr:.1f}%")

    # Individual trade log (last 10)
    print(f"\n  RECENT TRADES (last 10)")
    print(f"  {'─' * 50}")
    for t in all_trades[-10:]:
        emoji = "✅" if t["outcome"] == "win" else "❌"
        pd_tag = "✓PD" if t["pd_pass"] else "✗PD"
        zone = t["pd_zone"][:4].upper()
        print(f"  {emoji} {t['direction']:5s} @ ${t['entry']:.2f} → ${t['exit_price']:.2f} "
              f"| ${t['net_pnl']:+.2f} | {zone} {pd_tag} | {t['time'][:16]}")

    print(f"\n  {'═' * 60}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n  {'═' * 60}")
    print(f"  🔬 NQravi Premium/Discount Filter — Phase 2 Sandbox")
    print(f"  {'═' * 60}")

    df = fetch_data()
    if len(df) < 50:
        print("  ❌ Not enough data")
        sys.exit(1)

    print(f"  🔍 Detecting FVGs and computing premium/discount zones...")
    signals = generate_signals(df)
    print(f"  📊 {len(signals)} raw signals found")

    print(f"  ⚡ Simulating trades with friction...")
    results = simulate_trades(signals, df)

    print_report(results)


if __name__ == "__main__":
    main()
