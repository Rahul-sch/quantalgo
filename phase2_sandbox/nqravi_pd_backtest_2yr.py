#!/usr/bin/env python3
"""
NQravi Premium/Discount Filter — 2-Year Stress Test
=====================================================
Uses cached QQQ_15m_2yr.csv (31K bars, Apr 2024 → Mar 2026).
Walk-forward quarterly windows + max drawdown tracking.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "QQQ_15m_2yr.csv")

SWING_LOOKBACK = 20
ATR_PERIOD = 14
ATR_SL_MULT = 1.0
RR_RATIO = 2.0
COMMISSION_RT = 2.40
SLIPPAGE_PCT = 0.005
POSITION_SIZE = 10_000
RETEST_MAX_BARS = 5
MAX_HOLD_BARS = 40

# Session windows (ET minutes from midnight)
AM_START, AM_END = 570, 690    # 9:30–11:30
PM_START, PM_END = 810, 930    # 13:30–15:30


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    print(f"  📂 Loading cached data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    print(f"  ✅ {len(df):,} bars ({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD).mean()


def detect_fvgs(df: pd.DataFrame):
    h = df["High"].values
    l = df["Low"].values
    n = len(df)
    bull_top = np.full(n, np.nan)
    bull_bot = np.full(n, np.nan)
    bear_top = np.full(n, np.nan)
    bear_bot = np.full(n, np.nan)
    for i in range(2, n):
        if h[i - 2] < l[i]:
            bull_top[i] = l[i]
            bull_bot[i] = h[i - 2]
        if l[i - 2] > h[i]:
            bear_top[i] = l[i - 2]
            bear_bot[i] = h[i]
    return (pd.Series(bull_top, index=df.index), pd.Series(bull_bot, index=df.index),
            pd.Series(bear_top, index=df.index), pd.Series(bear_bot, index=df.index))


def find_swings(df: pd.DataFrame):
    window = 2 * SWING_LOOKBACK + 1
    roll_max = df["High"].rolling(window, center=True).max()
    roll_min = df["Low"].rolling(window, center=True).min()
    sh = pd.Series(np.nan, index=df.index)
    sl = pd.Series(np.nan, index=df.index)
    sh[df["High"] == roll_max] = df["High"][df["High"] == roll_max]
    sl[df["Low"] == roll_min] = df["Low"][df["Low"] == roll_min]
    return sh, sl


def compute_premium_discount(df, swing_highs, swing_lows):
    recent_sh = swing_highs.shift(SWING_LOOKBACK + 1).ffill()
    recent_sl = swing_lows.shift(SWING_LOOKBACK + 1).ffill()
    rng = (recent_sh - recent_sl).replace(0, np.nan)
    return (df["Close"] - recent_sl) / rng


def is_in_session(idx):
    try:
        et = idx.tz_convert("US/Eastern")
    except TypeError:
        et = idx.tz_localize("UTC").tz_convert("US/Eastern")
    mins = et.hour * 60 + et.minute
    return pd.Series((mins >= AM_START) & (mins < AM_END) | (mins >= PM_START) & (mins < PM_END), index=idx)


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_signals(df: pd.DataFrame) -> List[Dict]:
    atr = compute_atr(df)
    bull_top, bull_bot, bear_top, bear_bot = detect_fvgs(df)
    swing_highs, swing_lows = find_swings(df)
    pd_pos = compute_premium_discount(df, swing_highs, swing_lows)
    sess = is_in_session(df.index)

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    signals = []
    armed = {}

    for i in range(30, len(df)):
        atr_val = atr.iloc[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        # ARM bullish FVG
        if not np.isnan(bull_top.iloc[i]):
            armed[i] = ("long", float(bull_top.iloc[i]), float(bull_bot.iloc[i]), float(atr_val), i)
        # ARM bearish FVG
        if not np.isnan(bear_top.iloc[i]):
            armed[i] = ("short", float(bear_bot.iloc[i]), float(bear_top.iloc[i]), float(atr_val), i)

        # Check retests
        to_remove = []
        for fvg_bar, (direction, limit_price, other_edge, atr_arm, armed_bar) in armed.items():
            bars_elapsed = i - armed_bar
            if bars_elapsed > RETEST_MAX_BARS:
                to_remove.append(fvg_bar); continue
            if bars_elapsed == 0:
                continue
            if direction == "long" and low[i] < other_edge:
                to_remove.append(fvg_bar); continue
            if direction == "short" and high[i] > other_edge:
                to_remove.append(fvg_bar); continue
            if not (low[i] <= limit_price <= high[i]):
                continue
            if not sess.iloc[i]:
                continue

            entry = limit_price
            if direction == "long":
                sl_price = entry - atr_arm * ATR_SL_MULT
                risk = entry - sl_price
                if risk <= 0: continue
                tp = entry + risk * RR_RATIO
            else:
                sl_price = entry + atr_arm * ATR_SL_MULT
                risk = sl_price - entry
                if risk <= 0: continue
                tp = entry - risk * RR_RATIO

            pd_val = pd_pos.iloc[i]
            pd_pass = True
            if not np.isnan(pd_val):
                if direction == "long" and pd_val > 0.5:
                    pd_pass = False
                elif direction == "short" and pd_val < 0.5:
                    pd_pass = False

            signals.append({
                "bar_idx": i,
                "time": df.index[i],
                "direction": direction,
                "entry": round(entry, 2),
                "sl": round(sl_price, 2),
                "tp": round(tp, 2),
                "risk": round(risk, 2),
                "pd_position": round(pd_val, 4) if not np.isnan(pd_val) else None,
                "pd_zone": "premium" if (not np.isnan(pd_val) and pd_val > 0.5) else "discount",
                "pd_pass": pd_pass,
            })
            to_remove.append(fvg_bar)

        for k in to_remove:
            armed.pop(k, None)

    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_trades(signals: List[Dict], df: pd.DataFrame) -> List[Dict]:
    high = df["High"].values
    low = df["Low"].values
    n = len(df)
    results = []

    for sig in signals:
        entry = sig["entry"]
        sl = sig["sl"]
        tp = sig["tp"]
        d = sig["direction"]
        start = sig["bar_idx"] + 1

        outcome = "open"
        exit_price = entry
        exit_bar = None

        for j in range(start, min(start + MAX_HOLD_BARS, n)):
            if d == "long":
                if low[j] <= sl:
                    outcome, exit_price, exit_bar = "loss", sl, j; break
                if high[j] >= tp:
                    outcome, exit_price, exit_bar = "win", tp, j; break
            else:
                if high[j] >= sl:
                    outcome, exit_price, exit_bar = "loss", sl, j; break
                if low[j] <= tp:
                    outcome, exit_price, exit_bar = "win", tp, j; break

        if outcome == "open":
            last = min(start + MAX_HOLD_BARS - 1, n - 1)
            exit_price = float(df["Close"].iloc[last])
            exit_bar = last
            outcome = "win" if (d == "long" and exit_price > entry) or (d == "short" and exit_price < entry) else "loss"

        shares = POSITION_SIZE / entry
        gross = (exit_price - entry) * shares if d == "long" else (entry - exit_price) * shares
        slip = POSITION_SIZE * (SLIPPAGE_PCT / 100) * 2
        net = gross - COMMISSION_RT - slip

        results.append({**sig, "outcome": outcome, "exit_price": round(exit_price, 2),
                        "exit_bar": exit_bar, "gross_pnl": round(gross, 2), "net_pnl": round(net, 2)})
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAX DRAWDOWN
# ═══════════════════════════════════════════════════════════════════════════════

def compute_max_drawdown(trades: List[Dict]) -> Tuple[float, float]:
    """Returns (max_drawdown_$, max_drawdown_%)."""
    if not trades:
        return 0.0, 0.0
    equity = [POSITION_SIZE]  # starting capital as reference
    for t in trades:
        equity.append(equity[-1] + t["net_pnl"])
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = dd.min()
    max_dd_pct = (max_dd / peak[np.argmin(dd)]) * 100 if peak[np.argmin(dd)] > 0 else 0
    return round(max_dd, 2), round(max_dd_pct, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# QUARTERLY WINDOWS
# ═══════════════════════════════════════════════════════════════════════════════

def assign_quarter(ts) -> str:
    """Assign a trade to a quarterly bucket."""
    y = ts.year
    q = (ts.month - 1) // 3 + 1
    return f"Q{q} {y}"


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def stats_block(trades, label):
    if not trades:
        print(f"\n  {label}: 0 trades")
        return
    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    wr = len(wins) / len(trades) * 100
    total_pnl = sum(t["net_pnl"] for t in trades)
    avg_win = np.mean([t["net_pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["net_pnl"] for t in losses]) if losses else 0
    loss_sum = sum(t["net_pnl"] for t in losses)
    win_sum = sum(t["net_pnl"] for t in wins)
    pf = abs(win_sum / loss_sum) if loss_sum != 0 else float("inf")
    longs = [t for t in trades if t["direction"] == "long"]
    shorts = [t for t in trades if t["direction"] == "short"]
    long_wr = len([t for t in longs if t["outcome"] == "win"]) / len(longs) * 100 if longs else 0
    short_wr = len([t for t in shorts if t["outcome"] == "win"]) / len(shorts) * 100 if shorts else 0
    max_dd, max_dd_pct = compute_max_drawdown(trades)

    print(f"\n  {label}")
    print(f"  {'─' * 55}")
    print(f"  Total trades:     {len(trades)}")
    print(f"  Wins / Losses:    {len(wins)}W / {len(losses)}L")
    print(f"  Win Rate:         {wr:.1f}%")
    print(f"  Long WR:          {long_wr:.1f}% ({len(longs)} trades)")
    print(f"  Short WR:         {short_wr:.1f}% ({len(shorts)} trades)")
    print(f"  Net P&L:          ${total_pnl:,.2f}")
    print(f"  Avg Win:          ${avg_win:,.2f}")
    print(f"  Avg Loss:         ${avg_loss:,.2f}")
    pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
    print(f"  Profit Factor:    {pf_str}")
    print(f"  Max Drawdown:     ${max_dd:,.2f} ({max_dd_pct:.1f}%)")


def print_quarterly(trades, label):
    """Print quarterly breakdown."""
    if not trades:
        return
    quarters = {}
    for t in trades:
        q = assign_quarter(t["time"])
        quarters.setdefault(q, []).append(t)

    print(f"\n  {label} — QUARTERLY WALK-FORWARD")
    print(f"  {'─' * 55}")
    print(f"  {'Quarter':<10} {'Trades':>7} {'Wins':>6} {'WR':>7} {'Net P&L':>12} {'PF':>8}")
    print(f"  {'─' * 55}")

    for q in sorted(quarters.keys()):
        qt = quarters[q]
        wins = len([t for t in qt if t["outcome"] == "win"])
        wr = wins / len(qt) * 100 if qt else 0
        pnl = sum(t["net_pnl"] for t in qt)
        w_sum = sum(t["net_pnl"] for t in qt if t["outcome"] == "win")
        l_sum = sum(t["net_pnl"] for t in qt if t["outcome"] == "loss")
        pf = abs(w_sum / l_sum) if l_sum != 0 else float("inf")
        pf_s = f"{pf:.2f}" if pf != float("inf") else "∞"
        marker = "✅" if pnl > 0 else "❌"
        print(f"  {q:<10} {len(qt):>7} {wins:>6} {wr:>6.1f}% ${pnl:>10,.2f} {pf_s:>8}  {marker}")


def print_report(results: List[Dict]):
    all_trades = results
    filtered = [r for r in results if r["pd_pass"]]
    blocked = [r for r in results if not r["pd_pass"]]

    print(f"\n  {'═' * 60}")
    print(f"  📊 NQravi PREMIUM/DISCOUNT FILTER — 2-YEAR STRESS TEST")
    print(f"  {'═' * 60}")
    print(f"  Ticker:           QQQ 15m")
    print(f"  Period:           {all_trades[0]['time'].strftime('%Y-%m-%d')} → {all_trades[-1]['time'].strftime('%Y-%m-%d')}")
    print(f"  ATR SL:           {ATR_SL_MULT}x | RR: {RR_RATIO}:1")
    print(f"  Commission:       ${COMMISSION_RT} RT | Slippage: {SLIPPAGE_PCT}%")
    print(f"  Swing Lookback:   {SWING_LOOKBACK} bars")
    print(f"  Retest Max Bars:  {RETEST_MAX_BARS}")

    stats_block(all_trades, "ALL SIGNALS (no PD filter)")
    stats_block(filtered, "AFTER PD FILTER (NQravi model)")

    # Blocked trades
    if blocked:
        bw = len([t for t in blocked if t["outcome"] == "win"])
        bl = len([t for t in blocked if t["outcome"] == "loss"])
        bwr = bw / len(blocked) * 100
        bpnl = sum(t["net_pnl"] for t in blocked)
        print(f"\n  BLOCKED BY PD FILTER")
        print(f"  {'─' * 55}")
        print(f"  Blocked trades:   {len(blocked)}")
        print(f"  Blocked WR:       {bwr:.1f}% ({bw}W / {bl}L)")
        print(f"  Blocked P&L:      ${bpnl:,.2f}")
        if bpnl < 0:
            print(f"  → Filter saved:   ${abs(bpnl):,.2f}")
        else:
            print(f"  → Filter cost:    ${bpnl:,.2f} (missed profits)")

    # Zone breakdown
    premium = [t for t in all_trades if t["pd_zone"] == "premium"]
    discount = [t for t in all_trades if t["pd_zone"] == "discount"]
    print(f"\n  ZONE BREAKDOWN (all trades)")
    print(f"  {'─' * 55}")
    if premium:
        pwr = len([t for t in premium if t["outcome"] == "win"]) / len(premium) * 100
        ppnl = sum(t["net_pnl"] for t in premium)
        print(f"  Premium zone:     {len(premium)} trades, WR {pwr:.1f}%, P&L ${ppnl:,.2f}")
    if discount:
        dwr = len([t for t in discount if t["outcome"] == "win"]) / len(discount) * 100
        dpnl = sum(t["net_pnl"] for t in discount)
        print(f"  Discount zone:    {len(discount)} trades, WR {dwr:.1f}%, P&L ${dpnl:,.2f}")

    # Quarterly walk-forward
    print_quarterly(all_trades, "ALL SIGNALS")
    print_quarterly(filtered, "PD FILTERED")

    # Equity curve summary
    if filtered:
        eq = [POSITION_SIZE]
        for t in filtered:
            eq.append(eq[-1] + t["net_pnl"])
        print(f"\n  EQUITY CURVE (PD Filtered)")
        print(f"  {'─' * 55}")
        print(f"  Starting:         ${eq[0]:,.2f}")
        print(f"  Final:            ${eq[-1]:,.2f}")
        print(f"  Peak:             ${max(eq):,.2f}")
        print(f"  Trough:           ${min(eq):,.2f}")

    print(f"\n  {'═' * 60}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n  {'═' * 60}")
    print(f"  🔬 NQravi PD Filter — 2-Year Stress Test")
    print(f"  {'═' * 60}")

    df = load_data()
    if len(df) < 200:
        print("  ❌ Not enough data"); sys.exit(1)

    print(f"  🔍 Detecting FVGs + computing PD zones across {len(df):,} bars...")
    signals = generate_signals(df)
    print(f"  📊 {len(signals)} raw signals found")
    pd_pass = sum(1 for s in signals if s["pd_pass"])
    print(f"  🎯 {pd_pass} pass PD filter, {len(signals) - pd_pass} blocked")

    print(f"  ⚡ Simulating all trades with friction...")
    results = simulate_trades(signals, df)

    print_report(results)


if __name__ == "__main__":
    main()
