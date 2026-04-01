#!/usr/bin/env python3
"""
FVG Break-Even Exit Management — 2-Year Stress Test
=====================================================
NQravi's exit rule: once price reaches the next Internal Range Liquidity
(local pivot high for longs, local pivot low for shorts), move SL to entry
(break-even). Trade becomes a scratch instead of a full loss if it reverses.

Two models compared:
  A) Fixed SL/TP (baseline — 1x ATR SL, 2:1 RR)
  B) Break-Even management (SL moves to entry at IRL touch)
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "QQQ_15m_2yr.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

SWING_LOOKBACK = 20
ATR_PERIOD = 14
ATR_SL_MULT = 1.0
RR_RATIO = 2.0
COMMISSION_RT = 2.40
SLIPPAGE_PCT = 0.005
POSITION_SIZE = 10_000
RETEST_MAX_BARS = 5
MAX_HOLD_BARS = 40

AM_START, AM_END = 570, 690
PM_START, PM_END = 810, 930

# IRL pivot detection — fast pivots (3 bars each side)
IRL_PIVOT_BARS = 3


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    print(f"  📂 Loading: {DATA_PATH}")
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

def compute_atr(df):
    h, l, c = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD).mean()


def detect_fvgs(df):
    h, l = df["High"].values, df["Low"].values
    n = len(df)
    bt, bb, st, sb = [np.full(n, np.nan) for _ in range(4)]
    for i in range(2, n):
        if h[i-2] < l[i]: bt[i], bb[i] = l[i], h[i-2]
        if l[i-2] > h[i]: st[i], sb[i] = l[i-2], h[i]
    return (pd.Series(bt, index=df.index), pd.Series(bb, index=df.index),
            pd.Series(st, index=df.index), pd.Series(sb, index=df.index))


def find_swings(df):
    w = 2 * SWING_LOOKBACK + 1
    rm, rn = df["High"].rolling(w, center=True).max(), df["Low"].rolling(w, center=True).min()
    sh = pd.Series(np.nan, index=df.index); sl = pd.Series(np.nan, index=df.index)
    sh[df["High"] == rm] = df["High"][df["High"] == rm]
    sl[df["Low"] == rn] = df["Low"][df["Low"] == rn]
    return sh, sl


def is_in_session(idx):
    try:
        et = idx.tz_convert("US/Eastern")
    except TypeError:
        et = idx.tz_localize("UTC").tz_convert("US/Eastern")
    mins = et.hour * 60 + et.minute
    return pd.Series((mins >= AM_START) & (mins < AM_END) |
                     (mins >= PM_START) & (mins < PM_END), index=idx)


def find_local_pivots(high_arr: np.ndarray, low_arr: np.ndarray, n_bars: int = IRL_PIVOT_BARS):
    """
    Precompute local pivot highs and lows using fast n-bar lookback.
    A pivot high at bar j: high[j] >= max(high[j-n:j]) AND high[j] >= max(high[j+1:j+n+1])
    Returns arrays of pivot high/low prices (NaN where no pivot).
    """
    n = len(high_arr)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)

    for j in range(n_bars, n - n_bars):
        # Pivot high
        is_ph = True
        for k in range(1, n_bars + 1):
            if high_arr[j] < high_arr[j - k] or high_arr[j] < high_arr[j + k]:
                is_ph = False; break
        if is_ph:
            pivot_highs[j] = high_arr[j]

        # Pivot low
        is_pl = True
        for k in range(1, n_bars + 1):
            if low_arr[j] > low_arr[j - k] or low_arr[j] > low_arr[j + k]:
                is_pl = False; break
        if is_pl:
            pivot_lows[j] = low_arr[j]

    return pivot_highs, pivot_lows


def find_irl_target(
    direction: str,
    entry_bar: int,
    entry_price: float,
    tp_price: float,
    pivot_highs: np.ndarray,
    pivot_lows: np.ndarray,
) -> Optional[float]:
    """
    Find the nearest IRL (Internal Range Liquidity) between entry and TP.

    For LONGS: find the nearest pivot high ABOVE entry but BELOW TP.
    For SHORTS: find the nearest pivot low BELOW entry but ABOVE TP.

    This is the level where we move to break-even.
    Uses only pivots that formed BEFORE the entry bar (no lookahead).
    """
    if direction == "long":
        # Scan backward from entry_bar for recent pivot highs above entry
        candidates = []
        for j in range(entry_bar - 1, max(0, entry_bar - 60), -1):
            ph = pivot_highs[j]
            if not np.isnan(ph) and ph > entry_price and ph < tp_price:
                candidates.append(ph)
        if candidates:
            return min(candidates)  # nearest one above entry
        return None
    else:  # short
        candidates = []
        for j in range(entry_bar - 1, max(0, entry_bar - 60), -1):
            pl = pivot_lows[j]
            if not np.isnan(pl) and pl < entry_price and pl > tp_price:
                candidates.append(pl)
        if candidates:
            return max(candidates)  # nearest one below entry
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_signals(df) -> List[Dict]:
    atr = compute_atr(df)
    bull_top, bull_bot, bear_top, bear_bot = detect_fvgs(df)
    sess = is_in_session(df.index)
    high, low = df["High"].values, df["Low"].values
    signals = []
    armed = {}

    for i in range(30, len(df)):
        atr_val = atr.iloc[i]
        if np.isnan(atr_val) or atr_val <= 0: continue

        if not np.isnan(bull_top.iloc[i]):
            armed[i] = ("long", float(bull_top.iloc[i]), float(bull_bot.iloc[i]), float(atr_val), i)
        if not np.isnan(bear_top.iloc[i]):
            armed[i] = ("short", float(bear_bot.iloc[i]), float(bear_top.iloc[i]), float(atr_val), i)

        to_remove = []
        for fvg_bar, (direction, limit_price, other_edge, atr_arm, armed_bar) in armed.items():
            bars_elapsed = i - armed_bar
            if bars_elapsed > RETEST_MAX_BARS: to_remove.append(fvg_bar); continue
            if bars_elapsed == 0: continue
            if direction == "long" and low[i] < other_edge: to_remove.append(fvg_bar); continue
            if direction == "short" and high[i] > other_edge: to_remove.append(fvg_bar); continue
            if not (low[i] <= limit_price <= high[i]): continue
            if not sess.iloc[i]: continue

            entry = limit_price
            if direction == "long":
                sl = entry - atr_arm * ATR_SL_MULT; risk = entry - sl
                if risk <= 0: continue
                tp = entry + risk * RR_RATIO
            else:
                sl = entry + atr_arm * ATR_SL_MULT; risk = sl - entry
                if risk <= 0: continue
                tp = entry - risk * RR_RATIO

            signals.append({
                "bar_idx": i, "time": df.index[i], "direction": direction,
                "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
                "risk": round(risk, 2),
            })
            to_remove.append(fvg_bar)

        for k in to_remove:
            armed.pop(k, None)

    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION — TWO MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def calc_pnl(entry, exit_price, direction):
    shares = POSITION_SIZE / entry
    gross = (exit_price - entry) * shares if direction == "long" else (entry - exit_price) * shares
    slip = POSITION_SIZE * (SLIPPAGE_PCT / 100) * 2
    return gross - COMMISSION_RT - slip


def simulate_baseline(signals, df):
    """Model A: fixed SL/TP, no break-even."""
    high, low = df["High"].values, df["Low"].values
    n = len(df)
    results = []

    for sig in signals:
        entry, sl, tp, d = sig["entry"], sig["sl"], sig["tp"], sig["direction"]
        start = sig["bar_idx"] + 1
        outcome, exit_price = "open", entry

        for j in range(start, min(start + MAX_HOLD_BARS, n)):
            if d == "long":
                if low[j] <= sl: outcome, exit_price = "loss", sl; break
                if high[j] >= tp: outcome, exit_price = "win", tp; break
            else:
                if high[j] >= sl: outcome, exit_price = "loss", sl; break
                if low[j] <= tp: outcome, exit_price = "win", tp; break

        if outcome == "open":
            last = min(start + MAX_HOLD_BARS - 1, n - 1)
            exit_price = float(df["Close"].iloc[last])
            outcome = "win" if (d == "long" and exit_price > entry) or \
                               (d == "short" and exit_price < entry) else "loss"

        net = calc_pnl(entry, exit_price, d)
        results.append({**sig, "outcome": outcome, "exit_price": round(exit_price, 2),
                        "net_pnl": round(net, 2), "be_triggered": False, "model": "baseline"})
    return results


def simulate_breakeven(signals, df, pivot_highs, pivot_lows):
    """
    Model B: Break-Even management.

    Bar-by-bar simulation:
    1. Check if SL hit (using current SL which may be entry if BE triggered)
    2. Check if TP hit
    3. If neither, check if IRL target hit → move SL to entry (BE)
    """
    high, low = df["High"].values, df["Low"].values
    n = len(df)
    results = []

    for sig in signals:
        entry = sig["entry"]
        original_sl = sig["sl"]
        tp = sig["tp"]
        d = sig["direction"]
        start = sig["bar_idx"] + 1

        # Find IRL target for this trade
        irl = find_irl_target(d, sig["bar_idx"], entry, tp, pivot_highs, pivot_lows)

        current_sl = original_sl
        be_triggered = False
        outcome = "open"
        exit_price = entry

        for j in range(start, min(start + MAX_HOLD_BARS, n)):
            if d == "long":
                # 1. Check SL (priority — worst case first)
                if low[j] <= current_sl:
                    if be_triggered:
                        outcome = "scratch"
                        exit_price = entry  # closed at entry (BE)
                    else:
                        outcome = "loss"
                        exit_price = original_sl
                    break

                # 2. Check TP
                if high[j] >= tp:
                    outcome = "win"
                    exit_price = tp
                    break

                # 3. Check IRL → trigger BE
                if not be_triggered and irl is not None and high[j] >= irl:
                    be_triggered = True
                    current_sl = entry  # move SL to entry

            else:  # short
                if high[j] >= current_sl:
                    if be_triggered:
                        outcome = "scratch"
                        exit_price = entry
                    else:
                        outcome = "loss"
                        exit_price = original_sl
                    break

                if low[j] <= tp:
                    outcome = "win"
                    exit_price = tp
                    break

                if not be_triggered and irl is not None and low[j] <= irl:
                    be_triggered = True
                    current_sl = entry

        # Timeout
        if outcome == "open":
            last = min(start + MAX_HOLD_BARS - 1, n - 1)
            exit_price = float(df["Close"].iloc[last])
            if (d == "long" and exit_price > entry) or (d == "short" and exit_price < entry):
                outcome = "win"
            elif abs(exit_price - entry) < 0.01:
                outcome = "scratch"
            else:
                outcome = "loss"

        net = calc_pnl(entry, exit_price, d)
        results.append({**sig, "outcome": outcome, "exit_price": round(exit_price, 2),
                        "net_pnl": round(net, 2), "be_triggered": be_triggered,
                        "irl_target": round(irl, 2) if irl else None, "model": "breakeven"})
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def max_drawdown(trades):
    if not trades: return 0, 0
    eq = [POSITION_SIZE]
    for t in trades: eq.append(eq[-1] + t["net_pnl"])
    eq = np.array(eq); peak = np.maximum.accumulate(eq); dd = eq - peak
    md = dd.min(); mdp = (md / peak[np.argmin(dd)]) * 100 if peak[np.argmin(dd)] > 0 else 0
    return round(md, 2), round(mdp, 2)


def stats(trades, label):
    if not trades:
        print(f"\n  {label}: 0 trades"); return
    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    scratches = [t for t in trades if t["outcome"] == "scratch"]
    non_scratch = [t for t in trades if t["outcome"] != "scratch"]

    wr = len(wins) / len(trades) * 100
    wr_excl = len(wins) / len(non_scratch) * 100 if non_scratch else 0
    pnl = sum(t["net_pnl"] for t in trades)
    aw = np.mean([t["net_pnl"] for t in wins]) if wins else 0
    al = np.mean([t["net_pnl"] for t in losses]) if losses else 0
    ws = sum(t["net_pnl"] for t in wins); ls = sum(t["net_pnl"] for t in losses)
    pf = abs(ws / ls) if ls != 0 else float("inf")
    md, mdp = max_drawdown(trades)
    longs = [t for t in trades if t["direction"] == "long"]
    shorts = [t for t in trades if t["direction"] == "short"]
    lwr = len([t for t in longs if t["outcome"] == "win"]) / len(longs) * 100 if longs else 0
    swr = len([t for t in shorts if t["outcome"] == "win"]) / len(shorts) * 100 if shorts else 0
    pfs = f"{pf:.2f}" if pf != float("inf") else "∞"

    print(f"\n  {label}")
    print(f"  {'─' * 58}")
    print(f"  Trades:       {len(trades):>6}  │  Win Rate:     {wr:>6.1f}%")
    print(f"  W / L / S:    {len(wins)}W/{len(losses)}L/{len(scratches)}S  │  WR (excl S): {wr_excl:>6.1f}%")
    print(f"  Net P&L:      ${pnl:>9,.2f}  │  Long WR:      {lwr:>6.1f}% ({len(longs)})")
    print(f"  Avg Win:      ${aw:>9,.2f}  │  Short WR:     {swr:>6.1f}% ({len(shorts)})")
    print(f"  Avg Loss:     ${al:>9,.2f}  │  Profit Factor: {pfs:>6}")
    print(f"  Max DD:       ${md:>8,.2f} ({mdp:.1f}%)")


def quarterly(trades, label):
    if not trades: return
    qs = {}
    for t in trades:
        q = f"Q{(t['time'].month-1)//3+1} {t['time'].year}"; qs.setdefault(q, []).append(t)
    print(f"\n  {label}")
    print(f"  {'─' * 62}")
    print(f"  {'Quarter':<10} {'#':>5} {'W':>4} {'L':>4} {'S':>4} {'WR':>7} {'P&L':>11} {'PF':>7}")
    print(f"  {'─' * 62}")
    for q in sorted(qs.keys()):
        qt = qs[q]
        w = len([t for t in qt if t["outcome"] == "win"])
        lo = len([t for t in qt if t["outcome"] == "loss"])
        s = len([t for t in qt if t["outcome"] == "scratch"])
        wr = w/len(qt)*100; pnl = sum(t["net_pnl"] for t in qt)
        ws = sum(t["net_pnl"] for t in qt if t["outcome"] == "win")
        ls = sum(t["net_pnl"] for t in qt if t["outcome"] == "loss")
        pf = abs(ws/ls) if ls != 0 else float("inf")
        pfs = f"{pf:.2f}" if pf != float("inf") else "∞"
        m = "✅" if pnl > 0 else "❌"
        print(f"  {q:<10} {len(qt):>5} {w:>4} {lo:>4} {s:>4} {wr:>6.1f}% ${pnl:>9,.2f} {pfs:>7}  {m}")


def print_report(baseline, breakeven):
    print(f"\n  {'═' * 64}")
    print(f"  📊 FVG BREAK-EVEN EXIT MANAGEMENT — 2-YEAR STRESS TEST")
    print(f"  {'═' * 64}")
    print(f"  Period:           {baseline[0]['time'].strftime('%Y-%m-%d')} → {baseline[-1]['time'].strftime('%Y-%m-%d')}")
    print(f"  IRL Pivot:        {IRL_PIVOT_BARS}-bar pivots (fast detection)")
    print(f"  BE Logic:         SL moves to entry when IRL touched")
    print(f"  Baseline:         Fixed 1x ATR SL, 2:1 RR target")

    stats(baseline, "MODEL A — FIXED SL/TP (baseline)")
    stats(breakeven, "MODEL B — BREAK-EVEN AT IRL ⭐")

    # Detailed BE analysis
    be_trades = [t for t in breakeven if t["be_triggered"]]
    no_be = [t for t in breakeven if not t["be_triggered"]]
    irl_found = [t for t in breakeven if t.get("irl_target") is not None]

    print(f"\n  BREAK-EVEN MECHANICS")
    print(f"  {'─' * 58}")
    print(f"  IRL target found:    {len(irl_found)}/{len(breakeven)} ({len(irl_found)/len(breakeven)*100:.0f}%)")
    print(f"  BE triggered:        {len(be_trades)}/{len(breakeven)} ({len(be_trades)/len(breakeven)*100:.0f}%)")
    print(f"  BE not triggered:    {len(no_be)}")

    if be_trades:
        be_wins = len([t for t in be_trades if t["outcome"] == "win"])
        be_losses = len([t for t in be_trades if t["outcome"] == "loss"])
        be_scratches = len([t for t in be_trades if t["outcome"] == "scratch"])
        be_pnl = sum(t["net_pnl"] for t in be_trades)
        print(f"\n  TRADES WHERE BE TRIGGERED ({len(be_trades)} trades)")
        print(f"  {'─' * 58}")
        print(f"  Outcomes:   {be_wins}W / {be_losses}L / {be_scratches}S")
        print(f"  Net P&L:    ${be_pnl:,.2f}")

        if be_scratches:
            # These scratches would have been losses in baseline
            scratch_pnl_saved = sum(
                abs(t["risk"]) * (POSITION_SIZE / t["entry"]) 
                for t in be_trades if t["outcome"] == "scratch"
            )
            print(f"  Scratches:  {be_scratches} trades saved from full loss")
            print(f"  Est. saved: ~${scratch_pnl_saved:,.2f} in avoided losses")

    # Compare losses between models
    base_losses = [t for t in baseline if t["outcome"] == "loss"]
    be_losses_list = [t for t in breakeven if t["outcome"] == "loss"]
    be_scratches_list = [t for t in breakeven if t["outcome"] == "scratch"]

    base_loss_total = sum(t["net_pnl"] for t in base_losses)
    be_loss_total = sum(t["net_pnl"] for t in be_losses_list)
    scratch_total = sum(t["net_pnl"] for t in be_scratches_list)

    print(f"\n  LOSS COMPARISON")
    print(f"  {'─' * 58}")
    print(f"  Baseline losses:    {len(base_losses)} trades, ${base_loss_total:,.2f}")
    print(f"  BE model losses:    {len(be_losses_list)} trades, ${be_loss_total:,.2f}")
    print(f"  BE model scratches: {len(be_scratches_list)} trades, ${scratch_total:,.2f}")
    print(f"  Loss reduction:     ${abs(be_loss_total) - abs(base_loss_total):,.2f} "
          f"({'less' if abs(be_loss_total) < abs(base_loss_total) else 'more'} losses)")

    # Head-to-head
    a_pnl = sum(t["net_pnl"] for t in baseline)
    b_pnl = sum(t["net_pnl"] for t in breakeven)
    a_wr = len([t for t in baseline if t["outcome"] == "win"]) / len(baseline) * 100
    b_wr = len([t for t in breakeven if t["outcome"] == "win"]) / len(breakeven) * 100
    a_md, _ = max_drawdown(baseline)
    b_md, _ = max_drawdown(breakeven)
    a_scratches = len([t for t in baseline if t["outcome"] == "scratch"])
    b_scratches = len([t for t in breakeven if t["outcome"] == "scratch"])

    print(f"\n  HEAD-TO-HEAD")
    print(f"  {'─' * 64}")
    print(f"  {'Model':<30} {'#':>5} {'WR':>7} {'P&L':>11} {'PF':>7} {'DD':>9} {'S':>4}")
    print(f"  {'─' * 64}")
    a_ws = sum(t["net_pnl"] for t in baseline if t["outcome"] == "win")
    a_ls = sum(t["net_pnl"] for t in baseline if t["outcome"] == "loss")
    b_ws = sum(t["net_pnl"] for t in breakeven if t["outcome"] == "win")
    b_ls = sum(t["net_pnl"] for t in breakeven if t["outcome"] == "loss")
    a_pf = abs(a_ws/a_ls) if a_ls != 0 else float("inf")
    b_pf = abs(b_ws/b_ls) if b_ls != 0 else float("inf")
    a_pfs = f"{a_pf:.2f}"; b_pfs = f"{b_pf:.2f}"
    print(f"  {'A) Fixed SL/TP':<30} {len(baseline):>5} {a_wr:>6.1f}% ${a_pnl:>9,.2f} {a_pfs:>7} ${a_md:>7,.2f} {a_scratches:>4}")
    print(f"  {'B) Break-Even ⭐':<30} {len(breakeven):>5} {b_wr:>6.1f}% ${b_pnl:>9,.2f} {b_pfs:>7} ${b_md:>7,.2f} {b_scratches:>4}")
    delta_pnl = b_pnl - a_pnl
    delta_dd = abs(b_md) - abs(a_md)
    print(f"\n  Delta P&L:   ${delta_pnl:+,.2f}")
    print(f"  Delta DD:    ${delta_dd:+,.2f} ({'tighter' if delta_dd < 0 else 'wider'})")

    # Quarterly
    quarterly(breakeven, "MODEL B — QUARTERLY WALK-FORWARD")

    # Equity
    for lbl, tr in [("A) Fixed SL/TP", baseline), ("B) Break-Even", breakeven)]:
        eq = [POSITION_SIZE]
        for t in tr: eq.append(eq[-1] + t["net_pnl"])
        print(f"\n  EQUITY — {lbl}: ${eq[0]:,.0f} → ${eq[-1]:,.0f} "
              f"(peak ${max(eq):,.0f}, trough ${min(eq):,.0f})")

    print(f"\n  {'═' * 64}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n  {'═' * 64}")
    print(f"  🔬 FVG Break-Even Exit Management — 2-Year Stress Test")
    print(f"  {'═' * 64}")

    df = load_data()
    if len(df) < 200:
        print("  ❌ Not enough data"); sys.exit(1)

    print(f"  🔍 Computing pivots for IRL targets...")
    pivot_highs, pivot_lows = find_local_pivots(df["High"].values, df["Low"].values)
    n_ph = np.count_nonzero(~np.isnan(pivot_highs))
    n_pl = np.count_nonzero(~np.isnan(pivot_lows))
    print(f"  📊 {n_ph} pivot highs, {n_pl} pivot lows detected")

    print(f"  ⚡ Generating signals...")
    signals = generate_signals(df)
    print(f"  📊 {len(signals)} signals")

    print(f"  ⚡ Running Model A (baseline)...")
    baseline = simulate_baseline(signals, df)

    print(f"  ⚡ Running Model B (break-even at IRL)...")
    breakeven = simulate_breakeven(signals, df, pivot_highs, pivot_lows)

    print_report(baseline, breakeven)


if __name__ == "__main__":
    main()
