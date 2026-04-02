#!/usr/bin/env python3
"""
Forex FVG Continuation V3 — ICT Masterclass Architecture
=========================================================
Builds on V2 (PF 4.65) with three advanced ICT filters:

  1. Draw on Liquidity (DOL) — directional bias from untaken liquidity pools
  2. Sponsored Delivery — displacement quality filter (close position + body ratio)
  3. Premium/Discount Zone — only buy in discount, sell in premium

Stage A: All filters backtestable on 1h data (2yr).
Stage B: LTF iFVG trigger (1m) — live-only, not backtestable.

Usage:
    python3 forex_v3_backtest.py                    # full 4-pair backtest
    python3 forex_v3_backtest.py --pair EURUSD=X    # single pair
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

PAIRS = ["EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X"]
PAIR_LABELS = {"EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD",
               "JPY=X": "USD/JPY", "AUDUSD=X": "AUD/USD"}

# Risk management
RISK_PER_TRADE = 100.0
ATR_SL_MULT    = 0.5
RR_RATIO       = 2.0

# FVG parameters
RETEST_MAX_BARS = 20
MAX_HOLD_BARS   = 40
IRL_PIVOT_BARS  = 5

# Killzones (minutes since midnight ET)
LONDON_START, LONDON_END = 180, 300    # 03:00–05:00
NY_START, NY_END         = 480, 720    # 08:00–12:00
LATE_START, LATE_END     = 840, 900    # 14:00–15:00

# Costs
SPREAD_PIPS = 1.5
PIP_VALUE_STD = 0.0001
PIP_VALUE_JPY = 0.01
COMMISSION_PER_LOT = 3.0
LOT_SIZE = 100_000

# ── V3 Filter Parameters ──
DOL_LOOKBACK       = 20    # bars to look back for swing high/low (liquidity pools)
DOL_SWING_BARS     = 5     # local pivot detection for DOL
SPONSORED_CLOSE_PCT = 0.70  # candle close must be in top/bottom 70% of range
PREMIUM_DISCOUNT_PCT = 0.50 # dealing range midpoint for zone classification
DEALING_RANGE_BARS  = 24    # bars to compute dealing range (24h for 1h bars)


# ════════════════════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════════════════════

def fetch_data(pair: str, tf: str = "1h") -> pd.DataFrame:
    import yfinance as yf
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{pair.replace('=X','').replace('=','')}_{tf}_v3_cache.csv")

    if os.path.exists(cache_file):
        import time
        if time.time() - os.path.getmtime(cache_file) < 3600:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if len(df) > 100:
                print(f"  [cached] {PAIR_LABELS.get(pair, pair)} {tf}: {len(df)} bars")
                return df

    period = "2y" if tf == "1h" else "60d"
    print(f"  [downloading] {PAIR_LABELS.get(pair, pair)} {tf} {period}...")
    df = yf.download(pair, period=period, interval=tf, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.to_csv(cache_file)
    print(f"  [ok] {PAIR_LABELS.get(pair, pair)}: {len(df)} bars")
    return df


# ════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ════════════════════════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def detect_fvgs(df: pd.DataFrame):
    high, low = df["High"].values, df["Low"].values
    n = len(df)
    bt, bb, srt, srb = [np.full(n, np.nan) for _ in range(4)]
    for i in range(2, n):
        if high[i-2] < low[i]:
            bt[i-1], bb[i-1] = low[i], high[i-2]
        if low[i-2] > high[i]:
            srt[i-1], srb[i-1] = low[i-2], high[i]
    return bt, bb, srt, srb


def find_local_pivots(high, low, lookback=IRL_PIVOT_BARS):
    n = len(high)
    ph, pl = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(lookback, n - lookback):
        wh = high[max(0, i-lookback):i+lookback+1]
        if high[i] == wh.max(): ph[i] = high[i]
        wl = low[max(0, i-lookback):i+lookback+1]
        if low[i] == wl.min(): pl[i] = low[i]
    return ph, pl


def find_irl_target(direction, entry, tp, bar_idx, pivot_highs, pivot_lows):
    look_start, look_end = max(0, bar_idx - 100), bar_idx
    if direction == "long":
        c = [pivot_highs[j] for j in range(look_start, look_end)
             if not np.isnan(pivot_highs[j]) and entry < pivot_highs[j] < tp]
        return min(c) if c else None
    else:
        c = [pivot_lows[j] for j in range(look_start, look_end)
             if not np.isnan(pivot_lows[j]) and tp < pivot_lows[j] < entry]
        return max(c) if c else None


# ════════════════════════════════════════════════════════════════════════════
# V3 FILTERS
# ════════════════════════════════════════════════════════════════════════════

def compute_dol(df: pd.DataFrame, lookback: int = DOL_LOOKBACK, swing: int = DOL_SWING_BARS):
    """
    Draw on Liquidity (DOL) Detection.
    
    Scans for untaken swing highs (buy-side liquidity) and swing lows
    (sell-side liquidity). The DOL is the nearest untaken pool.
    
    Returns arrays:
      dol_direction: 1 = DOL is above (bias long), -1 = DOL is below (bias short), 0 = ambiguous
      dol_distance:  price distance to nearest untaken liquidity pool
    """
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    n = len(df)

    dol_dir = np.zeros(n, dtype=int)
    dol_dist = np.full(n, np.nan)

    for i in range(lookback, n):
        # Find swing highs and lows in the lookback window
        untaken_highs = []
        untaken_lows = []

        for j in range(max(swing, i - lookback), i - swing):
            # Swing high check
            wh = high[max(0, j-swing):j+swing+1]
            if high[j] == wh.max():
                # Check if this high has been taken (price traded above it)
                taken = False
                for k in range(j+1, i+1):
                    if high[k] > high[j]:
                        taken = True
                        break
                if not taken:
                    untaken_highs.append(high[j])

            # Swing low check
            wl = low[max(0, j-swing):j+swing+1]
            if low[j] == wl.min():
                taken = False
                for k in range(j+1, i+1):
                    if low[k] < low[j]:
                        taken = True
                        break
                if not taken:
                    untaken_lows.append(low[j])

        price = close[i]

        # Find nearest untaken pools
        nearest_high = min(untaken_highs, key=lambda x: abs(x - price)) if untaken_highs else None
        nearest_low = min(untaken_lows, key=lambda x: abs(x - price)) if untaken_lows else None

        if nearest_high is not None and nearest_low is not None:
            dist_high = nearest_high - price  # positive if above
            dist_low = price - nearest_low     # positive if below

            if dist_high < dist_low:
                dol_dir[i] = 1   # DOL above → bias long
                dol_dist[i] = dist_high
            else:
                dol_dir[i] = -1  # DOL below → bias short
                dol_dist[i] = dist_low
        elif nearest_high is not None:
            dol_dir[i] = 1
            dol_dist[i] = nearest_high - price
        elif nearest_low is not None:
            dol_dir[i] = -1
            dol_dist[i] = price - nearest_low
        else:
            dol_dir[i] = 0

    return dol_dir, dol_dist


def is_sponsored_delivery(open_val, high_val, low_val, close_val, 
                          atr_val, direction, close_pct=SPONSORED_CLOSE_PCT):
    """
    Sponsored Delivery Filter.
    
    Checks if a displacement candle shows genuine institutional sponsorship:
    1. Body >= 1.0x ATR (already checked before calling this)
    2. Close is in the top/bottom portion of the candle range (not a wick rejection)
    3. For bullish: close near high. For bearish: close near low.
    """
    candle_range = high_val - low_val
    if candle_range <= 0:
        return False

    if direction == "long":
        # Bullish: close should be in top portion of candle
        close_position = (close_val - low_val) / candle_range
        return close_position >= close_pct
    else:
        # Bearish: close should be in bottom portion of candle
        close_position = (high_val - close_val) / candle_range
        return close_position >= close_pct


def compute_dealing_range(df: pd.DataFrame, lookback: int = DEALING_RANGE_BARS):
    """
    Premium/Discount Zone Classification.
    
    Uses a rolling dealing range (lookback bars high/low) to determine
    if current price is in premium (above 50%) or discount (below 50%).
    
    Returns:
      zone: array of floats — 0.0 = at range low, 1.0 = at range high
      range_high: rolling high
      range_low: rolling low
    """
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    n = len(df)

    zone = np.full(n, 0.5)
    r_high = np.full(n, np.nan)
    r_low = np.full(n, np.nan)

    for i in range(lookback, n):
        rh = np.max(high[i-lookback:i])
        rl = np.min(low[i-lookback:i])
        r_high[i] = rh
        r_low[i] = rl
        rng = rh - rl
        if rng > 0:
            zone[i] = (close[i] - rl) / rng

    return zone, r_high, r_low


# ════════════════════════════════════════════════════════════════════════════
# SESSION FILTERS
# ════════════════════════════════════════════════════════════════════════════

def _to_et_minutes(idx):
    try:
        et = idx.tz_convert("US/Eastern")
    except TypeError:
        et = idx.tz_localize("UTC").tz_convert("US/Eastern")
    return np.array(et.hour * 60 + et.minute)


def is_in_killzone(idx):
    mins = _to_et_minutes(idx)
    return pd.Series(
        ((mins >= LONDON_START) & (mins < LONDON_END)) |
        ((mins >= NY_START) & (mins < NY_END)) |
        ((mins >= LATE_START) & (mins < LATE_END)),
        index=idx
    )


# ════════════════════════════════════════════════════════════════════════════
# TRADE SIMULATION
# ════════════════════════════════════════════════════════════════════════════

def simulate_breakeven(signals, df, pivot_highs, pivot_lows):
    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    results = []

    for sig in signals:
        entry, sl, tp = sig["entry"], sig["sl"], sig["tp"]
        direction, bar_idx, risk = sig["direction"], sig["bar_idx"], sig["risk"]
        if risk <= 0: continue

        units = RISK_PER_TRADE / risk
        irl = find_irl_target(direction, entry, tp, bar_idx, pivot_highs, pivot_lows)
        be_triggered, exit_price, exit_bar, outcome = False, None, None, None

        for j in range(bar_idx + 1, min(bar_idx + MAX_HOLD_BARS + 1, len(df))):
            h, l = high[j], low[j]
            if direction == "long":
                current_sl = entry if be_triggered else sl
                if l <= current_sl:
                    exit_price, exit_bar = current_sl, j
                    outcome = "scratch" if be_triggered else "loss"; break
                if irl and not be_triggered and h >= irl: be_triggered = True
                if h >= tp:
                    exit_price, exit_bar, outcome = tp, j, "win"; break
            else:
                current_sl = entry if be_triggered else sl
                if h >= current_sl:
                    exit_price, exit_bar = current_sl, j
                    outcome = "scratch" if be_triggered else "loss"; break
                if irl and not be_triggered and l <= irl: be_triggered = True
                if l <= tp:
                    exit_price, exit_bar, outcome = tp, j, "win"; break

        if exit_price is None:
            exit_bar = min(bar_idx + MAX_HOLD_BARS, len(df) - 1)
            exit_price = close[exit_bar]
            outcome = "win" if (exit_price > entry if direction == "long" else exit_price < entry) else "loss"

        gross = (exit_price - entry) * units if direction == "long" else (entry - exit_price) * units
        lots = units / LOT_SIZE
        commission = lots * COMMISSION_PER_LOT
        net = -commission if outcome == "scratch" else gross - commission

        results.append({
            "bar_idx": bar_idx, "time": sig["time"], "direction": direction,
            "entry": entry, "sl": sl, "tp": tp, "exit_price": exit_price,
            "outcome": outcome, "be_triggered": be_triggered, "net_pnl": round(net, 2),
            "dol_aligned": sig.get("dol_aligned", False),
            "sponsored": sig.get("sponsored", False),
            "in_discount": sig.get("in_discount", False),
        })
    return results


# ════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ════════════════════════════════════════════════════════════════════════════

def generate_signals(df, use_killzone=True, use_dol=False, use_sponsored=False,
                     use_pd_zone=False):
    """Generate FVG signals with optional V3 filters."""
    atr = compute_atr(df)
    bt, bb, srt, srb = detect_fvgs(df)
    sess = is_in_killzone(df.index) if use_killzone else pd.Series(True, index=df.index)

    high, low, open_, close = df["High"].values, df["Low"].values, df["Open"].values, df["Close"].values

    # Precompute V3 filters
    dol_dir, dol_dist = compute_dol(df) if use_dol else (np.zeros(len(df)), np.full(len(df), np.nan))
    zone, _, _ = compute_dealing_range(df) if use_pd_zone else (np.full(len(df), 0.5), None, None)

    pip_val = PIP_VALUE_JPY if "JPY" in str(df.columns) or "JPY" in str(getattr(df, 'name', '')) else PIP_VALUE_STD
    spread_penalty = SPREAD_PIPS * pip_val

    signals, armed = [], {}

    for i in range(30, len(df)):
        atr_val = atr.iloc[i]
        if np.isnan(atr_val) or atr_val <= 0: continue

        body = abs(close[i] - open_[i])
        if body < atr_val * 1.0: continue
        if not sess.iloc[i]: continue

        # ── V3: Sponsored Delivery check ──
        is_bull_candle = close[i] > open_[i]
        sponsored = True
        if use_sponsored:
            candle_dir = "long" if is_bull_candle else "short"
            sponsored = is_sponsored_delivery(open_[i], high[i], low[i], close[i],
                                              atr_val, candle_dir)
            if not sponsored:
                continue

        # ── V3: DOL alignment check ──
        dol_aligned = True
        if use_dol:
            if not np.isnan(bt[i]):  # bullish FVG
                dol_aligned = dol_dir[i] >= 0  # DOL above or neutral = OK for longs
            elif not np.isnan(srt[i]):  # bearish FVG
                dol_aligned = dol_dir[i] <= 0  # DOL below or neutral = OK for shorts
            if not dol_aligned:
                continue

        # ── V3: Premium/Discount zone check ──
        in_correct_zone = True
        if use_pd_zone:
            if not np.isnan(bt[i]):  # bullish FVG → must be in discount
                in_correct_zone = zone[i] < PREMIUM_DISCOUNT_PCT
            elif not np.isnan(srt[i]):  # bearish FVG → must be in premium
                in_correct_zone = zone[i] > PREMIUM_DISCOUNT_PCT
            if not in_correct_zone:
                continue

        # Arm FVGs
        if not np.isnan(bt[i]):
            armed[i] = ("long", float(bt[i]), float(bb[i]), float(atr_val), i,
                        dol_aligned, sponsored, zone[i] < 0.5)
        if not np.isnan(srt[i]):
            armed[i] = ("short", float(srb[i]), float(srt[i]), float(atr_val), i,
                        dol_aligned, sponsored, zone[i] > 0.5)

        # Scan armed for retests
        to_remove = []
        for fb, (d, lp, oe, aa, ab, da, sp, iz) in armed.items():
            elapsed = i - ab
            if elapsed > RETEST_MAX_BARS: to_remove.append(fb); continue
            if elapsed == 0: continue
            if d == "long" and low[i] < oe: to_remove.append(fb); continue
            if d == "short" and high[i] > oe: to_remove.append(fb); continue
            if not (low[i] <= lp <= high[i]): continue

            if d == "long":
                entry = lp + spread_penalty
                sl = entry - aa * ATR_SL_MULT
                risk = entry - sl
                if risk <= 0: continue
                tp = entry + risk * RR_RATIO
            else:
                entry = lp - spread_penalty
                sl = entry + aa * ATR_SL_MULT
                risk = sl - entry
                if risk <= 0: continue
                tp = entry - risk * RR_RATIO

            signals.append({
                "bar_idx": i, "time": df.index[i], "direction": d,
                "entry": round(entry, 6), "sl": round(sl, 6), "tp": round(tp, 6),
                "risk": round(risk, 6), "dol_aligned": da, "sponsored": sp,
                "in_discount": iz,
            })
            to_remove.append(fb)
        for k in to_remove: armed.pop(k, None)

    return signals


# ════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ════════════════════════════════════════════════════════════════════════════

def max_drawdown(trades):
    if not trades: return 0.0, 0.0
    eq, peak, mdd = 10000.0, 10000.0, 0.0
    for t in trades:
        eq += t["net_pnl"]
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return mdd, mdd / 10000.0 * 100


def stats(trades, label):
    if not trades:
        print(f"  {label}: 0 trades"); return (0, 0, 0, 0, 0)
    w = [t for t in trades if t["outcome"] == "win"]
    l = [t for t in trades if t["outcome"] == "loss"]
    s = [t for t in trades if t["outcome"] == "scratch"]
    wr = len(w) / len(trades) * 100
    pnl = sum(t["net_pnl"] for t in trades)
    ws = sum(t["net_pnl"] for t in w)
    ls = sum(t["net_pnl"] for t in l)
    pf = abs(ws / ls) if ls != 0 else float("inf")
    md, mdp = max_drawdown(trades)
    pfs = f"{pf:.2f}" if pf != float("inf") else "inf"

    print(f"\n  {label}")
    print(f"  {'─'*65}")
    print(f"  Trades: {len(trades):>5} | W/L/S: {len(w)}W/{len(l)}L/{len(s)}S | WR: {wr:.1f}%")
    print(f"  P&L: ${pnl:>10,.2f} | PF: {pfs:>6} | Max DD: ${md:>8,.2f} ({mdp:.1f}%)")
    return len(trades), wr, pnl, pf, md


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Forex V3 ICT Masterclass Backtest")
    parser.add_argument("--pair", help="Single pair (e.g., EURUSD=X)")
    args = parser.parse_args()

    sep = "=" * 72
    print(f"\n  {sep}")
    print(f"  Forex FVG Continuation V3 — ICT Masterclass")
    print(f"  DOL + Sponsored Delivery + Premium/Discount Zone")
    print(f"  {sep}")

    pairs = [args.pair] if args.pair else PAIRS
    all_results = {}

    # Define model configs
    models = {
        "V2 (Killzone only)":          {"kz": True,  "dol": False, "sp": False, "pd": False},
        "V3a (+ DOL)":                 {"kz": True,  "dol": True,  "sp": False, "pd": False},
        "V3b (+ Sponsored)":           {"kz": True,  "dol": False, "sp": True,  "pd": False},
        "V3c (+ Prem/Disc)":           {"kz": True,  "dol": False, "sp": False, "pd": True},
        "V3d (DOL + Sponsored)":       {"kz": True,  "dol": True,  "sp": True,  "pd": False},
        "V3 FULL (DOL+SP+PD)":         {"kz": True,  "dol": True,  "sp": True,  "pd": True},
    }

    for pair in pairs:
        label = PAIR_LABELS.get(pair, pair)
        print(f"\n  {'─'*72}")
        print(f"  {label}")
        print(f"  {'─'*72}")

        df = fetch_data(pair, "1h")
        if len(df) < 200:
            print(f"  Insufficient data"); continue

        # Detect pair type for pip calculation
        df.name = pair

        print(f"  Period: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')} ({len(df)} bars)")

        pivot_highs, pivot_lows = find_local_pivots(df["High"].values, df["Low"].values)

        pair_results = {}

        for model_name, cfg in models.items():
            sigs = generate_signals(df, use_killzone=cfg["kz"], use_dol=cfg["dol"],
                                    use_sponsored=cfg["sp"], use_pd_zone=cfg["pd"])
            res = simulate_breakeven(sigs, df, pivot_highs, pivot_lows)
            pair_results[model_name] = res
            stats(res, f"{label} — {model_name}")

        all_results[label] = pair_results

    # Combined summary
    print(f"\n\n  {sep}")
    print(f"  HEAD-TO-HEAD COMPARISON (All Pairs Combined)")
    print(f"  {sep}")
    print(f"  {'Model':<28} {'#':>5} {'WR%':>6} {'P&L':>10} {'PF':>6} {'MaxDD':>9}")
    print(f"  {'─'*72}")

    for model_name in models.keys():
        all_trades = []
        for label, pair_res in all_results.items():
            all_trades.extend(pair_res.get(model_name, []))

        if not all_trades:
            print(f"  {model_name:<28}     0"); continue

        w = len([t for t in all_trades if t["outcome"] == "win"])
        n = len(all_trades)
        wr = w / n * 100
        pnl = sum(t["net_pnl"] for t in all_trades)
        ws = sum(t["net_pnl"] for t in all_trades if t["outcome"] == "win")
        ls = sum(t["net_pnl"] for t in all_trades if t["outcome"] == "loss")
        pf = abs(ws / ls) if ls != 0 else float("inf")
        pfs = f"{pf:.2f}" if pf != float("inf") else "inf"
        md, _ = max_drawdown(all_trades)

        print(f"  {model_name:<28} {n:>5} {wr:>5.1f}% ${pnl:>8,.2f} {pfs:>6} ${md:>7,.2f}")

    print(f"\n  {sep}")
    print(f"  Done.")
    print(f"  {sep}\n")


if __name__ == "__main__":
    main()
