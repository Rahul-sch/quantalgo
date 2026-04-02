#!/usr/bin/env python3
"""
Goldbach Partition Backtest — FVG + Break-Even + Partition + Liquidity Sweep
=============================================================================
Adds two new filters (from Goldbach partition theory) to the existing
FVG continuation + break-even model.

Three models compared:
  A) Baseline:       Fixed SL/TP (1x ATR SL, 2:1 RR), no BE, no new filters
  B) Break-Even:     SL moves to entry when IRL touched, no new filters
  C) Goldbach:       Break-Even + Partition Filter + Liquidity Sweep Filter ⭐

NEW FILTERS:
  1. PARTITION FILTER
     - Dealing range = rolling 26-bar H/L as proxy for prior session
     - Equilibrium (EQ) = midpoint of dealing range
     - LONG  → FVG must be in DISCOUNT zone (FVG midpoint < EQ)
     - SHORT → FVG must be in PREMIUM zone  (FVG midpoint > EQ)

  2. LIQUIDITY SWEEP FILTER
     - Equal highs/lows = swing clusters within 0.1% of each other
     - Lookback: 20 bars for cluster detection
     - LONG  → equal-low cluster (below entry, within dealing range) must have been
               swept (price dipped below cluster level) within last 5 bars
     - SHORT → equal-high cluster (above entry, within dealing range) must have been
               swept (price popped above cluster level) within last 5 bars
"""

import sys
import os
import importlib.util
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

# ─── Dynamically import shared helpers from fvg_breakeven_backtest ────────────
_SIBLING = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fvg_breakeven_backtest.py")

def _load_sibling():
    spec = importlib.util.spec_from_file_location("fvg_be", _SIBLING)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_fvg = _load_sibling()

# Shared functions
load_data         = _fvg.load_data
compute_atr       = _fvg.compute_atr
detect_fvgs       = _fvg.detect_fvgs
is_in_session     = _fvg.is_in_session
find_local_pivots = _fvg.find_local_pivots
find_irl_target   = _fvg.find_irl_target
calc_pnl          = _fvg.calc_pnl
max_drawdown      = _fvg.max_drawdown

# Shared config (mirrors fvg_breakeven_backtest so we stay in sync)
ATR_SL_MULT     = _fvg.ATR_SL_MULT
RR_RATIO        = _fvg.RR_RATIO
POSITION_SIZE   = _fvg.POSITION_SIZE
RETEST_MAX_BARS = _fvg.RETEST_MAX_BARS
MAX_HOLD_BARS   = _fvg.MAX_HOLD_BARS

# ─── Goldbach-specific config ─────────────────────────────────────────────────
DEALING_RANGE_LOOKBACK = 26    # rolling bars for prior-session H/L proxy
EQL_CLUSTER_PCT        = 0.001 # 0.1% tolerance for equal-H/L cluster
EQL_LOOKBACK           = 20    # bars to look back for equal H/L clusters
SWEEP_LOOKBACK         = 5     # bars to look back for a confirmed sweep


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDBACH INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dealing_range(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rolling 26-bar dealing range as a proxy for the prior session's H/L.
    Shifted by 1 bar so there is no lookahead.
    Returns: (dr_high, dr_low, equilibrium) as numpy arrays.
    """
    dr_high = df["High"].shift(1).rolling(DEALING_RANGE_LOOKBACK).max().values
    dr_low  = df["Low"].shift(1).rolling(DEALING_RANGE_LOOKBACK).min().values
    eq      = (dr_high + dr_low) / 2.0
    return dr_high, dr_low, eq


def partition_filter(
    direction: str,
    fvg_mid:   float,
    eq:        float,
) -> Tuple[bool, str]:
    """
    Partition filter: FVGs must be in the correct zone.
    LONG  → fvg_mid must be BELOW EQ (discount zone).
    SHORT → fvg_mid must be ABOVE EQ (premium zone).
    Returns (passes, reason_string).
    """
    if np.isnan(eq):
        return False, "EQ_NaN"
    if direction == "long":
        if fvg_mid < eq:
            return True, f"discount (fvg={fvg_mid:.2f} < eq={eq:.2f})"
        return False, f"premium (fvg={fvg_mid:.2f} >= eq={eq:.2f})"
    else:
        if fvg_mid > eq:
            return True, f"premium (fvg={fvg_mid:.2f} > eq={eq:.2f})"
        return False, f"discount (fvg={fvg_mid:.2f} <= eq={eq:.2f})"


def find_equal_lows(
    low_arr:  np.ndarray,
    bar_idx:  int,
    ref_price: float,
    lookback: int = EQL_LOOKBACK,
    tol_pct:  float = EQL_CLUSTER_PCT,
) -> List[float]:
    """
    Find equal-low clusters BELOW ref_price in the last `lookback` bars before bar_idx.
    Two lows are "equal" if within tol_pct of each other (cluster mean).
    Returns list of cluster representative prices.
    """
    start  = max(0, bar_idx - lookback)
    window = low_arr[start:bar_idx]
    if len(window) < 2:
        return []

    sorted_lows = np.sort(window)
    clusters    = []
    cur         = [sorted_lows[0]]

    for v in sorted_lows[1:]:
        mean = np.mean(cur)
        if abs(v - mean) / (mean + 1e-9) <= tol_pct:
            cur.append(v)
        else:
            if len(cur) >= 2:
                clusters.append(float(np.mean(cur)))
            cur = [v]
    if len(cur) >= 2:
        clusters.append(float(np.mean(cur)))

    return [c for c in clusters if c < ref_price]


def find_equal_highs(
    high_arr:  np.ndarray,
    bar_idx:   int,
    ref_price: float,
    lookback:  int = EQL_LOOKBACK,
    tol_pct:   float = EQL_CLUSTER_PCT,
) -> List[float]:
    """
    Find equal-high clusters ABOVE ref_price in the last `lookback` bars before bar_idx.
    Returns list of cluster representative prices.
    """
    start  = max(0, bar_idx - lookback)
    window = high_arr[start:bar_idx]
    if len(window) < 2:
        return []

    sorted_highs = np.sort(window)[::-1]
    clusters     = []
    cur          = [sorted_highs[0]]

    for v in sorted_highs[1:]:
        mean = np.mean(cur)
        if abs(v - mean) / (mean + 1e-9) <= tol_pct:
            cur.append(v)
        else:
            if len(cur) >= 2:
                clusters.append(float(np.mean(cur)))
            cur = [v]
    if len(cur) >= 2:
        clusters.append(float(np.mean(cur)))

    return [c for c in clusters if c > ref_price]


def liquidity_sweep_filter(
    direction:      str,
    bar_idx:        int,
    entry_price:    float,
    high_arr:       np.ndarray,
    low_arr:        np.ndarray,
    dr_low:         float,
    dr_high:        float,
    lookback:       int = EQL_LOOKBACK,
    sweep_lookback: int = SWEEP_LOOKBACK,
    tol_pct:        float = EQL_CLUSTER_PCT,
) -> Tuple[bool, str]:
    """
    Liquidity sweep filter.

    LONG : equal-low cluster (within dealing range, below entry) must have been
           swept (price < cluster level) within the last sweep_lookback bars.
    SHORT: equal-high cluster (within dealing range, above entry) must have been
           swept (price > cluster level) within the last sweep_lookback bars.

    Returns (passes, reason_string).
    """
    if direction == "long":
        eq_levels = find_equal_lows(low_arr, bar_idx, entry_price, lookback, tol_pct)
        # Filter to within dealing range
        if not (np.isnan(dr_low) or np.isnan(dr_high)):
            eq_levels = [lv for lv in eq_levels if dr_low <= lv <= dr_high]

        if not eq_levels:
            return False, "no_equal_lows_found"

        sweep_start  = max(0, bar_idx - sweep_lookback)
        recent_lows  = low_arr[sweep_start:bar_idx]

        if len(recent_lows) == 0:
            return False, "no_recent_bars"

        min_recent = float(np.min(recent_lows))
        for lv in eq_levels:
            if min_recent < lv:
                return True, f"eq_low_swept@{lv:.2f}"

        lvs = [round(x, 2) for x in eq_levels]
        return False, f"eq_lows_not_swept levels={lvs} min_recent={min_recent:.2f}"

    else:  # short
        eq_levels = find_equal_highs(high_arr, bar_idx, entry_price, lookback, tol_pct)
        if not (np.isnan(dr_low) or np.isnan(dr_high)):
            eq_levels = [lv for lv in eq_levels if dr_low <= lv <= dr_high]

        if not eq_levels:
            return False, "no_equal_highs_found"

        sweep_start   = max(0, bar_idx - sweep_lookback)
        recent_highs  = high_arr[sweep_start:bar_idx]

        if len(recent_highs) == 0:
            return False, "no_recent_bars"

        max_recent = float(np.max(recent_highs))
        for lv in eq_levels:
            if max_recent > lv:
                return True, f"eq_high_swept@{lv:.2f}"

        lvs = [round(x, 2) for x in eq_levels]
        return False, f"eq_highs_not_swept levels={lvs} max_recent={max_recent:.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION (shared core, includes FVG midpoint for partition check)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_signals(df: pd.DataFrame) -> List[Dict]:
    """
    Generate all FVG signals with Goldbach context fields.
    Mirrors fvg_breakeven_backtest.generate_signals() exactly, plus fvg_mid.
    """
    atr = compute_atr(df)
    bull_top, bull_bot, bear_top, bear_bot = detect_fvgs(df)
    sess  = is_in_session(df.index)
    high  = df["High"].values
    low   = df["Low"].values
    armed: Dict[int, Tuple] = {}
    signals: List[Dict] = []

    for i in range(30, len(df)):
        atr_val = atr.iloc[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        if not np.isnan(bull_top.iloc[i]):
            # Bull FVG: top = bull_top (FVG gap top), bottom = bull_bot
            armed[i] = ("long",  float(bull_top.iloc[i]), float(bull_bot.iloc[i]), float(atr_val), i)
        if not np.isnan(bear_top.iloc[i]):
            # Bear FVG: entry at bear_bot (bottom edge), other edge = bear_top
            armed[i] = ("short", float(bear_bot.iloc[i]), float(bear_top.iloc[i]), float(atr_val), i)

        to_remove = []
        for fvg_bar, (direction, limit_price, other_edge, atr_arm, armed_bar) in armed.items():
            bars_elapsed = i - armed_bar
            if bars_elapsed > RETEST_MAX_BARS:
                to_remove.append(fvg_bar); continue
            if bars_elapsed == 0:
                continue
            # Invalidated if price closes through the opposite edge
            if direction == "long"  and low[i]  < other_edge: to_remove.append(fvg_bar); continue
            if direction == "short" and high[i] > other_edge:  to_remove.append(fvg_bar); continue
            # Wait for price to touch the entry edge
            if not (low[i] <= limit_price <= high[i]):
                continue
            if not sess.iloc[i]:
                continue

            entry = limit_price
            if direction == "long":
                sl   = entry - atr_arm * ATR_SL_MULT
                risk = entry - sl
                if risk <= 0: continue
                tp   = entry + risk * RR_RATIO
                # FVG midpoint: between limit_price (top) and other_edge (bottom)
                fvg_mid = (limit_price + other_edge) / 2.0
            else:
                sl   = entry + atr_arm * ATR_SL_MULT
                risk = sl - entry
                if risk <= 0: continue
                tp   = entry - risk * RR_RATIO
                # FVG midpoint: between other_edge (top) and limit_price (bottom)
                fvg_mid = (other_edge + limit_price) / 2.0

            signals.append({
                "bar_idx":   i,
                "time":      df.index[i],
                "direction": direction,
                "entry":     round(entry,   2),
                "sl":        round(sl,      2),
                "tp":        round(tp,      2),
                "risk":      round(risk,    2),
                "fvg_mid":   round(fvg_mid, 2),
            })
            to_remove.append(fvg_bar)

        for k in to_remove:
            armed.pop(k, None)

    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATORS
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_baseline(signals: List[Dict], df: pd.DataFrame) -> List[Dict]:
    """Model A: fixed SL/TP, no BE, no Goldbach filters."""
    high_arr = df["High"].values
    low_arr  = df["Low"].values
    n        = len(df)
    results  = []

    for sig in signals:
        entry, sl, tp, d = sig["entry"], sig["sl"], sig["tp"], sig["direction"]
        start   = sig["bar_idx"] + 1
        outcome = "open"
        exit_p  = entry

        for j in range(start, min(start + MAX_HOLD_BARS, n)):
            if d == "long":
                if low_arr[j]  <= sl: outcome, exit_p = "loss", sl; break
                if high_arr[j] >= tp: outcome, exit_p = "win",  tp; break
            else:
                if high_arr[j] >= sl: outcome, exit_p = "loss", sl; break
                if low_arr[j]  <= tp: outcome, exit_p = "win",  tp; break

        if outcome == "open":
            last   = min(start + MAX_HOLD_BARS - 1, n - 1)
            exit_p = float(df["Close"].iloc[last])
            outcome = ("win" if (d == "long" and exit_p > entry) or
                                 (d == "short" and exit_p < entry)
                       else "loss")

        net = calc_pnl(entry, exit_p, d)
        results.append({**sig,
                        "outcome":      outcome,
                        "exit_price":   round(exit_p, 2),
                        "net_pnl":      round(net, 2),
                        "be_triggered": False,
                        "model":        "A_baseline"})
    return results


def _run_be_simulation(
    sig:         Dict,
    df:          pd.DataFrame,
    high_arr:    np.ndarray,
    low_arr:     np.ndarray,
    pivot_highs: np.ndarray,
    pivot_lows:  np.ndarray,
    model_label: str,
    extra_fields: Optional[Dict] = None,
) -> Dict:
    """Shared bar-by-bar BE loop used by Models B and C."""
    entry       = sig["entry"]
    original_sl = sig["sl"]
    tp          = sig["tp"]
    d           = sig["direction"]
    i           = sig["bar_idx"]
    n           = len(df)
    start       = i + 1

    irl = find_irl_target(d, i, entry, tp, pivot_highs, pivot_lows)

    current_sl   = original_sl
    be_triggered = False
    outcome      = "open"
    exit_p       = entry

    for j in range(start, min(start + MAX_HOLD_BARS, n)):
        if d == "long":
            if low_arr[j] <= current_sl:
                outcome, exit_p = ("scratch", entry) if be_triggered else ("loss", original_sl)
                break
            if high_arr[j] >= tp:
                outcome, exit_p = "win", tp; break
            if not be_triggered and irl is not None and high_arr[j] >= irl:
                be_triggered = True
                current_sl   = entry
        else:
            if high_arr[j] >= current_sl:
                outcome, exit_p = ("scratch", entry) if be_triggered else ("loss", original_sl)
                break
            if low_arr[j] <= tp:
                outcome, exit_p = "win", tp; break
            if not be_triggered and irl is not None and low_arr[j] <= irl:
                be_triggered = True
                current_sl   = entry

    if outcome == "open":
        last   = min(start + MAX_HOLD_BARS - 1, n - 1)
        exit_p = float(df["Close"].iloc[last])
        if (d == "long" and exit_p > entry) or (d == "short" and exit_p < entry):
            outcome = "win"
        elif abs(exit_p - entry) < 0.01:
            outcome = "scratch"
        else:
            outcome = "loss"

    net = calc_pnl(entry, exit_p, d)
    record = {**sig,
              "outcome":      outcome,
              "exit_price":   round(exit_p, 2),
              "net_pnl":      round(net, 2),
              "be_triggered": be_triggered,
              "irl_target":   round(irl, 2) if irl else None,
              "model":        model_label}
    if extra_fields:
        record.update(extra_fields)
    return record


def simulate_breakeven(
    signals:     List[Dict],
    df:          pd.DataFrame,
    pivot_highs: np.ndarray,
    pivot_lows:  np.ndarray,
) -> List[Dict]:
    """Model B: break-even at IRL, no Goldbach filters."""
    high_arr = df["High"].values
    low_arr  = df["Low"].values
    return [
        _run_be_simulation(sig, df, high_arr, low_arr, pivot_highs, pivot_lows, "B_breakeven")
        for sig in signals
    ]


def simulate_goldbach(
    signals:     List[Dict],
    df:          pd.DataFrame,
    pivot_highs: np.ndarray,
    pivot_lows:  np.ndarray,
    dr_high_arr: np.ndarray,
    dr_low_arr:  np.ndarray,
    eq_arr:      np.ndarray,
    verbose:     bool = True,
) -> Tuple[List[Dict], Dict]:
    """
    Model C: Break-Even + Partition Filter + Liquidity Sweep Filter.

    Prints rejection labels when verbose=True.
    Returns (results, filter_stats).
    """
    high_arr = df["High"].values
    low_arr  = df["Low"].values

    filter_stats = {
        "total":              len(signals),
        "partition_rejected": 0,
        "sweep_rejected":     0,
        "both_rejected":      0,
        "passed":             0,
    }

    results          = []
    rejection_log    = []

    for sig in signals:
        i       = sig["bar_idx"]
        d       = sig["direction"]
        entry   = sig["entry"]
        fvg_mid = sig["fvg_mid"]
        eq_val  = eq_arr[i]
        drh     = dr_high_arr[i]
        drl     = dr_low_arr[i]

        # ── Filter 1: Partition ──────────────────────────────────────────────
        part_ok, part_reason = partition_filter(d, fvg_mid, eq_val)

        # ── Filter 2: Liquidity Sweep ────────────────────────────────────────
        sweep_ok, sweep_reason = liquidity_sweep_filter(
            d, i, entry, high_arr, low_arr, drl, drh
        )

        # ── Tally rejections ─────────────────────────────────────────────────
        if not part_ok and not sweep_ok:
            filter_stats["partition_rejected"] += 1
            filter_stats["sweep_rejected"]     += 1
            filter_stats["both_rejected"]      += 1
            rejection_log.append(
                f"  [BOTH]      bar={i:5d} {sig['time'].strftime('%Y-%m-%d %H:%M')} "
                f"{d:5s} entry={entry:.2f} | partition: {part_reason} | sweep: {sweep_reason}"
            )
            continue

        if not part_ok:
            filter_stats["partition_rejected"] += 1
            rejection_log.append(
                f"  [PARTITION] bar={i:5d} {sig['time'].strftime('%Y-%m-%d %H:%M')} "
                f"{d:5s} entry={entry:.2f} | {part_reason}"
            )
            continue

        if not sweep_ok:
            filter_stats["sweep_rejected"] += 1
            rejection_log.append(
                f"  [SWEEP]     bar={i:5d} {sig['time'].strftime('%Y-%m-%d %H:%M')} "
                f"{d:5s} entry={entry:.2f} | {sweep_reason}"
            )
            continue

        # ── Passed both filters → simulate with BE ───────────────────────────
        filter_stats["passed"] += 1
        extra = {
            "eq":     round(eq_val, 2) if not np.isnan(eq_val) else None,
            "dr_high": round(drh,   2) if not np.isnan(drh)    else None,
            "dr_low":  round(drl,   2) if not np.isnan(drl)    else None,
        }
        record = _run_be_simulation(
            sig, df, high_arr, low_arr, pivot_highs, pivot_lows, "C_goldbach", extra
        )
        results.append(record)

    # Print rejection log
    if verbose:
        print(f"\n  ── GOLDBACH FILTER REJECTION LOG "
              f"(first 40 of {len(rejection_log)}) ──────────────────")
        for line in rejection_log[:40]:
            print(line)
        if len(rejection_log) > 40:
            print(f"  ... {len(rejection_log) - 40} more rejections (not shown)")

    return results, filter_stats


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _pf(trades: List[Dict]) -> str:
    ws = sum(t["net_pnl"] for t in trades if t["outcome"] == "win")
    ls = sum(t["net_pnl"] for t in trades if t["outcome"] == "loss")
    return "∞" if ls == 0 else f"{abs(ws/ls):.2f}"


def print_stats(trades: List[Dict], label: str) -> None:
    if not trades:
        print(f"\n  {label}: 0 trades"); return

    wins     = [t for t in trades if t["outcome"] == "win"]
    losses   = [t for t in trades if t["outcome"] == "loss"]
    scratches= [t for t in trades if t["outcome"] == "scratch"]
    non_s    = [t for t in trades if t["outcome"] != "scratch"]

    wr      = len(wins) / len(trades) * 100
    wr_ns   = len(wins) / len(non_s)  * 100 if non_s else 0
    pnl     = sum(t["net_pnl"] for t in trades)
    aw      = np.mean([t["net_pnl"] for t in wins])   if wins   else 0
    al      = np.mean([t["net_pnl"] for t in losses]) if losses else 0
    pf      = _pf(trades)
    md, mdp = max_drawdown(trades)
    longs   = [t for t in trades if t["direction"] == "long"]
    shorts  = [t for t in trades if t["direction"] == "short"]
    lwr     = len([t for t in longs  if t["outcome"] == "win"]) / len(longs)  * 100 if longs  else 0
    swr     = len([t for t in shorts if t["outcome"] == "win"]) / len(shorts) * 100 if shorts else 0

    print(f"\n  {label}")
    print(f"  {'─' * 62}")
    print(f"  Trades:       {len(trades):>6}  │  Win Rate:      {wr:>6.1f}%")
    print(f"  W / L / S:    {len(wins)}W / {len(losses)}L / {len(scratches)}S  │  WR (excl S):  {wr_ns:>6.1f}%")
    print(f"  Net P&L:      ${pnl:>9,.2f}  │  Long WR:       {lwr:>6.1f}% ({len(longs)})")
    print(f"  Avg Win:      ${aw:>9,.2f}  │  Short WR:      {swr:>6.1f}% ({len(shorts)})")
    print(f"  Avg Loss:     ${al:>9,.2f}  │  Profit Factor: {pf:>6}")
    print(f"  Max DD:       ${md:>8,.2f} ({mdp:.1f}%)")


def print_quarterly(trades: List[Dict], label: str) -> None:
    if not trades: return
    qs: Dict[str, List[Dict]] = {}
    for t in trades:
        q = f"Q{(t['time'].month-1)//3+1} {t['time'].year}"
        qs.setdefault(q, []).append(t)

    print(f"\n  {label}")
    print(f"  {'─' * 66}")
    print(f"  {'Quarter':<10} {'#':>5} {'W':>4} {'L':>4} {'S':>4} {'WR':>7} {'P&L':>11} {'PF':>7}")
    print(f"  {'─' * 66}")
    for q in sorted(qs.keys()):
        qt  = qs[q]
        w   = len([t for t in qt if t["outcome"] == "win"])
        lo  = len([t for t in qt if t["outcome"] == "loss"])
        s   = len([t for t in qt if t["outcome"] == "scratch"])
        wr  = w / len(qt) * 100
        pnl = sum(t["net_pnl"] for t in qt)
        pf  = _pf(qt)
        m   = "✅" if pnl > 0 else "❌"
        print(f"  {q:<10} {len(qt):>5} {w:>4} {lo:>4} {s:>4} {wr:>6.1f}% ${pnl:>9,.2f} {pf:>7}  {m}")


def print_report(
    baseline:     List[Dict],
    breakeven:    List[Dict],
    goldbach:     List[Dict],
    filter_stats: Dict,
) -> None:
    """Print the full 3-model comparison report."""
    all_trades = baseline or breakeven or goldbach
    if not all_trades:
        print("  No trades to report"); return

    t0 = all_trades[0]["time"].strftime("%Y-%m-%d")
    t1 = all_trades[-1]["time"].strftime("%Y-%m-%d")

    print(f"\n  {'═' * 70}")
    print(f"  📊 GOLDBACH PARTITION BACKTEST — 3-MODEL COMPARISON")
    print(f"  {'═' * 70}")
    print(f"  Period:         {t0} → {t1}")
    print(f"  Instrument:     QQQ  (15-minute bars)")
    print(f"  SL/TP:          {ATR_SL_MULT}x ATR SL | {RR_RATIO}:1 RR")
    print(f"  BE logic:       SL → entry when IRL touched (Models B & C)")
    print(f"  Partition filter: FVG midpoint must be in correct zone (Model C)")
    print(f"  Sweep filter:     Equal H/L must be swept within 5 bars (Model C)")
    print(f"  Dealing range:    Rolling {DEALING_RANGE_LOOKBACK}-bar H/L (proxy for prior session)")

    # ── Per-model stats ──────────────────────────────────────────────────────
    print_stats(baseline,  "MODEL A — FIXED SL/TP (baseline, no filters)")
    print_stats(breakeven, "MODEL B — BREAK-EVEN AT IRL (no Goldbach filters)")
    print_stats(goldbach,  "MODEL C — GOLDBACH: BE + PARTITION + SWEEP ⭐")

    # ── Goldbach filter summary ──────────────────────────────────────────────
    tot  = filter_stats["total"]
    part = filter_stats["partition_rejected"]
    swp  = filter_stats["sweep_rejected"]
    both = filter_stats["both_rejected"]
    pas  = filter_stats["passed"]
    # Unique rejections: both counted once each in partition+sweep, so unique total =
    # total - passed.  Individual counts include "both" overlaps already.
    rejected_total = tot - pas

    print(f"\n  GOLDBACH FILTER SUMMARY")
    print(f"  {'─' * 62}")
    print(f"  Total signals examined:   {tot:>5}")
    print(f"  Passed both filters:      {pas:>5}  ({pas/tot*100:.1f}% pass rate)")
    print(f"  Rejected (total unique):  {rejected_total:>5}  ({rejected_total/tot*100:.1f}%)")
    print(f"  {'─' * 40}")
    part_only = part - both
    swp_only  = swp  - both
    print(f"  ├─ Partition ONLY:        {part_only:>5}  ({part_only/tot*100:.1f}%)")
    print(f"  ├─ Sweep ONLY:            {swp_only:>5}  ({swp_only/tot*100:.1f}%)")
    print(f"  └─ Both filters failed:   {both:>5}  ({both/tot*100:.1f}%)")
    print(f"  {'─' * 40}")
    print(f"  [Partition total rejections (incl both):  {part} | {part/tot*100:.1f}%]")
    print(f"  [Sweep total rejections    (incl both):  {swp}  | {swp/tot*100:.1f}%]")

    # ── Head-to-head ─────────────────────────────────────────────────────────
    def _row(label, trades):
        if not trades:
            return f"  {label:<35} {'N/A':>5} {'N/A':>7} {'N/A':>11} {'N/A':>7} {'N/A':>9} {'N/A':>4}"
        n    = len(trades)
        wr   = len([t for t in trades if t["outcome"] == "win"]) / n * 100
        pnl  = sum(t["net_pnl"] for t in trades)
        pf   = _pf(trades)
        md, _= max_drawdown(trades)
        s    = len([t for t in trades if t["outcome"] == "scratch"])
        return (f"  {label:<35} {n:>5} {wr:>6.1f}% ${pnl:>9,.2f} {pf:>7} ${md:>7,.2f} {s:>4}")

    print(f"\n  HEAD-TO-HEAD COMPARISON")
    print(f"  {'─' * 70}")
    print(f"  {'Model':<35} {'#':>5} {'WR':>7} {'P&L':>11} {'PF':>7} {'Max DD':>9} {'S':>4}")
    print(f"  {'─' * 70}")
    print(_row("A) Fixed SL/TP (baseline)",   baseline))
    print(_row("B) Break-Even at IRL",        breakeven))
    print(_row("C) Goldbach (BE+Part+Sweep) ⭐", goldbach))

    # Deltas vs baseline
    if baseline and goldbach:
        a_pnl = sum(t["net_pnl"] for t in baseline)
        c_pnl = sum(t["net_pnl"] for t in goldbach)
        a_md, _ = max_drawdown(baseline)
        c_md, _ = max_drawdown(goldbach)
        print(f"\n  Goldbach vs Baseline:")
        print(f"    ΔP&L:   ${c_pnl - a_pnl:+,.2f}")
        print(f"    ΔMaxDD: ${abs(c_md) - abs(a_md):+,.2f} "
              f"({'tighter' if abs(c_md) < abs(a_md) else 'wider'})")
        print(f"    ΔTrades: {len(goldbach) - len(baseline):+d} "
              f"(filtered out {len(baseline) - len(goldbach)} trades)")

    if breakeven and goldbach:
        b_pnl = sum(t["net_pnl"] for t in breakeven)
        c_pnl = sum(t["net_pnl"] for t in goldbach)
        b_md, _ = max_drawdown(breakeven)
        c_md, _ = max_drawdown(goldbach)
        print(f"\n  Goldbach vs Break-Even:")
        print(f"    ΔP&L:   ${c_pnl - b_pnl:+,.2f}")
        print(f"    ΔMaxDD: ${abs(c_md) - abs(b_md):+,.2f} "
              f"({'tighter' if abs(c_md) < abs(b_md) else 'wider'})")

    # ── Quarterly breakdowns ──────────────────────────────────────────────────
    print_quarterly(baseline,  "MODEL A — QUARTERLY WALK-FORWARD")
    print_quarterly(breakeven, "MODEL B — QUARTERLY WALK-FORWARD")
    print_quarterly(goldbach,  "MODEL C — QUARTERLY WALK-FORWARD")

    # ── Equity curves ────────────────────────────────────────────────────────
    print(f"\n  EQUITY CURVES")
    print(f"  {'─' * 62}")
    for lbl, tr in [("A) Fixed SL/TP", baseline),
                     ("B) Break-Even",  breakeven),
                     ("C) Goldbach ⭐", goldbach)]:
        if not tr: continue
        eq = [POSITION_SIZE]
        for t in tr:
            eq.append(eq[-1] + t["net_pnl"])
        print(f"  {lbl:<22}: ${eq[0]:>9,.0f} → ${eq[-1]:>9,.0f}  "
              f"(peak ${max(eq):>9,.0f} | trough ${min(eq):>9,.0f})")

    print(f"\n  {'═' * 70}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n  {'═' * 70}")
    print(f"  🔬 GOLDBACH PARTITION BACKTEST — FVG + BE + PARTITION + SWEEP")
    print(f"  {'═' * 70}")

    df = load_data()
    if len(df) < 200:
        print("  ❌ Not enough data"); return

    # Precompute shared indicators
    print(f"  🔍 Computing IRL pivots...")
    pivot_highs, pivot_lows = find_local_pivots(df["High"].values, df["Low"].values)
    n_ph = int(np.count_nonzero(~np.isnan(pivot_highs)))
    n_pl = int(np.count_nonzero(~np.isnan(pivot_lows)))
    print(f"  📊 {n_ph} pivot highs, {n_pl} pivot lows")

    print(f"  🔍 Computing dealing ranges and equilibrium...")
    dr_high_arr, dr_low_arr, eq_arr = compute_dealing_range(df)
    n_eq = int(np.count_nonzero(~np.isnan(eq_arr)))
    print(f"  📊 {n_eq} bars with valid EQ ({DEALING_RANGE_LOOKBACK}-bar rolling H/L midpoint)")

    print(f"  ⚡ Generating FVG signals...")
    signals = generate_signals(df)
    print(f"  📊 {len(signals)} raw signals")

    print(f"\n  ⚡ Running Model A (baseline: fixed SL/TP)...")
    baseline = simulate_baseline(signals, df)
    print(f"  ✅ {len(baseline)} trades")

    print(f"\n  ⚡ Running Model B (break-even at IRL)...")
    breakeven = simulate_breakeven(signals, df, pivot_highs, pivot_lows)
    print(f"  ✅ {len(breakeven)} trades")

    print(f"\n  ⚡ Running Model C (Goldbach: BE + partition + sweep)...")
    goldbach, filter_stats = simulate_goldbach(
        signals, df, pivot_highs, pivot_lows,
        dr_high_arr, dr_low_arr, eq_arr,
        verbose=True,
    )
    print(f"\n  ✅ {len(goldbach)} trades passed Goldbach filters "
          f"({filter_stats['total'] - filter_stats['passed']} rejected)")

    print_report(baseline, breakeven, goldbach, filter_stats)


if __name__ == "__main__":
    main()
