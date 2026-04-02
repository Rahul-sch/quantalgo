#!/usr/bin/env python3
"""
FVG V2 Statistical Enhancement Backtest
========================================
Compares 3 models on top of the baseline FVG break-even continuation model:

  Model B   — Baseline (exact replica of fvg_breakeven_backtest.py Model B)
              AM 9:30-11:30, PM 1:30-3:30, ATR_SL_MULT=1.0, RR=2.0, BE at IRL

  Model V2a — + Killzone filter
              AM 9:30-11:30 (same), PM 2:00-3:45 (skip early afternoon chop)
              Blocks 11:30-14:00 lunch zone entirely

  Model V2b — + Killzone + RVOL filter
              Same as V2a plus: the FVG-creating displacement candle must have
              Volume >= 1.2x the 20-period SMA of Volume at that bar.
              Weak-volume FVGs are discarded entirely (not armed).

All 3 models simulate using break-even at IRL logic (simulate_breakeven).
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

# ── Import shared helpers from the baseline ─────────────────────────────────
_base_dir = os.path.dirname(os.path.abspath(__file__))
if _base_dir not in sys.path:
    sys.path.insert(0, _base_dir)

from fvg_breakeven_backtest import (
    load_data,
    compute_atr,
    detect_fvgs,
    find_local_pivots,
    find_irl_target,
    calc_pnl,
    simulate_breakeven,
    max_drawdown,
    # constants
    ATR_SL_MULT,
    RR_RATIO,
    POSITION_SIZE,
    RETEST_MAX_BARS,
    MAX_HOLD_BARS,
    IRL_PIVOT_BARS,
    # baseline session windows (Model B)
    AM_START, AM_END,   # 570, 690  = 9:30-11:30
    PM_START, PM_END,   # 810, 930  = 13:30-15:30
)

# ════════════════════════════════════════════════════════════════════════════
# V2 SESSION CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

# V2a / V2b killzone: AM same, PM shifted to 14:00-15:45
V2_AM_START, V2_AM_END = 570, 690    # 9:30-11:30 ET (unchanged)
V2_PM_START, V2_PM_END = 840, 945    # 14:00-15:45 ET  <- key change

# RVOL parameters
RVOL_PERIOD = 20    # SMA window for volume
RVOL_MIN    = 1.2   # minimum relative volume on FVG-creating bar


# ════════════════════════════════════════════════════════════════════════════
# SESSION FILTER HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _to_et_minutes(idx: pd.DatetimeIndex) -> np.ndarray:
    """Convert a DatetimeIndex to minutes-since-midnight in US/Eastern."""
    try:
        et = idx.tz_convert("US/Eastern")
    except TypeError:
        et = idx.tz_localize("UTC").tz_convert("US/Eastern")
    return np.array(et.hour * 60 + et.minute)


def is_in_session_baseline(idx: pd.DatetimeIndex) -> pd.Series:
    """Baseline session filter: AM 9:30-11:30 + PM 13:30-15:30 (original)."""
    mins = _to_et_minutes(idx)
    mask = (((mins >= AM_START) & (mins < AM_END)) |
            ((mins >= PM_START) & (mins < PM_END)))
    return pd.Series(mask, index=idx)


def is_in_session_v2(idx: pd.DatetimeIndex) -> pd.Series:
    """V2 killzone filter: AM 9:30-11:30 + PM 14:00-15:45."""
    mins = _to_et_minutes(idx)
    mask = (((mins >= V2_AM_START) & (mins < V2_AM_END)) |
            ((mins >= V2_PM_START) & (mins < V2_PM_END)))
    return pd.Series(mask, index=idx)


# ════════════════════════════════════════════════════════════════════════════
# RVOL COMPUTATION
# ════════════════════════════════════════════════════════════════════════════

def compute_rvol(df: pd.DataFrame, period: int = RVOL_PERIOD) -> pd.Series:
    """
    Relative Volume = volume[i] / SMA(volume, period)[i]
    Returns NaN for the first `period` bars (insufficient history).
    """
    vol_sma = df["Volume"].rolling(period).mean()
    return df["Volume"] / vol_sma


# ════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION — shared core
# ════════════════════════════════════════════════════════════════════════════

def _generate_signals_core(
    df: pd.DataFrame,
    sess: pd.Series,
    rvol: Optional[pd.Series],
    rvol_min: Optional[float],
) -> List[Dict]:
    """
    Core signal generator shared by all 3 models.

    Parameters
    ----------
    sess     : boolean Series — True when entry is allowed (session filter)
    rvol     : Series of relative volumes per bar (or None → RVOL not tracked)
    rvol_min : minimum RVOL threshold on FVG-creating bar (or None = no gate)

    Each returned signal dict includes:
      'fvg_rvol'  : RVOL of the FVG-creating bar (float or NaN)
      'rvol_pass' : always True (signals that fail RVOL gate are not emitted)
    """
    atr = compute_atr(df)
    bull_top, bull_bot, bear_top, bear_bot = detect_fvgs(df)

    rvol_arr = rvol.values if rvol is not None else None
    high = df["High"].values
    low  = df["Low"].values

    signals: List[Dict] = []
    # fvg_bar -> (direction, limit_price, other_edge, atr_val, armed_bar, fvg_rvol)
    armed: Dict[int, tuple] = {}

    for i in range(30, len(df)):
        atr_val = atr.iloc[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        # ── RVOL of this bar (the FVG-creating displacement candle) ──────
        fvg_rvol = np.nan
        if rvol_arr is not None and i < len(rvol_arr):
            rv = float(rvol_arr[i])
            if not np.isnan(rv):
                fvg_rvol = rv

        # RVOL gate: skip arming weak-volume FVGs when filter is active
        rvol_ok = True
        if rvol_min is not None:
            rvol_ok = (not np.isnan(fvg_rvol)) and (fvg_rvol >= rvol_min)

        # ── Arm new FVGs created at bar i ────────────────────────────────
        if not np.isnan(bull_top.iloc[i]) and rvol_ok:
            armed[i] = ("long",  float(bull_top.iloc[i]), float(bull_bot.iloc[i]),
                        float(atr_val), i, fvg_rvol)
        if not np.isnan(bear_top.iloc[i]) and rvol_ok:
            armed[i] = ("short", float(bear_bot.iloc[i]), float(bear_top.iloc[i]),
                        float(atr_val), i, fvg_rvol)

        # ── Scan armed FVGs for a retest entry ───────────────────────────
        to_remove = []
        for fvg_bar, (direction, limit_price, other_edge,
                      atr_arm, armed_bar, sig_rvol) in armed.items():
            bars_elapsed = i - armed_bar
            if bars_elapsed > RETEST_MAX_BARS:
                to_remove.append(fvg_bar); continue
            if bars_elapsed == 0:
                continue

            # FVG invalidated: price trades through the other edge
            if direction == "long"  and low[i]  < other_edge:
                to_remove.append(fvg_bar); continue
            if direction == "short" and high[i] > other_edge:
                to_remove.append(fvg_bar); continue

            # Retest: current bar must touch the limit price
            if not (low[i] <= limit_price <= high[i]):
                continue

            # Session gate (entry bar must be in allowed session window)
            if not sess.iloc[i]:
                continue

            # Build trade
            entry = limit_price
            if direction == "long":
                sl   = entry - atr_arm * ATR_SL_MULT
                risk = entry - sl
                if risk <= 0: continue
                tp   = entry + risk * RR_RATIO
            else:
                sl   = entry + atr_arm * ATR_SL_MULT
                risk = sl - entry
                if risk <= 0: continue
                tp   = entry - risk * RR_RATIO

            signals.append({
                "bar_idx":   i,
                "time":      df.index[i],
                "direction": direction,
                "entry":     round(entry, 2),
                "sl":        round(sl, 2),
                "tp":        round(tp, 2),
                "risk":      round(risk, 2),
                "fvg_rvol":  round(sig_rvol, 4) if not np.isnan(sig_rvol) else np.nan,
                "rvol_pass": True,
            })
            to_remove.append(fvg_bar)

        for k in to_remove:
            armed.pop(k, None)

    return signals


def generate_signals_model_b(df: pd.DataFrame) -> List[Dict]:
    """Model B — baseline session filter, no RVOL gate."""
    sess = is_in_session_baseline(df.index)
    rvol = compute_rvol(df)
    return _generate_signals_core(df, sess, rvol, rvol_min=None)


def generate_signals_v2a(df: pd.DataFrame) -> List[Dict]:
    """Model V2a — killzone session filter, no RVOL gate."""
    sess = is_in_session_v2(df.index)
    rvol = compute_rvol(df)
    return _generate_signals_core(df, sess, rvol, rvol_min=None)


def generate_signals_v2b(df: pd.DataFrame) -> List[Dict]:
    """Model V2b — killzone session filter + RVOL >= 1.2x gate."""
    sess = is_in_session_v2(df.index)
    rvol = compute_rvol(df)
    return _generate_signals_core(df, sess, rvol, rvol_min=RVOL_MIN)


# ════════════════════════════════════════════════════════════════════════════
# FILTER ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def count_filter_impact(df: pd.DataFrame) -> Dict:
    """
    Quantify how many baseline signals (Model B pool) are killed by each filter.

    We generate all signals that pass the BASELINE session filter (Model B
    pool), then tag each one with:
      - kz_blocked   : would be blocked by V2 killzone?
      - rvol_blocked : would be blocked by RVOL gate?

    Returns counts by filter category.
    """
    sess_base = is_in_session_baseline(df.index)
    sess_v2   = is_in_session_v2(df.index)
    rvol      = compute_rvol(df)

    # All signals passing baseline session (RVOL recorded but not filtered)
    raw = _generate_signals_core(df, sess_base, rvol, rvol_min=None)

    kz_only   = 0
    rvol_only = 0
    both      = 0
    pass_v2a  = 0
    pass_v2b  = 0

    for sig in raw:
        bi = sig["bar_idx"]
        in_v2   = bool(sess_v2.iloc[bi])
        rv      = sig.get("fvg_rvol", np.nan)
        rv_val  = float(rv) if not (isinstance(rv, float) and np.isnan(rv)) else 0.0
        rvol_ok = rv_val >= RVOL_MIN

        kz_blocked   = not in_v2
        rvol_blocked = not rvol_ok

        if kz_blocked and rvol_blocked:
            both += 1
        elif kz_blocked:
            kz_only += 1
        elif rvol_blocked:
            rvol_only += 1
        else:
            pass_v2b += 1

        if not kz_blocked:
            pass_v2a += 1

    return {
        "total_base": len(raw),
        "kz_only":    kz_only,
        "rvol_only":  rvol_only,
        "both":       both,
        "pass_v2a":   pass_v2a,
        "pass_v2b":   pass_v2b,
    }


# ════════════════════════════════════════════════════════════════════════════
# REPORTING HELPERS
# ════════════════════════════════════════════════════════════════════════════

def print_stats_block(trades: List[Dict], label: str):
    """Print a full stats block for one model."""
    if not trades:
        print(f"\n  {label}: 0 trades"); return

    wins      = [t for t in trades if t["outcome"] == "win"]
    losses    = [t for t in trades if t["outcome"] == "loss"]
    scratches = [t for t in trades if t["outcome"] == "scratch"]
    non_scratch = [t for t in trades if t["outcome"] != "scratch"]

    wr      = len(wins) / len(trades) * 100
    wr_excl = len(wins) / len(non_scratch) * 100 if non_scratch else 0
    pnl     = sum(t["net_pnl"] for t in trades)
    aw      = np.mean([t["net_pnl"] for t in wins])   if wins   else 0.0
    al      = np.mean([t["net_pnl"] for t in losses]) if losses else 0.0
    ws      = sum(t["net_pnl"] for t in wins)
    ls      = sum(t["net_pnl"] for t in losses)
    pf      = abs(ws / ls) if ls != 0 else float("inf")
    md, mdp = max_drawdown(trades)

    longs  = [t for t in trades if t["direction"] == "long"]
    shorts = [t for t in trades if t["direction"] == "short"]
    lwr    = len([t for t in longs  if t["outcome"] == "win"]) / len(longs)  * 100 if longs  else 0.0
    swr    = len([t for t in shorts if t["outcome"] == "win"]) / len(shorts) * 100 if shorts else 0.0
    pfs    = f"{pf:.2f}" if pf != float("inf") else "inf"

    print(f"\n  {label}")
    print(f"  {'─' * 62}")
    print(f"  Trades:        {len(trades):>6}  |  Win Rate:      {wr:>6.1f}%")
    print(f"  W / L / S:     {len(wins)}W / {len(losses)}L / {len(scratches)}S"
          f"  |  WR (excl S):  {wr_excl:>6.1f}%")
    print(f"  Net P&L:       ${pnl:>10,.2f}  |  Long WR:       {lwr:>6.1f}% ({len(longs)})")
    print(f"  Avg Win:       ${aw:>10,.2f}  |  Short WR:      {swr:>6.1f}% ({len(shorts)})")
    print(f"  Avg Loss:      ${al:>10,.2f}  |  Profit Factor: {pfs:>6}")
    print(f"  Max DD:        ${md:>9,.2f} ({mdp:.1f}%)")


def print_quarterly(trades: List[Dict], label: str):
    """Print quarterly walk-forward for one model."""
    if not trades:
        return
    qs: Dict[str, list] = {}
    for t in trades:
        q = f"Q{(t['time'].month - 1) // 3 + 1} {t['time'].year}"
        qs.setdefault(q, []).append(t)

    print(f"\n  {label} -- QUARTERLY WALK-FORWARD")
    print(f"  {'─' * 70}")
    print(f"  {'Quarter':<10} {'#':>5} {'W':>4} {'L':>4} {'S':>4} {'WR%':>7}"
          f" {'P&L':>11} {'PF':>7}")
    print(f"  {'─' * 70}")
    for q in sorted(qs.keys()):
        qt   = qs[q]
        w    = len([t for t in qt if t["outcome"] == "win"])
        lo   = len([t for t in qt if t["outcome"] == "loss"])
        s    = len([t for t in qt if t["outcome"] == "scratch"])
        wr   = w / len(qt) * 100
        pnl  = sum(t["net_pnl"] for t in qt)
        ws   = sum(t["net_pnl"] for t in qt if t["outcome"] == "win")
        ls   = sum(t["net_pnl"] for t in qt if t["outcome"] == "loss")
        pf   = abs(ws / ls) if ls != 0 else float("inf")
        pfs  = f"{pf:.2f}" if pf != float("inf") else "inf"
        mark = "OK" if pnl > 0 else "--"
        print(f"  {q:<10} {len(qt):>5} {w:>4} {lo:>4} {s:>4} {wr:>6.1f}%"
              f" ${pnl:>9,.2f} {pfs:>7}  [{mark}]")


def _model_summary(trades: List[Dict]):
    """Return (n_trades, wr, pnl, pf, max_dd) for a result set."""
    if not trades:
        return 0, 0.0, 0.0, 0.0, 0.0
    wins   = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    wr     = len(wins) / len(trades) * 100
    pnl    = sum(t["net_pnl"] for t in trades)
    ws     = sum(t["net_pnl"] for t in wins)
    ls     = sum(t["net_pnl"] for t in losses)
    pf     = abs(ws / ls) if ls != 0 else float("inf")
    md, _  = max_drawdown(trades)
    return len(trades), wr, pnl, pf, md


def print_head_to_head(b_res, v2a_res, v2b_res):
    """Head-to-head comparison table for all 3 models."""
    print(f"\n  HEAD-TO-HEAD COMPARISON")
    print(f"  {'=' * 76}")
    print(f"  {'Model':<26} {'#':>5} {'WR%':>7} {'P&L':>12} {'PF':>7} {'MaxDD':>10}"
          f" {'S':>4}")
    print(f"  {'─' * 76}")

    for label, res in [
        ("B  -- Baseline",      b_res),
        ("V2a -- + Killzone",   v2a_res),
        ("V2b -- + KZ + RVOL", v2b_res),
    ]:
        n, wr, pnl, pf, md = _model_summary(res)
        sc  = len([t for t in res if t["outcome"] == "scratch"])
        pfs = f"{pf:.2f}" if pf != float("inf") else "inf"
        print(f"  {label:<26} {n:>5} {wr:>6.1f}% ${pnl:>10,.2f} {pfs:>7}"
              f" ${md:>8,.2f} {sc:>4}")

    print(f"  {'─' * 76}")

    nb,  wr_b,  pnl_b,  pf_b,  md_b  = _model_summary(b_res)
    n2a, wr_2a, pnl_2a, pf_2a, md_2a = _model_summary(v2a_res)
    n2b, wr_2b, pnl_2b, pf_2b, md_2b = _model_summary(v2b_res)

    def _d(a, b, fmt=",.2f"):
        sign = "+" if (b - a) >= 0 else ""
        return f"{sign}{b - a:{fmt}}"

    print(f"  {'V2a vs B (delta)':<26} {_d(nb, n2a, 'd'):>5}"
          f" {_d(wr_b, wr_2a, '+.1f'):>7}% ${_d(pnl_b, pnl_2a):>10}"
          f" {'':>7} ${_d(md_b, md_2a):>8}")
    print(f"  {'V2b vs B (delta)':<26} {_d(nb, n2b, 'd'):>5}"
          f" {_d(wr_b, wr_2b, '+.1f'):>7}% ${_d(pnl_b, pnl_2b):>10}"
          f" {'':>7} ${_d(md_b, md_2b):>8}")
    print(f"  {'V2b vs V2a (delta)':<26} {_d(n2a, n2b, 'd'):>5}"
          f" {_d(wr_2a, wr_2b, '+.1f'):>7}% ${_d(pnl_2a, pnl_2b):>10}"
          f" {'':>7} ${_d(md_2a, md_2b):>8}")


def print_filter_analysis(filter_info: Dict, b_trades, v2a_trades, v2b_trades):
    """Show how many signals were killed by each filter."""
    total  = filter_info["total_base"]
    kz     = filter_info["kz_only"]
    rv     = filter_info["rvol_only"]
    both   = filter_info["both"]
    p_v2a  = filter_info["pass_v2a"]
    p_v2b  = filter_info["pass_v2b"]

    def pct(n):
        return f"{n / total * 100:.1f}%" if total else "n/a"

    print(f"\n  FILTER ANALYSIS")
    print(f"  {'=' * 64}")
    print(f"  Baseline signals (Model B pool):            {total:>6}")
    print(f"  +-- Killed by Killzone only:                {kz:>6}  ({pct(kz)})")
    print(f"  +-- Killed by RVOL only (in V2a window):   {rv:>6}  ({pct(rv)})")
    print(f"  +-- Killed by both KZ + RVOL:              {both:>6}  ({pct(both)})")
    print(f"  +-- Pass V2a (survive killzone):            {p_v2a:>6}  ({pct(p_v2a)})")
    print(f"  +-- Pass V2b (survive KZ + RVOL):          {p_v2b:>6}  ({pct(p_v2b)})")
    print()
    print(f"  Actual executed trades:")
    print(f"  +-- Model B   (baseline):                   {len(b_trades):>6}")
    print(f"  +-- Model V2a (+ killzone):                 {len(v2a_trades):>6}")
    print(f"  +-- Model V2b (+ KZ + RVOL):               {len(v2b_trades):>6}")
    print(f"  Note: executed counts differ from filter analysis because the")
    print(f"        session gate is applied to the ENTRY bar, while the RVOL")
    print(f"        gate is applied to the FVG-CREATION bar.")


def print_rvol_distribution(b_trades: List[Dict], v2b_trades: List[Dict]):
    """
    RVOL distribution analysis:
      - Avg RVOL of winning vs losing trades (validate filter direction)
      - RVOL bucket breakdown for Model B
      - V2b accepted trades summary
    """
    print(f"\n  RVOL DISTRIBUTION ANALYSIS")
    print(f"  {'=' * 64}")

    def has_rvol(t):
        rv = t.get("fvg_rvol", np.nan)
        return not (isinstance(rv, float) and np.isnan(rv))

    # ── Model B: RVOL by outcome ─────────────────────────────────────────
    b_w  = [t for t in b_trades if t["outcome"] == "win"    and has_rvol(t)]
    b_l  = [t for t in b_trades if t["outcome"] == "loss"   and has_rvol(t)]
    b_s  = [t for t in b_trades if t["outcome"] == "scratch" and has_rvol(t)]

    if b_w or b_l:
        avg_rv_win  = np.mean([t["fvg_rvol"] for t in b_w]) if b_w else float("nan")
        avg_rv_loss = np.mean([t["fvg_rvol"] for t in b_l]) if b_l else float("nan")
        avg_rv_scr  = np.mean([t["fvg_rvol"] for t in b_s]) if b_s else float("nan")

        print(f"  Model B (Baseline) -- RVOL of FVG-creating bar:")
        print(f"  +-- Winning  trades ({len(b_w):>4}): avg RVOL = {avg_rv_win:.3f}x")
        print(f"  +-- Losing   trades ({len(b_l):>4}): avg RVOL = {avg_rv_loss:.3f}x")
        print(f"  +-- Scratch  trades ({len(b_s):>4}): avg RVOL = {avg_rv_scr:.3f}x")

        delta = avg_rv_win - avg_rv_loss
        if delta > 0:
            verdict = "[+] Winners have HIGHER RVOL -- filter directionally valid"
        else:
            verdict = "[-] Winners have LOWER RVOL -- filter may not add edge"
        print(f"\n  RVOL Win-vs-Loss delta: {delta:+.3f}x  ->  {verdict}")

    # ── RVOL bucket breakdown (Model B) ─────────────────────────────────
    all_rv = [t for t in b_trades if has_rvol(t)]
    if all_rv:
        buckets = [
            (0.0, 0.8,   "<0.8x"),
            (0.8, 1.0,   "0.8-1.0x"),
            (1.0, 1.2,   "1.0-1.2x"),
            (1.2, 1.5,   "1.2-1.5x"),
            (1.5, 2.0,   "1.5-2.0x"),
            (2.0, 9999,  ">2.0x"),
        ]
        print(f"\n  RVOL Bucket Analysis  (Model B, {len(all_rv)} trades with RVOL data):")
        print(f"  {'Bucket':<12} {'#':>5} {'W':>4} {'L':>4} {'S':>4} {'WR%':>7} {'P&L':>11}")
        print(f"  {'─' * 56}")
        for lo, hi, lbl in buckets:
            bt = [t for t in all_rv if lo <= t["fvg_rvol"] < hi]
            if not bt:
                continue
            bw   = len([t for t in bt if t["outcome"] == "win"])
            bl   = len([t for t in bt if t["outcome"] == "loss"])
            bs   = len([t for t in bt if t["outcome"] == "scratch"])
            bwr  = bw / len(bt) * 100
            bpnl = sum(t["net_pnl"] for t in bt)
            print(f"  {lbl:<12} {len(bt):>5} {bw:>4} {bl:>4} {bs:>4}"
                  f" {bwr:>6.1f}% ${bpnl:>9,.2f}")

    # ── V2b accepted trades ──────────────────────────────────────────────
    if v2b_trades:
        v2b_w  = [t for t in v2b_trades if t["outcome"] == "win"    and has_rvol(t)]
        v2b_l  = [t for t in v2b_trades if t["outcome"] == "loss"   and has_rvol(t)]
        avg_w  = np.mean([t["fvg_rvol"] for t in v2b_w]) if v2b_w else float("nan")
        avg_l  = np.mean([t["fvg_rvol"] for t in v2b_l]) if v2b_l else float("nan")
        print(f"\n  Model V2b (KZ + RVOL >=1.2x) -- accepted trades RVOL:")
        print(f"  +-- Winning trades ({len(v2b_w):>4}): avg RVOL = {avg_w:.3f}x")
        print(f"  +-- Losing  trades ({len(v2b_l):>4}): avg RVOL = {avg_l:.3f}x")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    sep  = "=" * 68
    sep2 = "-" * 68

    print(f"\n  {sep}")
    print(f"  FVG V2 Statistical Enhancement Backtest")
    print(f"  3-Model Comparison: Baseline vs Killzone vs Killzone+RVOL")
    print(f"  {sep}")

    # ── Load data ────────────────────────────────────────────────────────
    df = load_data()
    if len(df) < 200:
        print("  ERROR: Not enough data"); sys.exit(1)

    # ── Precompute pivot arrays (shared across all models) ───────────────
    print(f"\n  Computing IRL pivots...")
    pivot_highs, pivot_lows = find_local_pivots(df["High"].values, df["Low"].values)
    n_ph = int(np.count_nonzero(~np.isnan(pivot_highs)))
    n_pl = int(np.count_nonzero(~np.isnan(pivot_lows)))
    print(f"  {n_ph} pivot highs, {n_pl} pivot lows")

    # ── Generate signals for each model ─────────────────────────────────
    print(f"\n  Generating signals...")
    print(f"  Model B   (baseline session filter)...")
    sigs_b   = generate_signals_model_b(df)

    print(f"  Model V2a (killzone filter)...")
    sigs_v2a = generate_signals_v2a(df)

    print(f"  Model V2b (killzone + RVOL >= {RVOL_MIN}x)...")
    sigs_v2b = generate_signals_v2b(df)

    print(f"  Signals: B={len(sigs_b)}, V2a={len(sigs_v2a)}, V2b={len(sigs_v2b)}")

    # ── Simulate all 3 models (break-even at IRL) ────────────────────────
    print(f"\n  Simulating trades (break-even at IRL for all models)...")
    res_b   = simulate_breakeven(sigs_b,   df, pivot_highs, pivot_lows)
    res_v2a = simulate_breakeven(sigs_v2a, df, pivot_highs, pivot_lows)
    res_v2b = simulate_breakeven(sigs_v2b, df, pivot_highs, pivot_lows)

    # Carry fvg_rvol from signals into results (simulate_breakeven doesn't know about it)
    # Build a lookup: bar_idx -> fvg_rvol
    def _attach_rvol(sigs, results):
        rvol_map = {s["bar_idx"]: s.get("fvg_rvol", np.nan) for s in sigs}
        for r in results:
            r["fvg_rvol"] = rvol_map.get(r["bar_idx"], np.nan)

    _attach_rvol(sigs_b,   res_b)
    _attach_rvol(sigs_v2a, res_v2a)
    _attach_rvol(sigs_v2b, res_v2b)

    # ── Filter analysis (count impact of each filter) ────────────────────
    print(f"\n  Running filter impact analysis...")
    filter_info = count_filter_impact(df)

    # ════════════════════════════════════════════════════════════════════
    # PRINT REPORT
    # ════════════════════════════════════════════════════════════════════

    period_start = df.index[0].strftime("%Y-%m-%d")
    period_end   = df.index[-1].strftime("%Y-%m-%d")

    print(f"\n\n  {sep}")
    print(f"  RESULTS: {period_start} --> {period_end}")
    print(f"  {sep}")
    print(f"  Config: ATR_SL={ATR_SL_MULT}x, RR={RR_RATIO}:1, BE at IRL pivot")
    print(f"  Baseline session:  AM {AM_START//60}:{AM_START%60:02d}-{AM_END//60}:{AM_END%60:02d}"
          f"  PM {PM_START//60}:{PM_START%60:02d}-{PM_END//60}:{PM_END%60:02d} ET")
    print(f"  V2 killzone:       AM {V2_AM_START//60}:{V2_AM_START%60:02d}-{V2_AM_END//60}:{V2_AM_END%60:02d}"
          f"  PM {V2_PM_START//60}:{V2_PM_START%60:02d}-{V2_PM_END//60}:{V2_PM_END%60:02d} ET")
    print(f"  RVOL filter:       FVG-creating bar >= {RVOL_MIN}x 20-bar vol SMA")

    # ── 1. Stats block per model ─────────────────────────────────────────
    print(f"\n\n  {sep}")
    print(f"  1. STATS PER MODEL")
    print(f"  {sep}")
    print_stats_block(res_b,   "MODEL B   -- Baseline")
    print_stats_block(res_v2a, "MODEL V2a -- + Killzone")
    print_stats_block(res_v2b, "MODEL V2b -- + Killzone + RVOL")

    # ── 2. Quarterly walk-forward per model ──────────────────────────────
    print(f"\n\n  {sep}")
    print(f"  2. QUARTERLY WALK-FORWARD")
    print(f"  {sep}")
    print_quarterly(res_b,   "MODEL B   -- Baseline")
    print_quarterly(res_v2a, "MODEL V2a -- + Killzone")
    print_quarterly(res_v2b, "MODEL V2b -- + Killzone + RVOL")

    # ── 3. Head-to-head comparison ───────────────────────────────────────
    print(f"\n\n  {sep}")
    print(f"  3. HEAD-TO-HEAD COMPARISON")
    print(f"  {sep}")
    print_head_to_head(res_b, res_v2a, res_v2b)

    # ── 4. Filter analysis ───────────────────────────────────────────────
    print(f"\n\n  {sep}")
    print(f"  4. FILTER ANALYSIS")
    print(f"  {sep}")
    print_filter_analysis(filter_info, res_b, res_v2a, res_v2b)

    # ── 5. RVOL distribution ─────────────────────────────────────────────
    print(f"\n\n  {sep}")
    print(f"  5. RVOL DISTRIBUTION")
    print(f"  {sep}")
    print_rvol_distribution(res_b, res_v2b)

    print(f"\n  {sep}")
    print(f"  Done.")
    print(f"  {sep}\n")


if __name__ == "__main__":
    main()

