#!/usr/bin/env python3
"""
NQravi Regime-Aware PD Filter — 2-Year Stress Test
====================================================
Hypothesis: Apply NQravi's premium/discount filter ONLY in choppy regimes
(low ADX or high VIX). In strong trends, let the engine run unfiltered.

Three models compared:
  A) No PD filter (baseline)
  B) Always PD filter (naive NQravi)
  C) Regime-aware PD filter (PD on when ADX < threshold OR VIX > threshold)
"""

import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
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
AM_START, AM_END = 570, 690
PM_START, PM_END = 810, 930

# Regime thresholds
ADX_TREND_THRESHOLD = 25      # ADX >= 25 = strong trend → disable PD filter
ADX_PERIOD = 14
VIX_CHOP_THRESHOLD = 22       # VIX >= 22 = elevated vol/chop → enable PD filter


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


def fetch_vix_daily() -> pd.Series:
    """Fetch VIX daily close, shift by 1 day to avoid lookahead."""
    print(f"  📡 Fetching ^VIX daily data...")
    try:
        vix = yf.download("^VIX", period="2y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix_close = vix["Close"].shift(1)  # previous day's VIX (no lookahead)
        vix_close.index = vix_close.index.tz_localize("UTC") if vix_close.index.tz is None else vix_close.index.tz_convert("UTC")
        print(f"  ✅ VIX: {len(vix_close)} daily bars")
        return vix_close
    except Exception as e:
        print(f"  ⚠️  VIX fetch failed: {e} — using NaN (filter always off)")
        return pd.Series(dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD).mean()


def compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    """ADX indicator on 15m data, resampled to 1h for stability."""
    # Resample to 1h for less noise
    df_1h = df.resample("1h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()

    h = df_1h["High"]
    l = df_1h["Low"]
    c = df_1h["Close"]

    plus_dm = h.diff()
    minus_dm = -l.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < plus_dm)] = 0

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx_1h = dx.ewm(alpha=1/period, min_periods=period).mean()

    # Shift to avoid lookahead, then reindex to 15m
    adx_1h_shifted = adx_1h.shift(1)
    adx_15m = adx_1h_shifted.reindex(df.index, method="ffill")
    return adx_15m


def detect_fvgs(df):
    h = df["High"].values
    l = df["Low"].values
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


def compute_pd_position(df, swing_highs, swing_lows):
    rsh = swing_highs.shift(SWING_LOOKBACK + 1).ffill()
    rsl = swing_lows.shift(SWING_LOOKBACK + 1).ffill()
    rng = (rsh - rsl).replace(0, np.nan)
    return (df["Close"] - rsl) / rng


def is_in_session(idx):
    try:
        et = idx.tz_convert("US/Eastern")
    except TypeError:
        et = idx.tz_localize("UTC").tz_convert("US/Eastern")
    mins = et.hour * 60 + et.minute
    return pd.Series((mins >= AM_START) & (mins < AM_END) | (mins >= PM_START) & (mins < PM_END), index=idx)


def map_vix_to_15m(vix_daily: pd.Series, idx_15m: pd.DatetimeIndex) -> pd.Series:
    """Map daily VIX (shifted) to 15m bars via date-based ffill."""
    if vix_daily.empty:
        return pd.Series(np.nan, index=idx_15m)
    # Normalize both to date for merge
    vix_df = vix_daily.to_frame("vix")
    vix_df["date"] = vix_df.index.date

    bars_df = pd.DataFrame({"date": idx_15m.date}, index=idx_15m)
    merged = bars_df.merge(vix_df[["date", "vix"]], on="date", how="left")
    merged.index = idx_15m
    return merged["vix"].ffill()


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_signals(df: pd.DataFrame, adx: pd.Series, vix_15m: pd.Series) -> List[Dict]:
    atr = compute_atr(df)
    bull_top, bull_bot, bear_top, bear_bot = detect_fvgs(df)
    swing_highs, swing_lows = find_swings(df)
    pd_pos = compute_pd_position(df, swing_highs, swing_lows)
    sess = is_in_session(df.index)

    high = df["High"].values
    low = df["Low"].values
    signals = []
    armed = {}

    for i in range(30, len(df)):
        atr_val = atr.iloc[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue

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
                sl_p = entry - atr_arm * ATR_SL_MULT; risk = entry - sl_p
                if risk <= 0: continue
                tp = entry + risk * RR_RATIO
            else:
                sl_p = entry + atr_arm * ATR_SL_MULT; risk = sl_p - entry
                if risk <= 0: continue
                tp = entry - risk * RR_RATIO

            pd_val = pd_pos.iloc[i]
            adx_val = adx.iloc[i] if i < len(adx) else np.nan
            vix_val = vix_15m.iloc[i] if i < len(vix_15m) else np.nan

            # ── PD filter (always-on, naive NQravi) ──
            pd_pass_always = True
            if not np.isnan(pd_val):
                if direction == "long" and pd_val > 0.5: pd_pass_always = False
                if direction == "short" and pd_val < 0.5: pd_pass_always = False

            # ── Regime detection ──
            # Choppy regime = ADX < threshold OR VIX >= threshold
            adx_low = (not np.isnan(adx_val) and adx_val < ADX_TREND_THRESHOLD)
            vix_high = (not np.isnan(vix_val) and vix_val >= VIX_CHOP_THRESHOLD)
            is_choppy = adx_low or vix_high

            # ── Regime-aware PD filter ──
            # In choppy regime: apply PD filter
            # In trending regime (ADX >= 25 AND VIX < 22): no PD filter
            pd_pass_regime = True
            if is_choppy:
                pd_pass_regime = pd_pass_always  # apply PD in chop
            # In trend: pd_pass_regime stays True (no filter)

            signals.append({
                "bar_idx": i,
                "time": df.index[i],
                "direction": direction,
                "entry": round(entry, 2),
                "sl": round(sl_p, 2),
                "tp": round(tp, 2),
                "risk": round(risk, 2),
                "pd_position": round(pd_val, 4) if not np.isnan(pd_val) else None,
                "pd_zone": "premium" if (not np.isnan(pd_val) and pd_val > 0.5) else "discount",
                "pd_pass_always": pd_pass_always,
                "pd_pass_regime": pd_pass_regime,
                "adx": round(adx_val, 1) if not np.isnan(adx_val) else None,
                "vix": round(vix_val, 1) if not np.isnan(vix_val) else None,
                "regime": "choppy" if is_choppy else "trending",
            })
            to_remove.append(fvg_bar)

        for k in to_remove:
            armed.pop(k, None)

    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_trades(signals: List[Dict], df: pd.DataFrame) -> List[Dict]:
    high = df["High"].values
    low = df["Low"].values
    n = len(df)
    results = []
    for sig in signals:
        entry, sl, tp, d = sig["entry"], sig["sl"], sig["tp"], sig["direction"]
        start = sig["bar_idx"] + 1
        outcome, exit_price, exit_bar = "open", entry, None

        for j in range(start, min(start + MAX_HOLD_BARS, n)):
            if d == "long":
                if low[j] <= sl: outcome, exit_price, exit_bar = "loss", sl, j; break
                if high[j] >= tp: outcome, exit_price, exit_bar = "win", tp, j; break
            else:
                if high[j] >= sl: outcome, exit_price, exit_bar = "loss", sl, j; break
                if low[j] <= tp: outcome, exit_price, exit_bar = "win", tp, j; break

        if outcome == "open":
            last = min(start + MAX_HOLD_BARS - 1, n - 1)
            exit_price = float(df["Close"].iloc[last]); exit_bar = last
            outcome = "win" if (d == "long" and exit_price > entry) or (d == "short" and exit_price < entry) else "loss"

        shares = POSITION_SIZE / entry
        gross = (exit_price - entry) * shares if d == "long" else (entry - exit_price) * shares
        net = gross - COMMISSION_RT - POSITION_SIZE * (SLIPPAGE_PCT / 100) * 2

        results.append({**sig, "outcome": outcome, "exit_price": round(exit_price, 2),
                        "gross_pnl": round(gross, 2), "net_pnl": round(net, 2)})
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def max_drawdown(trades):
    if not trades: return 0, 0
    eq = [POSITION_SIZE]
    for t in trades: eq.append(eq[-1] + t["net_pnl"])
    eq = np.array(eq); peak = np.maximum.accumulate(eq); dd = eq - peak
    md = dd.min(); md_pct = (md / peak[np.argmin(dd)]) * 100 if peak[np.argmin(dd)] > 0 else 0
    return round(md, 2), round(md_pct, 2)


def stats(trades, label):
    if not trades:
        print(f"\n  {label}: 0 trades"); return
    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    wr = len(wins) / len(trades) * 100
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
    print(f"  Wins/Losses:  {len(wins):>3}W/{len(losses):>3}L  │  Long WR:      {lwr:>6.1f}% ({len(longs)})")
    print(f"  Net P&L:      ${pnl:>9,.2f}  │  Short WR:     {swr:>6.1f}% ({len(shorts)})")
    print(f"  Avg Win:      ${aw:>9,.2f}  │  Profit Factor: {pfs:>6}")
    print(f"  Avg Loss:     ${al:>9,.2f}  │  Max DD:       ${md:>8,.2f} ({mdp:.1f}%)")


def quarterly(trades, label):
    if not trades: return
    qs = {}
    for t in trades:
        q = f"Q{(t['time'].month-1)//3+1} {t['time'].year}"; qs.setdefault(q, []).append(t)
    print(f"\n  {label}")
    print(f"  {'─' * 58}")
    print(f"  {'Quarter':<10} {'#':>5} {'W':>4} {'WR':>7} {'P&L':>11} {'PF':>7}")
    print(f"  {'─' * 58}")
    for q in sorted(qs.keys()):
        qt = qs[q]; w = len([t for t in qt if t["outcome"] == "win"])
        wr = w/len(qt)*100; pnl = sum(t["net_pnl"] for t in qt)
        ws = sum(t["net_pnl"] for t in qt if t["outcome"] == "win")
        ls = sum(t["net_pnl"] for t in qt if t["outcome"] == "loss")
        pf = abs(ws/ls) if ls != 0 else float("inf"); pfs = f"{pf:.2f}" if pf != float("inf") else "∞"
        m = "✅" if pnl > 0 else "❌"
        print(f"  {q:<10} {len(qt):>5} {w:>4} {wr:>6.1f}% ${pnl:>9,.2f} {pfs:>7}  {m}")


def regime_breakdown(results):
    trending = [t for t in results if t["regime"] == "trending"]
    choppy = [t for t in results if t["regime"] == "choppy"]
    print(f"\n  REGIME BREAKDOWN (all trades)")
    print(f"  {'─' * 58}")
    for label, group in [("🔥 Trending (ADX≥25 & VIX<22)", trending), ("🌊 Choppy (ADX<25 or VIX≥22)", choppy)]:
        if not group: continue
        w = len([t for t in group if t["outcome"] == "win"])
        wr = w/len(group)*100; pnl = sum(t["net_pnl"] for t in group)
        print(f"  {label}: {len(group)} trades, WR {wr:.1f}%, P&L ${pnl:,.2f}")

    # Show what PD filter does in each regime
    print(f"\n  PD FILTER IMPACT BY REGIME")
    print(f"  {'─' * 58}")
    for regime_name, group in [("Trending", trending), ("Choppy", choppy)]:
        if not group: continue
        passed = [t for t in group if t["pd_pass_always"]]
        blocked = [t for t in group if not t["pd_pass_always"]]
        if passed:
            pwr = len([t for t in passed if t["outcome"] == "win"]) / len(passed) * 100
            ppnl = sum(t["net_pnl"] for t in passed)
        else:
            pwr, ppnl = 0, 0
        if blocked:
            bwr = len([t for t in blocked if t["outcome"] == "win"]) / len(blocked) * 100
            bpnl = sum(t["net_pnl"] for t in blocked)
        else:
            bwr, bpnl = 0, 0
        print(f"  {regime_name}: PD-pass {len(passed)} ({pwr:.0f}% WR, ${ppnl:,.0f}) │ "
              f"PD-blocked {len(blocked)} ({bwr:.0f}% WR, ${bpnl:,.0f})")


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(results):
    all_t = results
    always_pd = [r for r in results if r["pd_pass_always"]]
    regime_pd = [r for r in results if r["pd_pass_regime"]]

    print(f"\n  {'═' * 62}")
    print(f"  📊 NQravi REGIME-AWARE PD FILTER — 2-YEAR STRESS TEST")
    print(f"  {'═' * 62}")
    print(f"  Period:           {all_t[0]['time'].strftime('%Y-%m-%d')} → {all_t[-1]['time'].strftime('%Y-%m-%d')}")
    print(f"  ADX trend gate:   ≥{ADX_TREND_THRESHOLD} (1h ADX, .shift(1))")
    print(f"  VIX chop gate:    ≥{VIX_CHOP_THRESHOLD} (daily VIX, .shift(1))")
    print(f"  Logic:            PD filter ON when ADX<{ADX_TREND_THRESHOLD} OR VIX≥{VIX_CHOP_THRESHOLD}")
    print(f"                    PD filter OFF when ADX≥{ADX_TREND_THRESHOLD} AND VIX<{VIX_CHOP_THRESHOLD}")

    stats(all_t, "MODEL A — NO PD FILTER (baseline)")
    stats(always_pd, "MODEL B — ALWAYS PD FILTER (naive NQravi)")
    stats(regime_pd, "MODEL C — REGIME-AWARE PD FILTER ⭐")

    # Head-to-head comparison
    a_pnl = sum(t["net_pnl"] for t in all_t)
    b_pnl = sum(t["net_pnl"] for t in always_pd)
    c_pnl = sum(t["net_pnl"] for t in regime_pd)
    a_wr = len([t for t in all_t if t["outcome"] == "win"]) / len(all_t) * 100
    b_wr = len([t for t in always_pd if t["outcome"] == "win"]) / len(always_pd) * 100 if always_pd else 0
    c_wr = len([t for t in regime_pd if t["outcome"] == "win"]) / len(regime_pd) * 100 if regime_pd else 0
    a_md, _ = max_drawdown(all_t)
    b_md, _ = max_drawdown(always_pd)
    c_md, _ = max_drawdown(regime_pd)

    print(f"\n  HEAD-TO-HEAD COMPARISON")
    print(f"  {'─' * 58}")
    print(f"  {'Model':<35} {'Trades':>6} {'WR':>7} {'P&L':>11} {'MaxDD':>9}")
    print(f"  {'─' * 58}")
    print(f"  {'A) No filter':<35} {len(all_t):>6} {a_wr:>6.1f}% ${a_pnl:>9,.2f} ${a_md:>7,.2f}")
    print(f"  {'B) Always PD':<35} {len(always_pd):>6} {b_wr:>6.1f}% ${b_pnl:>9,.2f} ${b_md:>7,.2f}")
    print(f"  {'C) Regime-aware PD ⭐':<35} {len(regime_pd):>6} {c_wr:>6.1f}% ${c_pnl:>9,.2f} ${c_md:>7,.2f}")

    # Regime breakdown
    regime_breakdown(results)

    # Quarterly for regime-aware model
    quarterly(regime_pd, "MODEL C — QUARTERLY WALK-FORWARD")

    # Equity curves
    for label, trades in [("A) No filter", all_t), ("B) Always PD", always_pd), ("C) Regime-aware", regime_pd)]:
        if not trades: continue
        eq = [POSITION_SIZE]
        for t in trades: eq.append(eq[-1] + t["net_pnl"])
        print(f"\n  EQUITY — {label}: ${eq[0]:,.0f} → ${eq[-1]:,.0f} (peak ${max(eq):,.0f}, trough ${min(eq):,.0f})")

    print(f"\n  {'═' * 62}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n  {'═' * 62}")
    print(f"  🔬 NQravi Regime-Aware PD Filter — 2-Year Stress Test")
    print(f"  {'═' * 62}")

    df = load_data()
    if len(df) < 200:
        print("  ❌ Not enough data"); sys.exit(1)

    vix_daily = fetch_vix_daily()

    print(f"  🔍 Computing ADX (1h), VIX mapping, swings, FVGs...")
    adx = compute_adx(df)
    vix_15m = map_vix_to_15m(vix_daily, df.index)

    # Stats on regime classification
    adx_vals = adx.dropna()
    vix_vals = vix_15m.dropna()
    if len(adx_vals) > 0:
        pct_trending = (adx_vals >= ADX_TREND_THRESHOLD).mean() * 100
        print(f"  📈 ADX regime: {pct_trending:.0f}% trending (≥{ADX_TREND_THRESHOLD}), {100-pct_trending:.0f}% choppy")
    if len(vix_vals) > 0:
        pct_high_vix = (vix_vals >= VIX_CHOP_THRESHOLD).mean() * 100
        print(f"  🌊 VIX regime: {pct_high_vix:.0f}% elevated (≥{VIX_CHOP_THRESHOLD}), {100-pct_high_vix:.0f}% calm")

    print(f"  ⚡ Generating signals with regime tags...")
    signals = generate_signals(df, adx, vix_15m)
    print(f"  📊 {len(signals)} raw signals")

    n_trend = sum(1 for s in signals if s["regime"] == "trending")
    n_chop = sum(1 for s in signals if s["regime"] == "choppy")
    n_regime = sum(1 for s in signals if s["pd_pass_regime"])
    n_always = sum(1 for s in signals if s["pd_pass_always"])
    print(f"  🔥 {n_trend} in trending regime, 🌊 {n_chop} in choppy regime")
    print(f"  Model B (always PD): {n_always} pass | Model C (regime): {n_regime} pass")

    print(f"  ⚡ Simulating trades with friction...")
    results = simulate_trades(signals, df)

    print_report(results)


if __name__ == "__main__":
    main()
