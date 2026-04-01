#!/usr/bin/env python3
"""
FVG Sponsorship Filter — 2-Year Stress Test
=============================================
NQravi's key insight: FVGs created from higher-timeframe manipulation
("sponsored") hold better than random expansions ("unsponsored").

Sponsorship = the displacement leg that created the FVG was triggered by:
  1. A liquidity sweep (price swept a recent swing high/low before displacing)
  2. HTF level tap (price touched 1H or 4H EMA before displacing)

If neither condition is met, the FVG is "unsponsored" → block the trade.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

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

# Sponsorship parameters
SWEEP_LOOKBACK = 20          # bars before origin to find swing structure
SWEEP_MARGIN_PCT = 0.00     # price must trade AT or below swing (0% = exact sweep)
HTF_EMA_1H = 20             # 1H EMA period
HTF_EMA_4H = 20             # 4H EMA period
HTF_TOUCH_ATR_MULT = 0.10   # price within 0.1x ATR of EMA = tight "tap"
DISPLACEMENT_LOOKBACK = 5   # bars before FVG formation to check for sweep/tap origin


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

def compute_atr(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD).mean()


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
    rm = df["High"].rolling(w, center=True).max()
    rn = df["Low"].rolling(w, center=True).min()
    sh = pd.Series(np.nan, index=df.index)
    sl = pd.Series(np.nan, index=df.index)
    sh[df["High"] == rm] = df["High"][df["High"] == rm]
    sl[df["Low"] == rn] = df["Low"][df["Low"] == rn]
    return sh, sl


def compute_htf_emas(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Compute 1H and 4H EMAs, mapped back to 15m bars with .shift(1)."""
    # 1H EMA
    df_1h = df.resample("1h").agg({"Close": "last"}).dropna()
    ema_1h = df_1h["Close"].ewm(span=HTF_EMA_1H, adjust=False).mean().shift(1)
    ema_1h_15m = ema_1h.reindex(df.index, method="ffill")

    # 4H EMA
    df_4h = df.resample("4h").agg({"Close": "last"}).dropna()
    ema_4h = df_4h["Close"].ewm(span=HTF_EMA_4H, adjust=False).mean().shift(1)
    ema_4h_15m = ema_4h.reindex(df.index, method="ffill")

    return ema_1h_15m, ema_4h_15m


def is_in_session(idx):
    try:
        et = idx.tz_convert("US/Eastern")
    except TypeError:
        et = idx.tz_localize("UTC").tz_convert("US/Eastern")
    mins = et.hour * 60 + et.minute
    return pd.Series((mins >= AM_START) & (mins < AM_END) |
                     (mins >= PM_START) & (mins < PM_END), index=idx)


# ═══════════════════════════════════════════════════════════════════════════════
# SPONSORSHIP DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def check_sponsorship(
    direction: str,
    fvg_bar: int,
    df: pd.DataFrame,
    swing_highs: pd.Series,
    swing_lows: pd.Series,
    ema_1h: pd.Series,
    ema_4h: pd.Series,
    atr: pd.Series,
) -> Dict[str, bool]:
    """
    Check if the FVG at fvg_bar is "sponsored" by HTF manipulation.

    For BULLISH FVG:
      - Did price sweep below a recent swing low in the displacement origin? (bars i-DISP..i-2)
      - Did price tap the 1H or 4H EMA before displacing up?

    For BEARISH FVG:
      - Did price sweep above a recent swing high?
      - Did price tap the 1H or 4H EMA before displacing down?

    Returns dict with 'swept_liquidity', 'tapped_htf', 'sponsored'.
    """
    high = df["High"].values
    low = df["Low"].values
    atr_val = atr.iloc[fvg_bar] if fvg_bar < len(atr) else np.nan

    # Define the displacement origin window (bars before the FVG)
    origin_start = max(0, fvg_bar - DISPLACEMENT_LOOKBACK - 2)
    origin_end = fvg_bar - 1  # up to candle[i-1] (candle before the FVG confirmation bar)

    swept = False
    tapped_htf = False

    if direction == "long":
        # ── Check 1: Liquidity sweep below recent local low ──
        # Use raw recent lows (not confirmed swings) — find local lows in the
        # 10-40 bar window before the displacement. A "local low" is a bar whose
        # Low is lower than the 3 bars on each side (fast pivot, no lag).
        scan_start = max(3, origin_start - 40)
        scan_end = max(3, origin_start - 3)
        local_lows = []
        for j in range(scan_start, scan_end):
            if j < 3 or j >= len(low) - 3:
                continue
            if (low[j] <= low[j-1] and low[j] <= low[j-2] and low[j] <= low[j-3] and
                low[j] <= low[j+1] and low[j] <= low[j+2] and low[j] <= low[j+3]):
                local_lows.append((j, low[j]))

        for _, ll_price in local_lows:
            for k in range(origin_start, origin_end + 1):
                if low[k] <= ll_price:
                    swept = True; break
            if swept: break

        # ── Check 2: HTF EMA tap in origin window ──
        if not np.isnan(atr_val):
            touch_dist = atr_val * HTF_TOUCH_ATR_MULT
            for k in range(origin_start, origin_end + 1):
                ema1 = ema_1h.iloc[k] if k < len(ema_1h) else np.nan
                ema4 = ema_4h.iloc[k] if k < len(ema_4h) else np.nan
                if not np.isnan(ema1) and low[k] <= ema1 + touch_dist:
                    tapped_htf = True; break
                if not np.isnan(ema4) and low[k] <= ema4 + touch_dist:
                    tapped_htf = True; break

    elif direction == "short":
        # ── Check 1: Liquidity sweep above recent local high ──
        scan_start = max(3, origin_start - 40)
        scan_end = max(3, origin_start - 3)
        local_highs = []
        for j in range(scan_start, scan_end):
            if j < 3 or j >= len(high) - 3:
                continue
            if (high[j] >= high[j-1] and high[j] >= high[j-2] and high[j] >= high[j-3] and
                high[j] >= high[j+1] and high[j] >= high[j+2] and high[j] >= high[j+3]):
                local_highs.append((j, high[j]))

        for _, lh_price in local_highs:
            for k in range(origin_start, origin_end + 1):
                if high[k] >= lh_price:
                    swept = True; break
            if swept: break

        # ── Check 2: HTF EMA tap in origin window ──
        if not np.isnan(atr_val):
            touch_dist = atr_val * HTF_TOUCH_ATR_MULT
            for k in range(origin_start, origin_end + 1):
                ema1 = ema_1h.iloc[k] if k < len(ema_1h) else np.nan
                ema4 = ema_4h.iloc[k] if k < len(ema_4h) else np.nan
                if not np.isnan(ema1) and high[k] >= ema1 - touch_dist:
                    tapped_htf = True; break
                if not np.isnan(ema4) and high[k] >= ema4 - touch_dist:
                    tapped_htf = True; break

    sponsored = swept or tapped_htf

    return {
        "swept_liquidity": swept,
        "tapped_htf": tapped_htf,
        "sponsored": sponsored,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_signals(df, swing_highs, swing_lows, ema_1h, ema_4h) -> List[Dict]:
    atr = compute_atr(df)
    bull_top, bull_bot, bear_top, bear_bot = detect_fvgs(df)
    sess = is_in_session(df.index)
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
            sp = check_sponsorship("long", i, df, swing_highs, swing_lows, ema_1h, ema_4h, atr)
            armed[i] = ("long", float(bull_top.iloc[i]), float(bull_bot.iloc[i]), float(atr_val), i, sp)

        # ARM bearish FVG
        if not np.isnan(bear_top.iloc[i]):
            sp = check_sponsorship("short", i, df, swing_highs, swing_lows, ema_1h, ema_4h, atr)
            armed[i] = ("short", float(bear_bot.iloc[i]), float(bear_top.iloc[i]), float(atr_val), i, sp)

        # Check retests
        to_remove = []
        for fvg_bar, (direction, limit_price, other_edge, atr_arm, armed_bar, sp_info) in armed.items():
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

            signals.append({
                "bar_idx": i,
                "time": df.index[i],
                "direction": direction,
                "entry": round(entry, 2),
                "sl": round(sl_p, 2),
                "tp": round(tp, 2),
                "risk": round(risk, 2),
                "swept_liquidity": sp_info["swept_liquidity"],
                "tapped_htf": sp_info["tapped_htf"],
                "sponsored": sp_info["sponsored"],
            })
            to_remove.append(fvg_bar)

        for k in to_remove:
            armed.pop(k, None)

    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_trades(signals, df):
    high = df["High"].values
    low = df["Low"].values
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
            outcome = "win" if (d == "long" and exit_price > entry) or (d == "short" and exit_price < entry) else "loss"

        shares = POSITION_SIZE / entry
        gross = (exit_price - entry) * shares if d == "long" else (entry - exit_price) * shares
        net = gross - COMMISSION_RT - POSITION_SIZE * (SLIPPAGE_PCT / 100) * 2

        results.append({**sig, "outcome": outcome, "exit_price": round(exit_price, 2),
                        "gross_pnl": round(gross, 2), "net_pnl": round(net, 2)})
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


def print_report(results):
    all_t = results
    sponsored = [r for r in results if r["sponsored"]]
    unsponsored = [r for r in results if not r["sponsored"]]
    swept_only = [r for r in results if r["swept_liquidity"]]
    tapped_only = [r for r in results if r["tapped_htf"]]
    both = [r for r in results if r["swept_liquidity"] and r["tapped_htf"]]

    print(f"\n  {'═' * 62}")
    print(f"  📊 FVG SPONSORSHIP FILTER — 2-YEAR STRESS TEST")
    print(f"  {'═' * 62}")
    print(f"  Period:             {all_t[0]['time'].strftime('%Y-%m-%d')} → {all_t[-1]['time'].strftime('%Y-%m-%d')}")
    print(f"  Sweep lookback:     {SWEEP_LOOKBACK} bars before origin")
    print(f"  Displacement check: {DISPLACEMENT_LOOKBACK} bars before FVG")
    print(f"  HTF EMAs:           1H ({HTF_EMA_1H}p) + 4H ({HTF_EMA_4H}p)")
    print(f"  EMA touch dist:     {HTF_TOUCH_ATR_MULT}x ATR")

    # Signal classification
    n_swept = sum(1 for s in results if s["swept_liquidity"])
    n_tapped = sum(1 for s in results if s["tapped_htf"])
    n_both = sum(1 for s in results if s["swept_liquidity"] and s["tapped_htf"])
    n_sp = sum(1 for s in results if s["sponsored"])
    print(f"\n  SIGNAL CLASSIFICATION ({len(all_t)} total)")
    print(f"  {'─' * 58}")
    print(f"  Swept liquidity:    {n_swept} ({n_swept/len(all_t)*100:.0f}%)")
    print(f"  Tapped HTF EMA:     {n_tapped} ({n_tapped/len(all_t)*100:.0f}%)")
    print(f"  Both:               {n_both}")
    print(f"  Sponsored (either): {n_sp} ({n_sp/len(all_t)*100:.0f}%)")
    print(f"  Unsponsored:        {len(all_t)-n_sp} ({(len(all_t)-n_sp)/len(all_t)*100:.0f}%)")

    stats(all_t, "MODEL A — NO FILTER (baseline)")
    stats(sponsored, "MODEL B — SPONSORED ONLY ⭐")
    stats(unsponsored, "UNSPONSORED (blocked trades)")

    # Sponsorship type breakdown
    print(f"\n  SPONSORSHIP TYPE BREAKDOWN")
    print(f"  {'─' * 58}")
    for label, group in [("Swept liquidity only", [r for r in results if r["swept_liquidity"] and not r["tapped_htf"]]),
                         ("HTF EMA tap only", [r for r in results if r["tapped_htf"] and not r["swept_liquidity"]]),
                         ("Both (sweep + tap)", both)]:
        if not group: continue
        gw = len([t for t in group if t["outcome"] == "win"])
        gwr = gw/len(group)*100; gpnl = sum(t["net_pnl"] for t in group)
        print(f"  {label:<25} {len(group):>4} trades, WR {gwr:.1f}%, P&L ${gpnl:,.2f}")

    # Head-to-head
    a_pnl = sum(t["net_pnl"] for t in all_t)
    b_pnl = sum(t["net_pnl"] for t in sponsored)
    u_pnl = sum(t["net_pnl"] for t in unsponsored)
    a_wr = len([t for t in all_t if t["outcome"] == "win"]) / len(all_t) * 100
    b_wr = len([t for t in sponsored if t["outcome"] == "win"]) / len(sponsored) * 100 if sponsored else 0
    u_wr = len([t for t in unsponsored if t["outcome"] == "win"]) / len(unsponsored) * 100 if unsponsored else 0
    a_md, _ = max_drawdown(all_t)
    b_md, _ = max_drawdown(sponsored)

    print(f"\n  HEAD-TO-HEAD")
    print(f"  {'─' * 58}")
    print(f"  {'Model':<30} {'#':>5} {'WR':>7} {'P&L':>11} {'MaxDD':>9}")
    print(f"  {'─' * 58}")
    print(f"  {'A) No filter':<30} {len(all_t):>5} {a_wr:>6.1f}% ${a_pnl:>9,.2f} ${a_md:>7,.2f}")
    print(f"  {'B) Sponsored only ⭐':<30} {len(sponsored):>5} {b_wr:>6.1f}% ${b_pnl:>9,.2f} ${b_md:>7,.2f}")
    print(f"  {'Unsponsored (blocked)':<30} {len(unsponsored):>5} {u_wr:>6.1f}% ${u_pnl:>9,.2f}")

    # Quarterly
    quarterly(sponsored, "SPONSORED — QUARTERLY WALK-FORWARD")

    # Equity
    for lbl, tr in [("A) No filter", all_t), ("B) Sponsored", sponsored)]:
        if not tr: continue
        eq = [POSITION_SIZE]
        for t in tr: eq.append(eq[-1] + t["net_pnl"])
        print(f"\n  EQUITY — {lbl}: ${eq[0]:,.0f} → ${eq[-1]:,.0f} (peak ${max(eq):,.0f}, trough ${min(eq):,.0f})")

    print(f"\n  {'═' * 62}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n  {'═' * 62}")
    print(f"  🔬 FVG Sponsorship Filter — 2-Year Stress Test")
    print(f"  {'═' * 62}")

    df = load_data()
    if len(df) < 200:
        print("  ❌ Not enough data"); sys.exit(1)

    print(f"  🔍 Computing indicators...")
    swing_highs, swing_lows = find_swings(df)
    ema_1h, ema_4h = compute_htf_emas(df)
    n_sh = swing_highs.notna().sum()
    n_sl = swing_lows.notna().sum()
    print(f"  📊 Swings: {n_sh} highs, {n_sl} lows")
    print(f"  📊 1H EMA range: {ema_1h.dropna().iloc[0]:.2f} → {ema_1h.dropna().iloc[-1]:.2f}")
    print(f"  📊 4H EMA range: {ema_4h.dropna().iloc[0]:.2f} → {ema_4h.dropna().iloc[-1]:.2f}")

    print(f"  ⚡ Generating signals with sponsorship check...")
    signals = generate_signals(df, swing_highs, swing_lows, ema_1h, ema_4h)
    n_sp = sum(1 for s in signals if s["sponsored"])
    print(f"  📊 {len(signals)} signals ({n_sp} sponsored, {len(signals)-n_sp} unsponsored)")

    print(f"  ⚡ Simulating trades...")
    results = simulate_trades(signals, df)

    print_report(results)


if __name__ == "__main__":
    main()
