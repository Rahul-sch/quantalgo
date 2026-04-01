#!/usr/bin/env python3
"""
SPY FVG Continuation Backtest — Using same model as QQQ
Tests the FVG continuation + break-even at IRL strategy on SPY 1h data.
"""
import sys, os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phase2_sandbox'))

from fvg_breakeven_backtest import (
    compute_atr, detect_fvgs, find_local_pivots, find_irl_target,
    calc_pnl, max_drawdown, IRL_PIVOT_BARS, ATR_SL_MULT, RR_RATIO,
    RETEST_MAX_BARS, MAX_HOLD_BARS, SWING_LOOKBACK
)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'SPY_1h_2yr.csv')

def load_data():
    df = pd.read_csv(DATA, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('US/Eastern')
    df = df[['Open','High','Low','Close','Volume']].dropna()
    print(f"  {len(df):,} bars ({df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')})")
    return df

def generate_signals(df):
    atr = compute_atr(df)
    bt, bb, st, sb = detect_fvgs(df)
    high, low = df['High'].values, df['Low'].values
    et = df.index
    mins = et.hour * 60 + et.minute
    sess = (((mins >= 570) & (mins < 690)) | ((mins >= 810) & (mins < 930)))

    signals = []
    armed = {}
    for i in range(30, len(df)):
        av = atr.iloc[i]
        if np.isnan(av) or av <= 0:
            continue
        if not np.isnan(bt.iloc[i]):
            armed[i] = ('long', float(bt.iloc[i]), float(bb.iloc[i]), float(av), i)
        if not np.isnan(st.iloc[i]):
            armed[i] = ('short', float(sb.iloc[i]), float(st.iloc[i]), float(av), i)

        rm = []
        for fb, (d, lp, oe, aa, ab) in armed.items():
            be = i - ab
            if be > RETEST_MAX_BARS:
                rm.append(fb); continue
            if be == 0:
                continue
            if d == 'long' and low[i] < oe:
                rm.append(fb); continue
            if d == 'short' and high[i] > oe:
                rm.append(fb); continue
            if not (low[i] <= lp <= high[i]):
                continue
            if not sess[i]:
                continue

            entry = lp
            if d == 'long':
                sl = entry - aa * ATR_SL_MULT
                risk = entry - sl
                if risk <= 0:
                    continue
                tp = entry + risk * RR_RATIO
            else:
                sl = entry + aa * ATR_SL_MULT
                risk = sl - entry
                if risk <= 0:
                    continue
                tp = entry - risk * RR_RATIO

            signals.append({
                'bar_idx': i, 'time': df.index[i], 'direction': d,
                'entry': round(entry, 2), 'sl': round(sl, 2),
                'tp': round(tp, 2), 'risk': round(risk, 2),
            })
            rm.append(fb)
        for k in rm:
            armed.pop(k, None)

    return signals

def sim_baseline(signals, df):
    high, low = df['High'].values, df['Low'].values
    n = len(df)
    results = []
    for sig in signals:
        e, sl, tp, d = sig['entry'], sig['sl'], sig['tp'], sig['direction']
        s = sig['bar_idx'] + 1
        out, ep = 'open', e
        for j in range(s, min(s + MAX_HOLD_BARS, n)):
            if d == 'long':
                if low[j] <= sl:
                    out, ep = 'loss', sl; break
                if high[j] >= tp:
                    out, ep = 'win', tp; break
            else:
                if high[j] >= sl:
                    out, ep = 'loss', sl; break
                if low[j] <= tp:
                    out, ep = 'win', tp; break
        if out == 'open':
            last = min(s + MAX_HOLD_BARS - 1, n - 1)
            ep = float(df['Close'].iloc[last])
            out = 'win' if (d == 'long' and ep > e) or (d == 'short' and ep < e) else 'loss'
        results.append({**sig, 'outcome': out, 'exit_price': round(ep, 2),
                        'net_pnl': round(calc_pnl(e, ep, d), 2)})
    return results

def sim_breakeven(signals, df, pivot_highs, pivot_lows):
    high, low = df['High'].values, df['Low'].values
    n = len(df)
    results = []
    for sig in signals:
        e, osl, tp, d = sig['entry'], sig['sl'], sig['tp'], sig['direction']
        s = sig['bar_idx'] + 1
        irl = find_irl_target(d, sig['bar_idx'], e, tp, pivot_highs, pivot_lows)
        csl, be_on = osl, False
        out, ep = 'open', e
        for j in range(s, min(s + MAX_HOLD_BARS, n)):
            if d == 'long':
                if low[j] <= csl:
                    out = 'scratch' if be_on else 'loss'
                    ep = e if be_on else osl; break
                if high[j] >= tp:
                    out, ep = 'win', tp; break
                if not be_on and irl is not None and high[j] >= irl:
                    be_on = True; csl = e
            else:
                if high[j] >= csl:
                    out = 'scratch' if be_on else 'loss'
                    ep = e if be_on else osl; break
                if low[j] <= tp:
                    out, ep = 'win', tp; break
                if not be_on and irl is not None and low[j] <= irl:
                    be_on = True; csl = e
        if out == 'open':
            last = min(s + MAX_HOLD_BARS - 1, n - 1)
            ep = float(df['Close'].iloc[last])
            if (d == 'long' and ep > e) or (d == 'short' and ep < e):
                out = 'win'
            elif abs(ep - e) < 0.01:
                out = 'scratch'
            else:
                out = 'loss'
        results.append({**sig, 'outcome': out, 'exit_price': round(ep, 2),
                        'net_pnl': round(calc_pnl(e, ep, d), 2), 'be_triggered': be_on})
    return results

def report(trades, label):
    if not trades:
        print(f"\n  {label}: 0 trades"); return
    w = [t for t in trades if t['outcome'] == 'win']
    l = [t for t in trades if t['outcome'] == 'loss']
    s = [t for t in trades if t['outcome'] == 'scratch']
    ns = [t for t in trades if t['outcome'] != 'scratch']
    pnl = sum(t['net_pnl'] for t in trades)
    ws = sum(t['net_pnl'] for t in w)
    ls = sum(t['net_pnl'] for t in l)
    pf = abs(ws / ls) if ls != 0 else float('inf')
    wr = len(w) / len(trades) * 100
    wrx = len(w) / len(ns) * 100 if ns else 0
    md, mdp = max_drawdown(trades)
    pfs = f"{pf:.2f}" if pf != float('inf') else "inf"

    print(f"\n  {label}")
    print(f"  {'=' * 60}")
    print(f"  Trades: {len(trades):>5} | W/L/S: {len(w)}W/{len(l)}L/{len(s)}S")
    print(f"  Win Rate: {wr:>5.1f}% | Excl Scratches: {wrx:.1f}%")
    print(f"  Net P&L: ${pnl:>10,.2f} | Profit Factor: {pfs}")
    print(f"  Max DD:  ${md:>10,.2f} ({mdp:.1f}%)")

    # Quarterly
    qs = {}
    for t in trades:
        q = f"Q{(t['time'].month-1)//3+1} {t['time'].year}"
        qs.setdefault(q, []).append(t)

    print(f"\n  Quarterly:")
    print(f"  {'Quarter':<10} {'#':>4} {'W':>3} {'L':>3} {'S':>3} {'WR':>6} {'P&L':>10} {'PF':>6}")
    print(f"  {'-' * 55}")
    for q in sorted(qs.keys()):
        qt = qs[q]
        qw = len([t for t in qt if t['outcome'] == 'win'])
        ql = len([t for t in qt if t['outcome'] == 'loss'])
        qs2 = len([t for t in qt if t['outcome'] == 'scratch'])
        qpnl = sum(t['net_pnl'] for t in qt)
        qws = sum(t['net_pnl'] for t in qt if t['outcome'] == 'win')
        qls = sum(t['net_pnl'] for t in qt if t['outcome'] == 'loss')
        qpf = abs(qws / qls) if qls != 0 else float('inf')
        qwr = qw / len(qt) * 100
        m = '✅' if qpnl > 0 else '❌'
        pfstr = f"{qpf:.2f}" if qpf != float('inf') else "inf"
        print(f"  {q:<10} {len(qt):>4} {qw:>3} {ql:>3} {qs2:>3} {qwr:>5.1f}% ${qpnl:>8,.2f} {pfstr:>6} {m}")

    eq = [10000]
    for t in trades:
        eq.append(eq[-1] + t['net_pnl'])
    print(f"\n  Equity: ${eq[0]:,.0f} -> ${eq[-1]:,.0f} (peak ${max(eq):,.0f}, trough ${min(eq):,.0f})")

def main():
    print(f"\n{'=' * 65}")
    print(f"  📊 SPY FVG CONTINUATION — 3-YEAR BACKTEST (1h bars)")
    print(f"{'=' * 65}")

    df = load_data()
    high, low = df['High'].values, df['Low'].values

    print(f"  Computing pivots...")
    pivot_highs, pivot_lows = find_local_pivots(high, low)

    print(f"  Generating signals...")
    signals = generate_signals(df)
    longs = len([s for s in signals if s['direction'] == 'long'])
    shorts = len([s for s in signals if s['direction'] == 'short'])
    print(f"  {len(signals)} signals ({longs}L / {shorts}S)")

    print(f"  Running baseline...")
    baseline = sim_baseline(signals, df)

    print(f"  Running break-even...")
    breakeven = sim_breakeven(signals, df, pivot_highs, pivot_lows)

    report(baseline, "MODEL A — BASELINE (Fixed SL/TP)")
    report(breakeven, "MODEL B — BREAK-EVEN AT IRL ⭐")

    # Head to head
    a_pnl = sum(t['net_pnl'] for t in baseline)
    b_pnl = sum(t['net_pnl'] for t in breakeven)
    a_md, _ = max_drawdown(baseline)
    b_md, _ = max_drawdown(breakeven)
    print(f"\n  HEAD-TO-HEAD")
    print(f"  {'-' * 60}")
    print(f"  Delta P&L:  ${b_pnl - a_pnl:+,.2f}")
    dd_delta = abs(b_md) - abs(a_md)
    print(f"  Delta DD:   ${dd_delta:+,.2f} ({'tighter' if dd_delta < 0 else 'wider'})")

if __name__ == '__main__':
    main()
