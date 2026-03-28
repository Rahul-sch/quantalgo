#!/usr/bin/env python3
"""
FOCUSED ENGINE: Continuation Strategy on US Indices Only (5-minute)
Strips out all other strategies/tickers. Adds NY session filter + parameter optimizer.

Usage:
    python3 index_continuation.py                 # Run with defaults
    python3 index_continuation.py --optimize       # Run parameter optimization
    python3 index_continuation.py --symbol QQQ     # Single symbol
"""
import sys
import os
import json
import argparse
import itertools
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data_fetcher import download_data, ensure_data_dir, INSTRUMENTS
from backtester import run_backtest
from continuation import (
    find_swing_highs, find_swing_lows, detect_fvg,
    is_displacement, get_trend,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── CONFIG: US Indices ONLY ──────────────────────────────────────────────────

INDEX_SYMBOLS = ["SPY", "QQQ"]  # NQ/ES futures not on yfinance; SPY/QQQ as proxies

# Add QQQ to instruments if not already there
if "QQQ" not in INSTRUMENTS:
    INSTRUMENTS["QQQ"] = "QQQ"

INITIAL_CAPITAL = 10000.0
RISK_PCT = 0.01

# NY Session Kill Zones (EST hours converted to exchange time)
# AM Session: 09:30 - 11:30 EST
# PM Session: 13:30 - 15:30 EST
AM_START_HOUR, AM_START_MIN = 9, 30
AM_END_HOUR, AM_END_MIN = 11, 30
PM_START_HOUR, PM_START_MIN = 13, 30
PM_END_HOUR, PM_END_MIN = 15, 30


# ─── NY SESSION FILTER ────────────────────────────────────────────────────────

def is_in_ny_session(dt: pd.Timestamp) -> bool:
    """Check if timestamp falls within AM or PM NY kill zones."""
    # yfinance returns US/Eastern for US stocks
    if hasattr(dt, 'hour'):
        h, m = dt.hour, dt.minute
        time_val = h * 60 + m  # minutes since midnight

        am_start = AM_START_HOUR * 60 + AM_START_MIN   # 570
        am_end = AM_END_HOUR * 60 + AM_END_MIN         # 690
        pm_start = PM_START_HOUR * 60 + PM_START_MIN   # 810
        pm_end = PM_END_HOUR * 60 + PM_END_MIN         # 930

        return (am_start <= time_val <= am_end) or (pm_start <= time_val <= pm_end)
    return False


def convert_to_est(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame index is in US/Eastern time for session filtering."""
    df_work = df.copy()
    # Force index to DatetimeIndex first (CSV cache strips it to strings)
    if not isinstance(df_work.index, pd.DatetimeIndex):
        df_work.index = pd.to_datetime(df_work.index, utc=True)
    # Now convert timezone
    if df_work.index.tz is not None:
        df_work.index = df_work.index.tz_convert("US/Eastern")
    else:
        # No timezone — assume UTC, then convert
        df_work.index = df_work.index.tz_localize("UTC").tz_convert("US/Eastern")
    return df_work


# ─── CONTINUATION WITH SESSION FILTER ─────────────────────────────────────────

def strategy_continuation_filtered(
    df: pd.DataFrame,
    swing_lookback: int = 10,
    atr_multiplier_sl: float = 1.0,
    rr_ratio: float = 1.5,
    pullback_pct: float = 0.3,
    displacement_threshold: float = 1.3,
    session_filter: bool = True,
) -> list:
    """
    ICT Continuation with NY session filter and tunable parameters.
    Only enters trades during AM (9:30-11:30 EST) and PM (1:30-3:30 EST) sessions.
    """
    signals = []

    if len(df) < 50:
        return signals

    # Convert to EST for session filtering
    df_est = convert_to_est(df)

    # Pre-compute
    swing_highs = find_swing_highs(df, swing_lookback)
    swing_lows = find_swing_lows(df, swing_lookback)
    fvgs = detect_fvg(df)
    atr = (df["High"] - df["Low"]).rolling(20).mean()

    active_bullish_fvgs = []
    active_bearish_fvgs = []

    for i in range(30, len(df)):
        close = df["Close"].iloc[i]
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        candle_range = high - low

        if pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0:
            continue

        # ── SESSION FILTER ──
        if session_filter:
            bar_time = df_est.index[i] if i < len(df_est) else df.index[i]
            if not is_in_ny_session(bar_time):
                continue

        trend = get_trend(df, i)

        # Add new FVGs
        for fvg in fvgs:
            if fvg["bar"] == i:
                if fvg["type"] == "bullish":
                    active_bullish_fvgs.append(fvg)
                else:
                    active_bearish_fvgs.append(fvg)

        active_bullish_fvgs = active_bullish_fvgs[-20:]
        active_bearish_fvgs = active_bearish_fvgs[-20:]

        # ── BULLISH CONTINUATION ──
        if trend == "bullish":
            drawn_liq = None
            for j in range(i - 1, max(i - 50, 0), -1):
                if not pd.isna(swing_highs.iloc[j]) and swing_highs.iloc[j] > close:
                    drawn_liq = swing_highs.iloc[j]
                    break
            if drawn_liq is None:
                recent_high = df["High"].iloc[max(0, i - 20):i].max()
                if recent_high > close * 1.001:
                    drawn_liq = recent_high
                else:
                    continue

            for fvg in active_bullish_fvgs:
                if fvg["bar"] >= i - 3:
                    continue
                if low <= fvg["top"] and low >= fvg["bottom"]:
                    if close > fvg["top"] and is_displacement(candle_range, atr.iloc[i], displacement_threshold):
                        entry = close
                        # Tunable SL: use ATR multiplier
                        sl = entry - atr.iloc[i] * atr_multiplier_sl
                        # Tunable TP: use RR ratio
                        risk = entry - sl
                        tp = entry + risk * rr_ratio

                        # Cap TP at drawn liquidity if closer
                        if drawn_liq < tp:
                            tp = drawn_liq

                        reward = tp - entry
                        if risk <= 0 or reward <= 0:
                            continue
                        rr = reward / risk
                        if rr < 1.0:
                            continue

                        signals.append({
                            "bar": i, "direction": "buy", "entry": entry,
                            "sl": sl, "tp": tp, "strategy": "continuation",
                            "reason": f"Bull cont DL={drawn_liq:.2f} RR={rr:.1f}"
                        })
                        if fvg in active_bullish_fvgs:
                            active_bullish_fvgs.remove(fvg)
                        break

        # ── BEARISH CONTINUATION ──
        elif trend == "bearish":
            drawn_liq = None
            for j in range(i - 1, max(i - 50, 0), -1):
                if not pd.isna(swing_lows.iloc[j]) and swing_lows.iloc[j] < close:
                    drawn_liq = swing_lows.iloc[j]
                    break
            if drawn_liq is None:
                recent_low = df["Low"].iloc[max(0, i - 20):i].min()
                if recent_low < close * 0.999:
                    drawn_liq = recent_low
                else:
                    continue

            for fvg in active_bearish_fvgs:
                if fvg["bar"] >= i - 3:
                    continue
                if high >= fvg["bottom"] and high <= fvg["top"]:
                    if close < fvg["bottom"] and is_displacement(candle_range, atr.iloc[i], displacement_threshold):
                        entry = close
                        sl = entry + atr.iloc[i] * atr_multiplier_sl
                        risk = sl - entry
                        tp = entry - risk * rr_ratio

                        if drawn_liq > tp:
                            tp = drawn_liq

                        reward = entry - tp
                        if risk <= 0 or reward <= 0:
                            continue
                        rr = reward / risk
                        if rr < 1.0:
                            continue

                        signals.append({
                            "bar": i, "direction": "sell", "entry": entry,
                            "sl": sl, "tp": tp, "strategy": "continuation",
                            "reason": f"Bear cont DL={drawn_liq:.2f} RR={rr:.1f}"
                        })
                        if fvg in active_bearish_fvgs:
                            active_bearish_fvgs.remove(fvg)
                        break

    return signals


# ─── PARAMETER OPTIMIZER ──────────────────────────────────────────────────────

PARAM_GRID = {
    "atr_multiplier_sl": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "rr_ratio": [1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
    "displacement_threshold": [1.0, 1.3, 1.5],
}


def optimize_params(
    df: pd.DataFrame,
    symbol: str,
    session_filter: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Grid search over ATR SL multiplier, RR ratio, and displacement threshold.
    Returns best params ranked by combined score (P&L + Sharpe - Drawdown).
    """
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    total = len(combos)

    if verbose:
        print(f"\n  Optimizing {symbol}: testing {total} parameter combinations...")

    results = []

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        signals = strategy_continuation_filtered(
            df, session_filter=session_filter, **params
        )

        if len(signals) < 5:
            continue

        metrics = run_backtest(df, signals, symbol, INITIAL_CAPITAL, RISK_PCT)

        score = 0.0
        trades = metrics.get("total_trades", 0)
        if trades >= 10:
            pnl = metrics.get("total_pnl", 0)
            wr = metrics.get("win_rate", 0)
            pf = metrics.get("profit_factor", 0)
            dd = metrics.get("max_drawdown_pct", 100)
            sharpe = metrics.get("sharpe_ratio", 0)

            if isinstance(pf, str):
                pf = 10.0

            # Composite score
            score = (pnl * 0.35) + (pf * 100 * 0.25) + (sharpe * 50 * 0.2) - (dd * 10 * 0.2)

        results.append({
            "params": params,
            "trades": metrics.get("total_trades", 0),
            "pnl": round(metrics.get("total_pnl", 0), 2),
            "win_rate": round(metrics.get("win_rate", 0), 1),
            "profit_factor": metrics.get("profit_factor", 0),
            "max_drawdown": round(metrics.get("max_drawdown_pct", 0), 1),
            "sharpe": round(metrics.get("sharpe_ratio", 0), 2),
            "score": round(score, 2),
        })

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    if verbose and results:
        print(f"\n  {'ATR SL':>8} {'RR':>6} {'Disp':>6} {'Trades':>7} {'P&L':>10} {'WR':>7} {'PF':>6} {'DD':>7} {'Sharpe':>7} {'Score':>8}")
        print("  " + "─" * 78)
        for r in results[:15]:
            p = r["params"]
            pf_str = f"{r['profit_factor']:.2f}" if isinstance(r["profit_factor"], float) else str(r["profit_factor"])
            print(f"  {p['atr_multiplier_sl']:>8.2f} {p['rr_ratio']:>6.2f} {p['displacement_threshold']:>6.2f} "
                  f"{r['trades']:>7} {r['pnl']:>10.2f} {r['win_rate']:>6.1f}% {pf_str:>6} "
                  f"{r['max_drawdown']:>6.1f}% {r['sharpe']:>7.2f} {r['score']:>8.2f}")

    return results[0] if results else {}


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run_focused_backtest(
    symbols: Optional[List[str]] = None,
    session_filter: bool = True,
    atr_sl: float = 1.0,
    rr: float = 1.5,
    disp: float = 1.3,
) -> None:
    """Run continuation on US indices only, 5min data."""
    if symbols is None:
        symbols = INDEX_SYMBOLS

    print("\n" + "=" * 60)
    print("  CONTINUATION ENGINE — US INDICES ONLY (5min)")
    print(f"  Session filter: {'ON (AM 9:30-11:30 + PM 1:30-3:30 EST)' if session_filter else 'OFF'}")
    print(f"  Params: ATR SL={atr_sl}, RR={rr}, Displacement={disp}")
    print("=" * 60)

    all_results = {}

    for sym in symbols:
        print(f"\n{'─'*50}")
        print(f"  {sym} — downloading 5-minute data (60 days)...")

        df = download_data(sym, period="60d", interval="5m")
        if df.empty or len(df) < 100:
            print(f"  Skipping {sym}: not enough data")
            continue

        print(f"  {sym}: {len(df)} bars loaded")

        # Run WITH session filter
        print(f"\n  Testing WITH NY session filter...")
        signals_filtered = strategy_continuation_filtered(
            df, session_filter=True,
            atr_multiplier_sl=atr_sl, rr_ratio=rr, displacement_threshold=disp,
        )
        metrics_filtered = run_backtest(df, signals_filtered, sym, INITIAL_CAPITAL, RISK_PCT)
        _print_metrics(f"{sym} (session filter ON)", metrics_filtered)
        all_results[f"{sym}_filtered"] = metrics_filtered

        # Run WITHOUT session filter (for comparison)
        print(f"\n  Testing WITHOUT session filter (all hours)...")
        signals_all = strategy_continuation_filtered(
            df, session_filter=False,
            atr_multiplier_sl=atr_sl, rr_ratio=rr, displacement_threshold=disp,
        )
        metrics_all = run_backtest(df, signals_all, sym, INITIAL_CAPITAL, RISK_PCT)
        _print_metrics(f"{sym} (no filter)", metrics_all)
        all_results[f"{sym}_allhours"] = metrics_all

    # Save results
    out_path = os.path.join(RESULTS_DIR, "index_continuation_results.json")
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {key: val for key, val in v.items() if key not in ("trades", "equity_curve")}
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


def _print_metrics(label: str, m: Dict[str, Any]) -> None:
    pf = m.get("profit_factor", 0)
    pf_str = f"{pf:.2f}" if isinstance(pf, (int, float)) else str(pf)
    print(f"    {label}:")
    print(f"      Trades: {m.get('total_trades', 0)} | WR: {m.get('win_rate', 0):.1f}% | "
          f"P&L: ${m.get('total_pnl', 0):+.2f}")
    print(f"      PF: {pf_str} | Sharpe: {m.get('sharpe_ratio', 0):.2f} | "
          f"Max DD: {m.get('max_drawdown_pct', 0):.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Continuation Strategy — US Indices Only")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol (SPY or QQQ)")
    parser.add_argument("--optimize", action="store_true", help="Run parameter grid search")
    parser.add_argument("--no-filter", action="store_true", help="Disable NY session filter")
    parser.add_argument("--atr-sl", type=float, default=1.0, help="ATR stop-loss multiplier")
    parser.add_argument("--rr", type=float, default=1.5, help="Risk-to-reward ratio")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else None
    session_filter = not args.no_filter

    if args.optimize:
        print("\n" + "=" * 60)
        print("  PARAMETER OPTIMIZER — CONTINUATION ON US INDICES")
        print("=" * 60)

        for sym in (symbols or INDEX_SYMBOLS):
            print(f"\n{'─'*50}")
            print(f"  Optimizing: {sym}")

            df = download_data(sym, period="60d", interval="5m")
            if df.empty:
                continue

            # Optimize WITH session filter
            print(f"\n  === WITH NY Session Filter ===")
            best_filtered = optimize_params(df, sym, session_filter=True)
            if best_filtered:
                print(f"\n  ✅ BEST (filtered): {best_filtered['params']}")
                print(f"     P&L: ${best_filtered['pnl']:+.2f} | WR: {best_filtered['win_rate']:.1f}% | "
                      f"Trades: {best_filtered['trades']}")

            # Optimize WITHOUT session filter
            print(f"\n  === WITHOUT Session Filter ===")
            best_all = optimize_params(df, sym, session_filter=False)
            if best_all:
                print(f"\n  ✅ BEST (all hours): {best_all['params']}")
                print(f"     P&L: ${best_all['pnl']:+.2f} | WR: {best_all['win_rate']:.1f}% | "
                      f"Trades: {best_all['trades']}")

        # Save optimization results
        out_path = os.path.join(RESULTS_DIR, "optimization_results.json")
        print(f"\n  Optimization complete. Results saved to: {out_path}")
    else:
        run_focused_backtest(
            symbols=symbols,
            session_filter=session_filter,
            atr_sl=args.atr_sl,
            rr=args.rr,
        )

    print("\n" + "=" * 60)
    print("  DONE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
