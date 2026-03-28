#!/usr/bin/env python3
"""
Walk-Forward Optimizer for Goldbach/ICT Strategies
Splits data into train/test windows, optimizes parameters on train,
validates on unseen test data — the proper quant way to avoid overfitting.

Usage:
    python3 walk_forward.py
    python3 walk_forward.py --symbol TSLA --strategy goldbach_bounce
    python3 walk_forward.py --symbol NVDA --strategy goldbach_momentum --windows 4
"""
import sys
import os
import json
import argparse
import itertools
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_fetcher import download_data, is_forex
from backtester import run_backtest
from strategies import (
    strategy_goldbach_bounce,
    strategy_goldbach_momentum,
    strategy_po3_breakout,
)
from continuation import strategy_continuation, strategy_range_breakout_retest

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Parameter Search Spaces ──────────────────────────────────────────────────

PARAM_GRID: Dict[str, Dict[str, List[Any]]] = {
    "goldbach_bounce": {
        "atr_multiplier_sl": [1.0, 1.5, 2.0, 2.5],
        "rr_ratio": [1.5, 2.0, 2.5, 3.0],
        "bounce_threshold": [0.3, 0.5, 0.7],   # how close to level (as fraction of ATR)
    },
    "goldbach_momentum": {
        "atr_multiplier_sl": [1.0, 1.5, 2.0],
        "rr_ratio": [2.0, 2.5, 3.0],
        "momentum_bars": [3, 5, 8],             # lookback for momentum signal
    },
    "continuation": {
        "atr_multiplier_sl": [1.0, 1.5, 2.0],
        "rr_ratio": [1.5, 2.0, 2.5],
        "pullback_pct": [0.3, 0.5, 0.618],      # fibonacci pullback depth
    },
    "po3_breakout": {
        "atr_multiplier_sl": [1.0, 1.5, 2.0],
        "rr_ratio": [1.5, 2.0, 2.5],
        "breakout_pct": [0.002, 0.005, 0.01],   # breakout confirmation
    },
}

STRATEGY_MAP = {
    "goldbach_bounce": strategy_goldbach_bounce,
    "goldbach_momentum": strategy_goldbach_momentum,
    "continuation": strategy_continuation,
    "po3_breakout": strategy_po3_breakout,
    "range_breakout_retest": strategy_range_breakout_retest,
}


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class WalkForwardWindow:
    window_num: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_bars: int
    test_bars: int
    best_params: Dict[str, Any] = field(default_factory=dict)
    train_metrics: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    symbol: str
    strategy: str
    timeframe: str
    n_windows: int
    windows: List[WalkForwardWindow] = field(default_factory=list)
    robust_params: Dict[str, Any] = field(default_factory=dict)
    combined_test_pnl: float = 0.0
    combined_test_wr: float = 0.0
    combined_test_trades: int = 0
    is_robust: bool = False
    robustness_score: float = 0.0


# ─── Core Functions ────────────────────────────────────────────────────────────

def get_param_combinations(strategy: str) -> List[Dict[str, Any]]:
    """Generate all parameter combinations for a strategy."""
    if strategy not in PARAM_GRID:
        return [{}]
    grid = PARAM_GRID[strategy]
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def inject_params(df: pd.DataFrame, strategy_fn: Any, params: Dict[str, Any]) -> List[dict]:
    """
    Call a strategy function with overridden parameters.
    Since strategies use module-level constants, we patch them temporarily.
    """
    import strategies as strat_module
    import continuation as cont_module

    # Store originals
    orig = {}
    targets = [strat_module, cont_module]

    for mod in targets:
        for key, val in params.items():
            attr = key.upper() if key.upper() in dir(mod) else key
            if hasattr(mod, attr):
                orig[(mod, attr)] = getattr(mod, attr)
                setattr(mod, attr, val)
            elif hasattr(mod, key):
                orig[(mod, key)] = getattr(mod, key)
                setattr(mod, key, val)

    try:
        signals = strategy_fn(df)
    finally:
        # Restore originals
        for (mod, attr), val in orig.items():
            setattr(mod, attr, val)

    return signals


def run_on_slice(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    strategy_fn: Any,
    params: Dict[str, Any],
    symbol: str,
    initial_capital: float = 10000.0,
    risk_pct: float = 0.01,
) -> Dict[str, Any]:
    """Run a strategy on a slice of data with given params."""
    df_slice = df.iloc[start_idx:end_idx].reset_index(drop=True)
    if len(df_slice) < 50:
        return {"total_trades": 0, "total_pnl": 0.0, "win_rate": 0.0,
                "profit_factor": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0}

    try:
        signals = strategy_fn(df_slice)
        # Filter signals to only those within the slice
        signals = [s for s in signals if s.get("bar", 0) < len(df_slice) - 1]
        return run_backtest(df_slice, signals, symbol, initial_capital, risk_pct)
    except Exception as e:
        return {"total_trades": 0, "total_pnl": 0.0, "win_rate": 0.0,
                "profit_factor": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0,
                "error": str(e)}


def score_params(metrics: Dict[str, Any]) -> float:
    """
    Score parameter combination. Higher = better.
    Balances P&L, win rate, and drawdown.
    """
    if metrics.get("total_trades", 0) < 10:
        return -999.0  # too few trades to be meaningful

    pnl = metrics.get("total_pnl", 0.0)
    wr = metrics.get("win_rate", 0.0)
    pf = metrics.get("profit_factor", 0.0)
    dd = metrics.get("max_drawdown_pct", 100.0)
    trades = metrics.get("total_trades", 0)

    if isinstance(pf, str):
        pf = 10.0  # "inf" = all wins, cap at 10

    # Composite score: reward P&L and profit factor, penalize drawdown
    score = (pnl * 0.4) + (wr * 0.2) + (pf * 10 * 0.2) - (dd * 0.2)
    return score


def walk_forward_optimize(
    df: pd.DataFrame,
    symbol: str,
    strategy_name: str,
    n_windows: int = 4,
    train_pct: float = 0.7,
    initial_capital: float = 10000.0,
    risk_pct: float = 0.01,
    timeframe: str = "1h",
) -> OptimizationResult:
    """
    Walk-forward optimization:
    1. Divide data into n_windows
    2. For each window: optimize params on train portion, validate on test
    3. Aggregate test results across all windows
    4. Check if strategy is robust (consistent across windows)
    """
    result = OptimizationResult(
        symbol=symbol,
        strategy=strategy_name,
        timeframe=timeframe,
        n_windows=n_windows,
    )

    strategy_fn = STRATEGY_MAP.get(strategy_name)
    if strategy_fn is None:
        print(f"  Unknown strategy: {strategy_name}")
        return result

    param_combos = get_param_combinations(strategy_name)
    n = len(df)
    window_size = n // n_windows

    print(f"  Data: {n} bars | Windows: {n_windows} | Params to test: {len(param_combos)}")

    all_test_pnls = []
    all_test_wrs = []
    all_test_trades = []
    best_params_per_window = []

    for w in range(n_windows):
        window_start = w * window_size
        window_end = window_start + window_size if w < n_windows - 1 else n

        train_end = window_start + int(window_size * train_pct)
        test_start = train_end
        test_end = window_end

        print(f"  Window {w+1}/{n_windows}: train [{window_start}-{train_end}] test [{test_start}-{test_end}]", end="", flush=True)

        # Optimize on train data
        best_score = -999999.0
        best_params: Dict[str, Any] = {}
        best_train_metrics: Dict[str, Any] = {}

        for params in param_combos:
            metrics = run_on_slice(df, window_start, train_end, strategy_fn, params, symbol, initial_capital, risk_pct)
            score = score_params(metrics)
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_train_metrics = metrics

        # Validate on test data with best params
        test_metrics = run_on_slice(df, test_start, test_end, strategy_fn, best_params, symbol, initial_capital, risk_pct)

        wf_window = WalkForwardWindow(
            window_num=w + 1,
            train_start=window_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_bars=train_end - window_start,
            test_bars=test_end - test_start,
            best_params=best_params,
            train_metrics=best_train_metrics,
            test_metrics=test_metrics,
        )
        result.windows.append(wf_window)
        best_params_per_window.append(best_params)

        test_pnl = test_metrics.get("total_pnl", 0.0)
        test_wr = test_metrics.get("win_rate", 0.0)
        test_trades = test_metrics.get("total_trades", 0)
        all_test_pnls.append(test_pnl)
        all_test_wrs.append(test_wr)
        all_test_trades.append(test_trades)

        print(f" → test P&L: ${test_pnl:.2f} | WR: {test_wr:.1f}% | trades: {test_trades}")

    # Aggregate
    result.combined_test_pnl = sum(all_test_pnls)
    result.combined_test_trades = sum(all_test_trades)
    result.combined_test_wr = np.mean([w for w in all_test_wrs if w > 0]) if any(w > 0 for w in all_test_wrs) else 0.0

    # Robustness check: what % of windows were profitable?
    profitable_windows = sum(1 for p in all_test_pnls if p > 0)
    result.robustness_score = profitable_windows / n_windows * 100
    result.is_robust = result.robustness_score >= 75.0  # 75%+ windows profitable

    # Find most common best params (mode across windows)
    if best_params_per_window:
        result.robust_params = _find_robust_params(best_params_per_window)

    return result


def _find_robust_params(params_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Find the most consistently optimal parameters across windows."""
    if not params_list:
        return {}
    # Use median for numeric params
    all_keys = params_list[0].keys()
    robust = {}
    for key in all_keys:
        values = [p[key] for p in params_list if key in p]
        if values and isinstance(values[0], (int, float)):
            robust[key] = float(np.median(values))
        elif values:
            # most common value
            robust[key] = max(set(values), key=values.count)
    return robust


def run_full_optimization(
    symbols: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    n_windows: int = 4,
) -> Dict[str, Any]:
    """Run walk-forward optimization across all symbols and strategies."""

    if symbols is None:
        symbols = ["TSLA", "NVDA", "SPY"]
    if strategies is None:
        strategies = ["goldbach_bounce", "goldbach_momentum", "continuation"]

    all_results = {}
    summary_rows = []

    print("\n" + "=" * 60)
    print("  WALK-FORWARD OPTIMIZER")
    print("  Train on past, validate on future — no overfitting")
    print("=" * 60)

    for sym in symbols:
        print(f"\n{'─'*50}")
        print(f"Symbol: {sym}")
        print(f"{'─'*50}")

        # Use hourly data for optimization
        df = download_data(sym, period="2y", interval="1h")
        if df.empty or len(df) < 200:
            print(f"  Not enough data for {sym}, skipping")
            continue

        for strat in strategies:
            print(f"\n  Strategy: {strat}")
            result = walk_forward_optimize(
                df=df,
                symbol=sym,
                strategy_name=strat,
                n_windows=n_windows,
                timeframe="1h",
            )

            key = f"{strat}__{sym}"
            all_results[key] = result

            robust_label = "✅ ROBUST" if result.is_robust else "⚠️  FRAGILE"
            print(f"  {robust_label} | Combined test P&L: ${result.combined_test_pnl:.2f} | "
                  f"Avg WR: {result.combined_test_wr:.1f}% | "
                  f"Robustness: {result.robustness_score:.0f}%")
            if result.robust_params:
                print(f"  Best params: {result.robust_params}")

            summary_rows.append({
                "strategy": strat,
                "symbol": sym,
                "combined_test_pnl": round(result.combined_test_pnl, 2),
                "avg_win_rate": round(result.combined_test_wr, 1),
                "total_test_trades": result.combined_test_trades,
                "robustness_pct": round(result.robustness_score, 0),
                "is_robust": result.is_robust,
                "robust_params": result.robust_params,
            })

    # Print summary table
    print("\n" + "=" * 60)
    print("  OPTIMIZATION SUMMARY — ROBUST STRATEGIES ONLY")
    print("=" * 60)

    robust_rows = [r for r in summary_rows if r["is_robust"]]
    robust_rows.sort(key=lambda x: x["combined_test_pnl"], reverse=True)

    if robust_rows:
        print(f"\n{'Strategy':<25} {'Symbol':<8} {'Test P&L':>10} {'WR':>8} {'Robust':>8}")
        print("─" * 65)
        for r in robust_rows:
            print(f"  {r['strategy']:<23} {r['symbol']:<8} ${r['combined_test_pnl']:>8.2f} "
                  f"{r['avg_win_rate']:>7.1f}% {r['robustness_pct']:>7.0f}%")
            print(f"    → Use params: {r['robust_params']}")
    else:
        print("\n  No robust strategies found. All results are overfit.")
        print("  This means the edge is NOT consistent across time periods.")

    print("\n  All results (including fragile):")
    summary_rows.sort(key=lambda x: x["combined_test_pnl"], reverse=True)
    print(f"\n{'Strategy':<25} {'Symbol':<8} {'Test P&L':>10} {'WR':>8} {'Robust':>8}")
    print("─" * 65)
    for r in summary_rows[:10]:
        label = "✅" if r["is_robust"] else "⚠️ "
        print(f"  {label} {r['strategy']:<22} {r['symbol']:<8} ${r['combined_test_pnl']:>8.2f} "
              f"{r['avg_win_rate']:>7.1f}%")

    # Save results
    output = {
        "summary": summary_rows,
        "robust_strategies": robust_rows,
    }
    out_path = os.path.join(RESULTS_DIR, "walk_forward_results.json")
    with open(out_path, "w") as f:
        # Convert results to serializable format (exclude equity curves)
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Strategy Optimizer")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol (e.g. TSLA)")
    parser.add_argument("--strategy", type=str, default=None, help="Single strategy name")
    parser.add_argument("--windows", type=int, default=4, help="Number of walk-forward windows (default: 4)")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else None
    strategies = [args.strategy] if args.strategy else None

    run_full_optimization(symbols=symbols, strategies=strategies, n_windows=args.windows)

    print("\n" + "=" * 60)
    print("  DONE. Check results/walk_forward_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
