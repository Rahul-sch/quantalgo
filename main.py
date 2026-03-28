#!/usr/bin/env python3
"""
Goldbach/ICT Trading Backtester - Main Runner
Downloads data, runs 5 strategies on all instruments, generates report.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data_fetcher import download_all, download_data, is_forex
from strategies import (
    strategy_goldbach_bounce,
    strategy_amd_cycle,
    strategy_goldbach_momentum,
    strategy_po3_breakout,
    strategy_multi_tf_goldbach,
)
from continuation import strategy_continuation, strategy_range_breakout_retest
from backtester import run_backtest
from report import save_results

# ─── Configuration ─────────────────────────────────────────────
INITIAL_CAPITAL = 10000.0
RISK_PCT = 0.01  # 1% risk per trade

FOREX_PAIRS = ["GBPJPY", "GBPUSD", "EURUSD", "EURJPY", "USDJPY"]
STOCKS = ["NVDA", "TSLA", "SPY"]
ALL_SYMBOLS = FOREX_PAIRS + STOCKS


def main():
    print("=" * 60)
    print("  GOLDBACH / ICT BACKTESTING ENGINE")
    print("  Paper Trading Only - Educational Purposes")
    print("=" * 60)
    print()

    # 1. Download data (multiple timeframes for more data)
    hourly_data = download_all(period="2y", interval="1h")

    daily_data = {}
    print("Downloading 5-year daily data...")
    for sym in ALL_SYMBOLS:
        df = download_data(sym, period="5y", interval="1d")
        if not df.empty:
            daily_data[sym] = df

    daily_10y = {}
    print("Downloading 10-year daily data (max history)...")
    for sym in ALL_SYMBOLS:
        df = download_data(sym, period="max", interval="1d")
        if not df.empty:
            daily_10y[sym] = df

    fivemin_data = {}
    print("Downloading 5-minute data (last 60 days)...")
    for sym in ALL_SYMBOLS:
        df = download_data(sym, period="60d", interval="5m")
        if not df.empty:
            fivemin_data[sym] = df
    print()

    # 2. Run strategies
    all_results = {}
    strategy_map = {
        "goldbach_bounce": strategy_goldbach_bounce,
        "amd_cycle": strategy_amd_cycle,
        "goldbach_momentum": strategy_goldbach_momentum,
        "po3_breakout": strategy_po3_breakout,
        "continuation": strategy_continuation,
        "range_breakout_retest": strategy_range_breakout_retest,
    }

    for sym, df in hourly_data.items():
        print(f"Testing {sym}...")
        for strat_name, strat_fn in strategy_map.items():
            print(f"  → {strat_name}...", end=" ")
            try:
                signals = strat_fn(df)
                result = run_backtest(df, signals, sym, INITIAL_CAPITAL, RISK_PCT)
                key = f"{strat_name}__{sym}"
                all_results[key] = result
                result["strategy"] = strat_name
                print(f"{result['total_trades']} trades, {result['win_rate']}% win rate, ${result['total_pnl']} P&L")
            except Exception as e:
                print(f"ERROR: {e}")

    # Run strategies on 5-year daily data too
    print("\nTesting on 5-YEAR daily data...")
    for sym, df in daily_data.items():
        print(f"Testing {sym} (5y daily)...")
        for strat_name, strat_fn in strategy_map.items():
            print(f"  → {strat_name}...", end=" ")
            try:
                signals = strat_fn(df)
                result = run_backtest(df, signals, sym, INITIAL_CAPITAL, RISK_PCT)
                key = f"{strat_name}_5y__{sym}"
                all_results[key] = result
                result["strategy"] = f"{strat_name} (5y)"
                print(f"{result['total_trades']} trades, {result['win_rate']}% win rate, ${result['total_pnl']} P&L")
            except Exception as e:
                print(f"ERROR: {e}")

    # Run on 5-minute data (most granular, best for continuation)
    print("\nTesting on 5-MINUTE data (60 days)...")
    for sym, df in fivemin_data.items():
        for strat_name in ["continuation", "range_breakout_retest", "goldbach_bounce"]:
            if strat_name in strategy_map:
                print(f"  → {strat_name} on {sym} (5m)...", end=" ")
                try:
                    signals = strategy_map[strat_name](df)
                    result = run_backtest(df, signals, sym, INITIAL_CAPITAL, RISK_PCT)
                    key = f"{strat_name}_5m__{sym}"
                    all_results[key] = result
                    result["strategy"] = f"{strat_name} (5m)"
                    print(f"{result['total_trades']} trades, {result['win_rate']}% win rate, ${result['total_pnl']} P&L")
                except Exception as e:
                    print(f"ERROR: {e}")

    # Multi-TF strategy (needs daily + hourly)
    print("\nTesting multi-TF strategies...")
    for sym in ALL_SYMBOLS:
        if sym in daily_data and sym in hourly_data:
            print(f"  → multi_tf_goldbach on {sym}...", end=" ")
            try:
                signals = strategy_multi_tf_goldbach(daily_data[sym], hourly_data[sym])
                result = run_backtest(daily_data[sym], signals, sym, INITIAL_CAPITAL, RISK_PCT)
                key = f"multi_tf_goldbach__{sym}"
                all_results[key] = result
                result["strategy"] = "multi_tf_goldbach"
                print(f"{result['total_trades']} trades, {result['win_rate']}% win rate, ${result['total_pnl']} P&L")
            except Exception as e:
                print(f"ERROR: {e}")

    # 3. Rank results
    ranked = sorted(
        [r for r in all_results.values() if r["total_trades"] > 0],
        key=lambda x: (x["win_rate"], x["total_return_pct"]),
        reverse=True,
    )

    # 4. Save report
    save_results(all_results, ranked)

    # 5. Print summary
    print()
    print("=" * 60)
    print("  TOP 5 STRATEGY + PAIR COMBOS")
    print("=" * 60)
    for i, r in enumerate(ranked[:5], 1):
        print(f"\n  #{i}: {r.get('strategy', '?')} on {r['symbol']}")
        print(f"      Win Rate: {r['win_rate']}% | Trades: {r['total_trades']}")
        print(f"      Total Return: {r['total_return_pct']}% | P&L: ${r['total_pnl']}")
        print(f"      Profit Factor: {r['profit_factor']} | Sharpe: {r['sharpe_ratio']}")
        print(f"      Max Drawdown: {r['max_drawdown_pct']}%")

    print()
    print("=" * 60)
    print("  DONE. Check results/ for full report.")
    print("=" * 60)


if __name__ == "__main__":
    main()
