#!/usr/bin/env python3
"""
SIGNAL QUALITY ANALYSIS
Compares mathematical conditions of winning vs losing trades across all periods.
Answers: What was different about Jan-Mar 2026 vs 2024/2025?

Usage:
    python3 signal_analysis.py                    # Full analysis
    python3 signal_analysis.py --period jan2026   # Single period
    python3 signal_analysis.py --no-plots         # Skip chart generation
"""
import sys
import os
import json
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from quant_engine import (
    Config, generate_signals, execute_backtest,
    compute_indicators, detect_fvgs_vectorized, find_swings_vectorized,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEEP_CACHE = os.path.join(DATA_DIR, "QQQ_15m_2yr.csv")
EST = ZoneInfo("US/Eastern")

# Base config — no regime filters to capture all signals for analysis
ANALYSIS_CONFIG = Config(
    symbols=["QQQ"],
    atr_multiplier_sl=0.5,
    rr_ratio=2.5,
    displacement_threshold=1.0,
    session_filter=True,
    use_htf_filter=True,
    use_adx_filter=True,
    adx_threshold=18.0,
    use_rvol_filter=True,
    rvol_multiplier=1.2,
    use_vix_filter=False,      # OFF — capture everything for analysis
    use_weekly_trend_filter=False,
    etf_slippage_pct=0.00005,
    commission_round_trip=2.40,
    initial_capital=10000.0,
    risk_pct=0.01,
)

# Define analysis periods
PERIODS = {
    "jan2026":  ("2026-01-01", "2026-01-31", "Jan 2026 (WIN)"),
    "feb2026":  ("2026-02-01", "2026-02-28", "Feb 2026 (WIN)"),
    "mar2026":  ("2026-03-01", "2026-03-27", "Mar 2026 (MIXED)"),
    "q1_2024":  ("2024-04-05", "2024-06-30", "Q1 2024 (LOSE)"),
    "q2_2024":  ("2024-07-01", "2024-09-30", "Q2 2024 (LOSE)"),
    "q3_2024":  ("2024-10-01", "2024-12-31", "Q3 2024 (LOSE)"),
    "q1_2025":  ("2025-01-01", "2025-03-31", "Q1 2025 (LOSE)"),
    "q2_2025":  ("2025-04-01", "2025-06-30", "Q2 2025 (LOSE)"),
    "q3_2025":  ("2025-07-01", "2025-09-30", "Q3 2025 (LOSE)"),
    "q4_2025":  ("2025-10-01", "2025-12-31", "Q4 2025 (MIXED)"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    if not os.path.exists(DEEP_CACHE):
        print(f"❌ Cache not found: {DEEP_CACHE}")
        print("   Run: python3 deep_backtest.py --source massive")
        sys.exit(1)

    df = pd.read_csv(DEEP_CACHE, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        try:
            df.index = df.index.tz_localize("US/Eastern")
        except Exception:
            df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")
    return df


def slice_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_dt = pd.Timestamp(start, tz="US/Eastern")
    end_dt = pd.Timestamp(end, tz="US/Eastern") + pd.Timedelta(days=1)
    return df[(df.index >= start_dt) & (df.index < end_dt)].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE-LEVEL FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_trade_features(
    df: pd.DataFrame, signals: List[Dict], cfg: Config
) -> pd.DataFrame:
    """
    For each signal, extract the full mathematical context at entry:
    ATR, ADX, RVOL, EMA direction, VIX (if available), price momentum,
    intraday timing, distance to swing high/low, FVG age, etc.
    """
    if not signals:
        return pd.DataFrame()

    ind = compute_indicators(df, cfg)
    bull_fvg_top, bull_fvg_bot, bear_fvg_top, bear_fvg_bot = detect_fvgs_vectorized(df)
    swing_highs, swing_lows = find_swings_vectorized(df, cfg.swing_lookback)

    # Download VIX for the period
    try:
        import yfinance as yf
        start = df.index[0].date()
        end = (df.index[-1] + pd.Timedelta(days=2)).date()
        vix_df = yf.download("^VIX", start=str(start), end=str(end),
                             interval="1d", progress=False, auto_adjust=True)
        vix_close = vix_df["Close"].squeeze()
        if isinstance(vix_close, pd.DataFrame):
            vix_close = vix_close.iloc[:, 0]
        vix_daily = pd.Series(
            vix_close.values,
            index=pd.to_datetime(vix_close.index).tz_localize("US/Eastern")
        )
        vix_15m = vix_daily.reindex(df.index, method="ffill").ffill()
    except Exception:
        vix_15m = pd.Series(np.nan, index=df.index)

    records = []
    for sig in signals:
        i = sig["bar"]
        if i >= len(df) or i < 10:
            continue

        entry = sig["entry"]
        sl = sig["sl"]
        tp = sig["tp"]
        direction = sig["direction"]

        # ── Simulate outcome ──
        outcome_type = "timeout"
        exit_price = df["Close"].iloc[min(i + 50, len(df) - 1)]
        for j in range(i + 1, min(i + 51, len(df))):
            h = df["High"].iloc[j]
            l = df["Low"].iloc[j]
            if direction == "buy":
                if l <= sl:
                    outcome_type = "sl_hit"
                    exit_price = sl
                    break
                if h >= tp:
                    outcome_type = "tp_hit"
                    exit_price = tp
                    break
            else:
                if h >= sl:
                    outcome_type = "sl_hit"
                    exit_price = sl
                    break
                if l <= tp:
                    outcome_type = "tp_hit"
                    exit_price = tp
                    break

        # ── P&L ──
        sl_dist = abs(entry - sl)
        risk = 10000 * 0.01  # $100 risk per trade
        size = risk / sl_dist if sl_dist > 0 else 0
        if direction == "buy":
            gross_pnl = (exit_price - entry) * size
        else:
            gross_pnl = (entry - exit_price) * size
        net_pnl = gross_pnl - 2.40  # commission only for simplicity
        win = net_pnl > 0

        # ── Entry context ──
        atr = float(ind["atr"].iloc[i]) if not np.isnan(ind["atr"].iloc[i]) else 0
        adx = float(ind["adx"].iloc[i]) if not np.isnan(ind["adx"].iloc[i]) else 0
        rvol = float(ind["rvol"].iloc[i]) if not np.isnan(ind["rvol"].iloc[i]) else 0
        htf = int(ind["htf_signal"].iloc[i])
        price = float(df["Close"].iloc[i])
        vix = float(vix_15m.iloc[i]) if not np.isnan(vix_15m.iloc[i]) else 0

        # ATR as % of price (normalized volatility)
        atr_pct = atr / price * 100 if price > 0 else 0

        # Price momentum: 5-bar and 20-bar
        mom_5 = (price - float(df["Close"].iloc[max(0, i-5)])) / float(df["Close"].iloc[max(0, i-5)]) * 100
        mom_20 = (price - float(df["Close"].iloc[max(0, i-20)])) / float(df["Close"].iloc[max(0, i-20)]) * 100

        # Intraday timing (minutes from open)
        bar_time = df.index[i]
        minutes_from_open = bar_time.hour * 60 + bar_time.minute - 570  # 9:30 = 0
        session = "AM" if 0 <= minutes_from_open <= 120 else "PM"

        # Distance from price to nearest swing (liquidity proximity)
        sh_dist = np.nan
        for j in range(i - 1, max(i - 50, 0), -1):
            sh = swing_highs.iloc[j]
            if not np.isnan(sh) and sh > price:
                sh_dist = (sh - price) / atr if atr > 0 else 0
                break

        sl_dist_atr = (price - swing_lows.iloc[max(0, i-50):i].dropna().min()) / atr if atr > 0 else 0

        # Candle body ratio (displacement quality)
        candle_range = float(df["High"].iloc[i] - df["Low"].iloc[i])
        body = abs(float(df["Close"].iloc[i]) - float(df["Open"].iloc[i]))
        body_ratio = body / candle_range if candle_range > 0 else 0

        # Volume spike magnitude
        vol_ma = float(df["Volume"].iloc[max(0, i-10):i].mean())
        vol_spike = float(df["Volume"].iloc[i]) / vol_ma if vol_ma > 0 else 1

        # RR ratio actually achieved
        actual_rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0

        records.append({
            "bar": i,
            "timestamp": str(df.index[i]),
            "direction": direction,
            # Outcome
            "win": win,
            "net_pnl": round(net_pnl, 2),
            "outcome": outcome_type,
            # Core filters
            "atr": round(atr, 4),
            "atr_pct": round(atr_pct, 4),
            "adx": round(adx, 1),
            "rvol": round(rvol, 2),
            "htf_signal": htf,
            "vix": round(vix, 1),
            # Price context
            "mom_5bar_pct": round(mom_5, 4),
            "mom_20bar_pct": round(mom_20, 4),
            # Trade geometry
            "actual_rr": round(actual_rr, 2),
            "candle_body_ratio": round(body_ratio, 3),
            "vol_spike": round(vol_spike, 2),
            # Timing
            "session": session,
            "minutes_from_open": minutes_from_open,
            # Liquidity
            "dist_to_swing_high_atr": round(sh_dist, 2) if not np.isnan(sh_dist) else 0,
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARATIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compare_features(
    win_df: pd.DataFrame, lose_df: pd.DataFrame
) -> pd.DataFrame:
    """Compare mean feature values between winning and losing trade sets."""
    numeric_cols = [
        "atr_pct", "adx", "rvol", "vix",
        "mom_5bar_pct", "mom_20bar_pct",
        "actual_rr", "candle_body_ratio", "vol_spike",
        "minutes_from_open", "dist_to_swing_high_atr",
    ]

    rows = []
    for col in numeric_cols:
        if col not in win_df.columns or col not in lose_df.columns:
            continue
        w_mean = win_df[col].mean()
        l_mean = lose_df[col].mean()
        w_median = win_df[col].median()
        l_median = lose_df[col].median()
        diff_pct = (w_mean - l_mean) / abs(l_mean) * 100 if l_mean != 0 else 0
        rows.append({
            "feature": col,
            "win_mean": round(w_mean, 3),
            "lose_mean": round(l_mean, 3),
            "win_median": round(w_median, 3),
            "lose_median": round(l_median, 3),
            "diff_pct": round(diff_pct, 1),
            "edge_direction": "WIN higher" if w_mean > l_mean else "LOSE higher",
        })

    return pd.DataFrame(rows).sort_values("diff_pct", key=abs, ascending=False)


def analyze_win_conditions(df_features: pd.DataFrame) -> Dict[str, Any]:
    """Find the optimal thresholds that separate wins from losses."""
    if df_features.empty or "win" not in df_features.columns:
        return {}

    wins = df_features[df_features["win"] == True]
    losses = df_features[df_features["win"] == False]
    win_rate = len(wins) / len(df_features) * 100 if len(df_features) > 0 else 0

    conditions = {}

    # VIX threshold analysis
    for vix_thresh in [20, 22, 25, 28, 30]:
        filtered = df_features[df_features["vix"] < vix_thresh]
        if len(filtered) >= 3:
            wr = filtered["win"].mean() * 100
            conditions[f"vix<{vix_thresh}"] = {
                "win_rate": round(wr, 1),
                "n_trades": len(filtered),
                "net_pnl": round(filtered["net_pnl"].sum(), 2),
            }

    # ADX threshold analysis
    for adx_thresh in [18, 22, 25, 30]:
        filtered = df_features[df_features["adx"] >= adx_thresh]
        if len(filtered) >= 3:
            wr = filtered["win"].mean() * 100
            conditions[f"adx>={adx_thresh}"] = {
                "win_rate": round(wr, 1),
                "n_trades": len(filtered),
                "net_pnl": round(filtered["net_pnl"].sum(), 2),
            }

    # RVOL threshold
    for rvol_thresh in [1.2, 1.5, 2.0, 2.5]:
        filtered = df_features[df_features["rvol"] >= rvol_thresh]
        if len(filtered) >= 3:
            wr = filtered["win"].mean() * 100
            conditions[f"rvol>={rvol_thresh}"] = {
                "win_rate": round(wr, 1),
                "n_trades": len(filtered),
                "net_pnl": round(filtered["net_pnl"].sum(), 2),
            }

    # Momentum alignment
    for mom_thresh in [0.0, 0.1, 0.2]:
        # Bullish trades with positive momentum, bearish with negative
        aligned = df_features[
            ((df_features["direction"] == "buy") & (df_features["mom_20bar_pct"] > mom_thresh)) |
            ((df_features["direction"] == "sell") & (df_features["mom_20bar_pct"] < -mom_thresh))
        ]
        if len(aligned) >= 3:
            wr = aligned["win"].mean() * 100
            conditions[f"momentum_aligned>{mom_thresh}%"] = {
                "win_rate": round(wr, 1),
                "n_trades": len(aligned),
                "net_pnl": round(aligned["net_pnl"].sum(), 2),
            }

    # Session analysis
    for session in ["AM", "PM"]:
        filtered = df_features[df_features["session"] == session]
        if len(filtered) >= 3:
            wr = filtered["win"].mean() * 100
            conditions[f"session={session}"] = {
                "win_rate": round(wr, 1),
                "n_trades": len(filtered),
                "net_pnl": round(filtered["net_pnl"].sum(), 2),
            }

    # Combined: low VIX + high ADX + momentum aligned
    combined = df_features[
        (df_features["vix"] < 22) &
        (df_features["adx"] >= 22) &
        (df_features["rvol"] >= 1.5) &
        (
            ((df_features["direction"] == "buy") & (df_features["mom_20bar_pct"] > 0)) |
            ((df_features["direction"] == "sell") & (df_features["mom_20bar_pct"] < 0))
        )
    ]
    if len(combined) >= 2:
        wr = combined["win"].mean() * 100
        conditions["COMBINED: vix<22 + adx>=22 + rvol>=1.5 + momentum"] = {
            "win_rate": round(wr, 1),
            "n_trades": len(combined),
            "net_pnl": round(combined["net_pnl"].sum(), 2),
        }

    return {
        "overall_win_rate": round(win_rate, 1),
        "total_trades": len(df_features),
        "conditions": conditions,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(df_full: pd.DataFrame, periods_to_run: List[str] = None) -> None:
    all_features = []
    period_summaries = []

    run_periods = periods_to_run or list(PERIODS.keys())

    print("\n" + "═" * 75)
    print("  SIGNAL QUALITY ANALYSIS — Per-Period Breakdown")
    print("═" * 75)

    for key in run_periods:
        if key not in PERIODS:
            continue
        start, end, label = PERIODS[key]

        df_period = slice_period(df_full, start, end)
        if len(df_period) < 100:
            print(f"\n  {label}: insufficient data ({len(df_period)} bars)")
            continue

        signals = generate_signals(df_period, ANALYSIS_CONFIG)
        if not signals:
            print(f"\n  {label}: no signals generated")
            continue

        metrics = execute_backtest(df_period, signals, "QQQ", ANALYSIS_CONFIG)
        features_df = extract_trade_features(df_period, signals, ANALYSIS_CONFIG)

        if features_df.empty:
            continue

        features_df["period"] = key
        features_df["period_label"] = label
        all_features.append(features_df)

        wins = features_df[features_df["win"] == True]
        losses = features_df[features_df["win"] == False]
        wr = len(wins) / len(features_df) * 100 if len(features_df) > 0 else 0
        net_pnl = metrics.get("net_pnl", 0)
        result_emoji = "✅" if net_pnl > 0 else "❌"

        period_summaries.append({
            "period": key,
            "label": label,
            "trades": len(features_df),
            "win_rate": round(wr, 1),
            "net_pnl": round(net_pnl, 2),
            "avg_adx": round(features_df["adx"].mean(), 1),
            "avg_rvol": round(features_df["rvol"].mean(), 2),
            "avg_vix": round(features_df["vix"].mean(), 1),
            "avg_atr_pct": round(features_df["atr_pct"].mean(), 4),
            "avg_mom_20": round(features_df["mom_20bar_pct"].mean(), 3),
        })

        print(f"\n  {result_emoji} {label}")
        print(f"     Net P&L: ${net_pnl:+.2f} | WR: {wr:.1f}% | Trades: {len(features_df)}")
        print(f"     Avg ADX: {features_df['adx'].mean():.1f} | "
              f"Avg RVOL: {features_df['rvol'].mean():.2f}x | "
              f"Avg VIX: {features_df['vix'].mean():.1f} | "
              f"Avg ATR%: {features_df['atr_pct'].mean():.3f}%")
        print(f"     Avg 20-bar Momentum: {features_df['mom_20bar_pct'].mean():+.3f}%")

    if not all_features:
        print("\n  ❌ No features extracted. Check data.")
        return

    combined_df = pd.concat(all_features, ignore_index=True)

    # ── Global win vs loss comparison ──
    global_wins = combined_df[combined_df["win"] == True]
    global_losses = combined_df[combined_df["win"] == False]

    print("\n\n" + "═" * 75)
    print("  FEATURE COMPARISON: WINNING vs LOSING TRADES (All Periods)")
    print("═" * 75)
    print(f"  Total trades analyzed: {len(combined_df)} | "
          f"Wins: {len(global_wins)} | Losses: {len(global_losses)}")

    comparison = compare_features(global_wins, global_losses)
    print(f"\n  {'Feature':35} {'Win Mean':>10} {'Lose Mean':>10} {'Diff%':>8} {'Edge':>15}")
    print(f"  {'─' * 73}")
    for _, row in comparison.iterrows():
        print(f"  {row['feature']:35} {row['win_mean']:>10.3f} {row['lose_mean']:>10.3f} "
              f"{row['diff_pct']:>7.1f}% {row['edge_direction']:>15}")

    # ── Period-level feature table ──
    print("\n\n" + "═" * 75)
    print("  PERIOD COMPARISON TABLE")
    print("═" * 75)
    print(f"  {'Period':30} {'WR%':>6} {'P&L':>10} {'ADX':>6} {'RVOL':>6} {'VIX':>6} {'ATR%':>7} {'Mom20':>8}")
    print(f"  {'─' * 73}")
    for s in period_summaries:
        emoji = "✅" if s["net_pnl"] > 0 else "❌"
        print(f"  {emoji} {s['label']:28} {s['win_rate']:>6.1f} ${s['net_pnl']:>+9.2f} "
              f"{s['avg_adx']:>6.1f} {s['avg_rvol']:>6.2f} {s['avg_vix']:>6.1f} "
              f"{s['avg_atr_pct']:>7.4f} {s['avg_mom_20']:>+8.3f}")

    # ── Optimal filter thresholds ──
    print("\n\n" + "═" * 75)
    print("  OPTIMAL FILTER THRESHOLDS (Based on All Trade Data)")
    print("═" * 75)
    cond_analysis = analyze_win_conditions(combined_df)
    print(f"  Baseline win rate: {cond_analysis['overall_win_rate']}% ({cond_analysis['total_trades']} trades)")
    print(f"\n  {'Filter Condition':45} {'WR%':>6} {'Trades':>7} {'Net P&L':>10}")
    print(f"  {'─' * 73}")
    for cond, stats in sorted(
        cond_analysis["conditions"].items(),
        key=lambda x: x[1]["win_rate"], reverse=True
    ):
        delta = stats["win_rate"] - cond_analysis["overall_win_rate"]
        delta_str = f"({delta:+.1f}%)"
        print(f"  {cond:45} {stats['win_rate']:>5.1f}% {delta_str:>7} "
              f"{stats['n_trades']:>7} ${stats['net_pnl']:>+9.2f}")

    # ── Key findings ──
    print("\n\n" + "═" * 75)
    print("  KEY FINDINGS — What Made Jan-Mar 2026 Different")
    print("═" * 75)

    jan_mar = combined_df[combined_df["period"].isin(["jan2026", "feb2026", "mar2026"])]
    rest = combined_df[~combined_df["period"].isin(["jan2026", "feb2026", "mar2026"])]

    if len(jan_mar) > 0 and len(rest) > 0:
        comparison2 = compare_features(jan_mar, rest)

        print(f"\n  Jan-Mar 2026: {len(jan_mar)} trades, "
              f"WR={jan_mar['win'].mean()*100:.1f}%, "
              f"P&L=${jan_mar['net_pnl'].sum():+.2f}")
        print(f"  All other periods: {len(rest)} trades, "
              f"WR={rest['win'].mean()*100:.1f}%, "
              f"P&L=${rest['net_pnl'].sum():+.2f}")

        print(f"\n  {'Feature':35} {'Jan-Mar 26':>12} {'Other Periods':>14} {'Diff%':>8}")
        print(f"  {'─' * 73}")
        for _, row in comparison2.head(8).iterrows():
            arrow = "↑" if row["win_mean"] > row["lose_mean"] else "↓"
            print(f"  {row['feature']:35} {row['win_mean']:>12.3f} {row['lose_mean']:>14.3f} "
                  f"  {arrow} {abs(row['diff_pct']):.1f}%")

        # Specific insight about VIX
        vix_jan = jan_mar["vix"].mean()
        vix_rest = rest["vix"].mean()
        print(f"\n  💡 VIX was {vix_jan:.1f} avg in Jan-Mar 2026 vs {vix_rest:.1f} in other periods")

        adx_jan = jan_mar["adx"].mean()
        adx_rest = rest["adx"].mean()
        print(f"  💡 ADX was {adx_jan:.1f} avg in Jan-Mar 2026 vs {adx_rest:.1f} in other periods")

        mom_jan = jan_mar["mom_20bar_pct"].mean()
        mom_rest = rest["mom_20bar_pct"].mean()
        print(f"  💡 20-bar momentum: {mom_jan:+.3f}% in Jan-Mar 2026 vs {mom_rest:+.3f}% other periods")

        rvol_jan = jan_mar["rvol"].mean()
        rvol_rest = rest["rvol"].mean()
        print(f"  💡 RVOL: {rvol_jan:.2f}x in Jan-Mar 2026 vs {rvol_rest:.2f}x other periods")

    # ── Recommended new filter parameters ──
    print("\n\n" + "═" * 75)
    print("  RECOMMENDED FILTER UPGRADES")
    print("═" * 75)

    # Find best combined threshold
    best_cond = max(
        cond_analysis["conditions"].items(),
        key=lambda x: x[1]["net_pnl"] if x[1]["n_trades"] >= 5 else -9999
    )
    print(f"\n  Best single filter: {best_cond[0]}")
    print(f"    → WR: {best_cond[1]['win_rate']}% | Trades: {best_cond[1]['n_trades']} | P&L: ${best_cond[1]['net_pnl']:+.2f}")

    if "COMBINED" in str(list(cond_analysis["conditions"].keys())):
        combo_key = [k for k in cond_analysis["conditions"] if "COMBINED" in k]
        if combo_key:
            combo = cond_analysis["conditions"][combo_key[0]]
            print(f"\n  Combined filter: {combo_key[0]}")
            print(f"    → WR: {combo['win_rate']}% | Trades: {combo['n_trades']} | P&L: ${combo['net_pnl']:+.2f}")

    print(f"\n  Suggested quant_engine.py updates:")
    if len(jan_mar) > 0 and len(rest) > 0:
        vix_jan = jan_mar["vix"].mean()
        adx_jan = jan_mar["adx"].mean()
        rvol_jan = jan_mar["rvol"].mean()
        print(f"    1. vix_threshold: 25.0 → {min(22.0, round(vix_jan + 2, 0)):.0f}.0")
        print(f"    2. adx_threshold: 18.0 → {max(20.0, round(adx_jan - 2, 0)):.0f}.0")
        print(f"    3. rvol_multiplier: 1.2 → {max(1.3, round(rvol_jan * 0.9, 1)):.1f}")
        print(f"    4. Add: momentum_filter (require 20-bar trend aligned with trade direction)")

    # ── Save results ──
    out_path = os.path.join(RESULTS_DIR, "signal_analysis.json")
    output = {
        "period_summaries": period_summaries,
        "total_trades_analyzed": len(combined_df),
        "global_win_rate": cond_analysis["overall_win_rate"],
        "optimal_conditions": cond_analysis["conditions"],
        "feature_comparison": comparison.to_dict("records") if not comparison.empty else [],
        "jan_mar_2026_vs_rest": {
            "jan_mar_trades": len(jan_mar) if len(jan_mar) > 0 else 0,
            "jan_mar_wr": round(jan_mar["win"].mean()*100, 1) if len(jan_mar) > 0 else 0,
            "jan_mar_pnl": round(jan_mar["net_pnl"].sum(), 2) if len(jan_mar) > 0 else 0,
            "rest_trades": len(rest) if len(rest) > 0 else 0,
            "rest_wr": round(rest["win"].mean()*100, 1) if len(rest) > 0 else 0,
            "rest_pnl": round(rest["net_pnl"].sum(), 2) if len(rest) > 0 else 0,
        }
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  💾 Full analysis saved: {out_path}")
    print("═" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Signal Quality Analysis")
    parser.add_argument("--period", type=str, default=None,
                        help=f"Run single period: {list(PERIODS.keys())}")
    parser.add_argument("--no-plots", action="store_true", help="Skip chart generation")
    args = parser.parse_args()

    print("\n" + "═" * 75)
    print("  SIGNAL QUALITY ANALYSIS — QQQ 15m Continuation Strategy")
    print("  Comparing Jan-Mar 2026 (winning) vs 2024-2025 (losing)")
    print("═" * 75)

    df = load_data()
    print(f"  Dataset: {len(df)} bars | {df.index[0].date()} → {df.index[-1].date()}")

    periods = [args.period] if args.period else None
    run_analysis(df, periods)


if __name__ == "__main__":
    main()
