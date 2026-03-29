#!/usr/bin/env python3
"""
DEEP HISTORICAL BACKTEST ENGINE
2-Year QQQ 15m Walk-Forward Validation across 8 Quarterly Windows

Integrates with quant_engine.py — same filters, same friction, deeper history.

Usage:
    python3 deep_backtest.py                          # Run full 2yr backtest
    python3 deep_backtest.py --source alpaca          # Use Alpaca API
    python3 deep_backtest.py --source polygon         # Use Polygon.io API
    python3 deep_backtest.py --source yfinance        # Use yfinance (default)
    python3 deep_backtest.py --no-download            # Use cached CSV only
    python3 deep_backtest.py --atr-sl 0.5 --rr 2.5   # Custom params

Data Sources:
    yfinance  — free, no API key, limited to ~60 days on 15m (auto-stitched)
    alpaca    — free tier, set ALPACA_API_KEY + ALPACA_SECRET_KEY in .env
    polygon   — free tier, set POLYGON_API_KEY in .env
"""
import sys
import os
import json
import argparse
import itertools
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

# ── Load .env if present ──
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, os.path.dirname(__file__))
from quant_engine import (
    Config, generate_signals, execute_backtest,
    apply_friction, walk_forward_validate,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEEP_CACHE = os.path.join(DATA_DIR, "QQQ_15m_2yr.csv")

# Live-ready params from walk-forward validation
LIVE_CONFIG = Config(
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
    # Regime filters
    use_vix_filter=True,
    vix_threshold=25.0,
    use_weekly_trend_filter=True,
    weekly_ema_period=10,
    etf_slippage_pct=0.00005,
    commission_round_trip=2.40,
    initial_capital=10000.0,
    risk_pct=0.01,
    # Deep walk-forward: 8 quarterly windows
    wf_windows=8,
    wf_train_days=42,   # ~2 months train
    wf_test_days=21,    # ~1 month test
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DEEP DATA INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

def download_alpaca(symbol: str = "QQQ", years: int = 2) -> pd.DataFrame:
    """
    Download 2yr 15m data from Alpaca Markets (free tier).
    Requires: ALPACA_API_KEY and ALPACA_SECRET_KEY in environment / .env
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError:
        print("  [alpaca] Install: pip install alpaca-py")
        return pd.DataFrame()

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        print("  [alpaca] Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        return pd.DataFrame()

    print(f"  [alpaca] Downloading {symbol} 15m ({years}yr)...")
    client = StockHistoricalDataClient(api_key, secret_key)

    end = datetime.now()
    start = end - timedelta(days=years * 365)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute * 15,
        start=start,
        end=end,
        adjustment="all",
    )
    bars = client.get_stock_bars(request).df

    if bars.empty:
        print("  [alpaca] No data returned.")
        return pd.DataFrame()

    # Normalize to standard OHLCV format
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")

    bars.index = pd.to_datetime(bars.index, utc=True).tz_convert("US/Eastern")
    bars = bars.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    bars = bars[["Open", "High", "Low", "Close", "Volume"]]

    print(f"  [alpaca] ✅ {len(bars)} bars downloaded")
    return bars


def download_massive(symbol: str = "QQQ", years: int = 2) -> pd.DataFrame:
    """
    Download 2yr 15m data from Massive.com (formerly Polygon.io).
    Free tier supports 2yr+ of historical 15m data.
    Requires: POLYGON_API_KEY or MASSIVE_API_KEY in environment / .env
    """
    try:
        from massive import RESTClient
    except ImportError:
        print("  [massive] Install: pip3 install -U massive")
        return pd.DataFrame()

    api_key = os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("  [massive] Set MASSIVE_API_KEY in .env")
        return pd.DataFrame()

    print(f"  [massive] Downloading {symbol} 15m ({years}yr)...")
    client = RESTClient(api_key)

    end = datetime.now()
    start = end - timedelta(days=years * 365)

    import time

    bars = []
    # Fetch in 6-month chunks to avoid rate limits
    chunk_months = 6
    current_end = end
    current_start = end - timedelta(days=chunk_months * 30)
    chunk_num = 0
    total_chunks = (years * 12) // chunk_months

    while current_start >= start - timedelta(days=30):
        chunk_num += 1
        chunk_start_str = max(current_start, start).strftime("%Y-%m-%d")
        chunk_end_str = current_end.strftime("%Y-%m-%d")

        print(f"  [massive] Chunk {chunk_num}/{total_chunks}: {chunk_start_str} → {chunk_end_str}...")

        retries = 3
        for attempt in range(retries):
            try:
                for agg in client.list_aggs(
                    symbol,
                    15, "minute",
                    chunk_start_str,
                    chunk_end_str,
                    limit=50000,
                    adjusted=True,
                ):
                    bars.append({
                        "timestamp": pd.Timestamp(agg.timestamp, unit="ms", tz="UTC"),
                        "Open": agg.open, "High": agg.high,
                        "Low": agg.low, "Close": agg.close,
                        "Volume": agg.volume,
                    })
                break  # success
            except Exception as e:
                if "429" in str(e) and attempt < retries - 1:
                    wait = 15 * (attempt + 1)
                    print(f"  [massive] Rate limited — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  [massive] Chunk error: {e}")
                    break

        # Small pause between chunks
        time.sleep(2)

        current_end = current_start - timedelta(days=1)
        current_start = current_end - timedelta(days=chunk_months * 30)

        if current_end < start:
            break

    if not bars:
        print("  [massive] No data returned — check API key and subscription tier.")
        return pd.DataFrame()

    df = pd.DataFrame(bars).set_index("timestamp")
    df.index = df.index.tz_convert("US/Eastern")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    print(f"  [massive] ✅ {len(df)} bars | {df.index[0].date()} → {df.index[-1].date()}")
    return df


# Keep polygon as alias for backward compat
def download_polygon(symbol: str = "QQQ", years: int = 2) -> pd.DataFrame:
    return download_massive(symbol, years)


def download_yfinance_stitched(symbol: str = "QQQ", years: int = 2) -> pd.DataFrame:
    """
    Download QQQ 15m data via yfinance.
    ⚠️  HARD LIMIT: Yahoo Finance only provides 15m data for the last 60 days.
    For 2yr deep backtest, use --source alpaca or --source polygon instead.
    This function will only return ~60 days regardless of years parameter.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  [yfinance] Install: pip install yfinance")
        return pd.DataFrame()

    print(f"  [yfinance] Stitching {years}yr of QQQ 15m data (60-day chunks)...")

    end = datetime.now()
    start = end - timedelta(days=years * 365)

    all_chunks = []
    chunk_end = end
    chunk_size = timedelta(days=58)  # slightly under 60 to be safe

    while chunk_end > start:
        chunk_start = max(chunk_end - chunk_size, start)
        try:
            ticker = yf.Ticker(symbol)
            df_chunk = ticker.history(
                start=chunk_start.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval="15m",
            )
            if not df_chunk.empty:
                all_chunks.append(df_chunk)
                print(f"  [yfinance] Chunk {chunk_start.strftime('%Y-%m-%d')} → "
                      f"{chunk_end.strftime('%Y-%m-%d')}: {len(df_chunk)} bars")
        except Exception as e:
            print(f"  [yfinance] Chunk error: {e}")

        chunk_end = chunk_start - timedelta(days=1)

    if not all_chunks:
        print("  [yfinance] No data downloaded.")
        return pd.DataFrame()

    df = pd.concat(all_chunks)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # Normalize columns
    df = df.rename(columns={
        "Open": "Open", "High": "High", "Low": "Low",
        "Close": "Close", "Volume": "Volume",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    print(f"  [yfinance] ✅ {len(df)} bars total after dedup")
    return df


def load_deep_data(source: str = "yfinance", force_download: bool = False) -> pd.DataFrame:
    """
    Load 2yr QQQ 15m data. Uses cache if available, downloads otherwise.
    """
    # Check cache
    if os.path.exists(DEEP_CACHE) and not force_download:
        print(f"  [cache] Loading {DEEP_CACHE}...")
        df = pd.read_csv(DEEP_CACHE, index_col=0, parse_dates=True)
        # Ensure timezone
        if df.index.tz is None:
            try:
                df.index = df.index.tz_localize("US/Eastern")
            except Exception:
                df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
        else:
            df.index = df.index.tz_convert("US/Eastern")

        # Check if data is recent enough (within 2 days)
        latest = df.index[-1]
        age_days = (datetime.now(tz=df.index.tz) - latest).days
        if age_days <= 2:
            print(f"  [cache] ✅ {len(df)} bars | Latest: {latest.date()} | Age: {age_days}d")
            return df
        else:
            print(f"  [cache] Stale ({age_days}d old) — re-downloading...")

    # Download
    print(f"\n  Downloading 2yr QQQ 15m data via {source}...")
    if source == "alpaca":
        df = download_alpaca()
    elif source in ("polygon", "massive"):
        df = download_massive()
    else:
        df = download_yfinance_stitched()

    if df.empty:
        print("  ❌ Download failed. Check API keys or try a different source.")
        sys.exit(1)

    # Save to cache
    df.to_csv(DEEP_CACHE)
    print(f"  [cache] Saved to {DEEP_CACHE}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. QUARTERLY WALK-FORWARD ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def quarterly_walk_forward(df: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """
    8-window quarterly walk-forward over 2 years.
    Each window: ~2 months train → 1 month test (out-of-sample).
    """
    # Ensure DatetimeIndex
    df_work = df.copy()
    if not isinstance(df_work.index, pd.DatetimeIndex):
        df_work.index = pd.to_datetime(df_work.index, utc=True)
    if df_work.index.tz is None:
        try:
            df_work.index = df_work.index.tz_localize("US/Eastern")
        except Exception:
            df_work.index = df_work.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        df_work.index = df_work.index.tz_convert("US/Eastern")

    # Get unique trading dates
    dates_series = df_work.index.date
    unique_dates = sorted(set(dates_series))
    total_days = len(unique_dates)

    print(f"\n  Deep Walk-Forward: {total_days} trading days | {cfg.wf_windows} quarterly windows")
    print(f"  Train: {cfg.wf_train_days}d (~2 months) | Test: {cfg.wf_test_days}d (~1 month)")

    # Parameter grid
    combos = list(itertools.product(cfg.opt_atr_sl, cfg.opt_rr, cfg.opt_disp))
    print(f"  Testing {len(combos)} parameter combos per window\n")

    days_per_window = total_days // cfg.wf_windows
    windows = []
    oos_equity = [cfg.initial_capital]
    running_capital = cfg.initial_capital

    for w in range(cfg.wf_windows):
        w_start = w * days_per_window
        w_end = w_start + days_per_window if w < cfg.wf_windows - 1 else total_days

        train_end = w_start + cfg.wf_train_days
        test_start = train_end
        test_end = w_end

        if train_end >= total_days or test_start >= total_days:
            print(f"  Window {w+1}: Skipping — insufficient data")
            continue

        train_dates = set(unique_dates[w_start:train_end])
        test_dates = set(unique_dates[test_start:min(test_end, total_days)])

        if not test_dates:
            continue

        train_mask = pd.Series([d in train_dates for d in dates_series], index=df_work.index)
        test_mask = pd.Series([d in test_dates for d in dates_series], index=df_work.index)

        df_train = df_work[train_mask].copy()
        df_test = df_work[test_mask].copy()

        if len(df_train) < 100 or len(df_test) < 30:
            print(f"  Window {w+1}: Skipping — too few bars (train={len(df_train)}, test={len(df_test)})")
            continue

        quarter_label = f"Q{w+1} ({unique_dates[w_start].strftime('%b %Y')}→{unique_dates[min(test_end-1, total_days-1)].strftime('%b %Y')})"
        print(f"  ── {quarter_label} ──")
        print(f"     Train: {unique_dates[w_start]} → {unique_dates[min(train_end-1, total_days-1)]} ({len(df_train)} bars)")
        print(f"     Test:  {unique_dates[test_start]} → {unique_dates[min(test_end-1, total_days-1)]} ({len(df_test)} bars)")

        # ── Optimize on train ──
        best_score = -999999
        best_params = {"atr_sl": 0.5, "rr": 2.5, "disp": 1.0}
        best_train_pnl = 0

        for atr_sl, rr, disp in combos:
            test_cfg = Config(
                atr_multiplier_sl=atr_sl, rr_ratio=rr, displacement_threshold=disp,
                session_filter=cfg.session_filter, use_htf_filter=cfg.use_htf_filter,
                use_adx_filter=cfg.use_adx_filter, use_rvol_filter=cfg.use_rvol_filter,
                adx_threshold=cfg.adx_threshold, rvol_multiplier=cfg.rvol_multiplier,
                ema_period=cfg.ema_period, adx_period=cfg.adx_period,
                rvol_period=cfg.rvol_period, etf_slippage_pct=cfg.etf_slippage_pct,
                commission_round_trip=cfg.commission_round_trip,
                initial_capital=cfg.initial_capital, risk_pct=cfg.risk_pct,
            )
            sigs = generate_signals(df_train, test_cfg)
            if len(sigs) < 2:
                continue
            m = execute_backtest(df_train, sigs, "QQQ", test_cfg)
            pnl = m.get("net_pnl", 0)
            trades = m.get("total_trades", 0)
            pf = m.get("profit_factor", 0)
            if isinstance(pf, str):
                pf = 10.0
            score = pnl * 0.5 + pf * 100 * 0.3 + trades * 0.2
            if score > best_score:
                best_score = score
                best_params = {"atr_sl": atr_sl, "rr": rr, "disp": disp}
                best_train_pnl = pnl

        # ── Test on OOS ──
        oos_cfg = Config(
            atr_multiplier_sl=best_params["atr_sl"],
            rr_ratio=best_params["rr"],
            displacement_threshold=best_params["disp"],
            session_filter=cfg.session_filter, use_htf_filter=cfg.use_htf_filter,
            use_adx_filter=cfg.use_adx_filter, use_rvol_filter=cfg.use_rvol_filter,
            adx_threshold=cfg.adx_threshold, rvol_multiplier=cfg.rvol_multiplier,
            ema_period=cfg.ema_period, adx_period=cfg.adx_period,
            rvol_period=cfg.rvol_period, etf_slippage_pct=cfg.etf_slippage_pct,
            commission_round_trip=cfg.commission_round_trip,
            initial_capital=running_capital, risk_pct=cfg.risk_pct,
        )
        oos_sigs = generate_signals(df_test, oos_cfg)
        oos_m = execute_backtest(df_test, oos_sigs, "QQQ", oos_cfg)

        oos_pnl = oos_m.get("net_pnl", 0)
        oos_trades = oos_m.get("total_trades", 0)
        oos_wr = oos_m.get("win_rate", 0)
        oos_gross = oos_m.get("gross_pnl", 0)
        oos_comm = oos_m.get("total_commission", 0)
        oos_slip = oos_m.get("total_slippage", 0)
        running_capital += oos_pnl

        for eq in oos_m.get("equity_curve", [])[1:]:
            oos_equity.append(eq)

        status = "✅ PASS" if oos_pnl > 0 else "❌ FAIL"
        print(f"     Best params: ATR={best_params['atr_sl']} RR={best_params['rr']} Disp={best_params['disp']}")
        print(f"     Train P&L: ${best_train_pnl:+.2f} | OOS: Gross ${oos_gross:+.2f} → Net ${oos_pnl:+.2f} "
              f"| WR: {oos_wr:.1f}% | Trades: {oos_trades} | {status}")
        print(f"     Friction:  Comm -${oos_comm:.2f} | Slip -${oos_slip:.2f}")
        print()

        windows.append({
            "window": w + 1,
            "label": quarter_label,
            "period_start": str(unique_dates[w_start]),
            "period_end": str(unique_dates[min(test_end-1, total_days-1)]),
            "best_params": best_params,
            "train_pnl": round(best_train_pnl, 2),
            "oos_gross_pnl": round(oos_gross, 2),
            "oos_net_pnl": round(oos_pnl, 2),
            "oos_commission": round(oos_comm, 2),
            "oos_slippage": round(oos_slip, 2),
            "oos_trades": oos_trades,
            "oos_win_rate": round(oos_wr, 1),
            "profitable": oos_pnl > 0,
            "capital_after": round(running_capital, 2),
        })

    return {
        "windows": windows,
        "oos_equity": oos_equity,
        "final_capital": round(running_capital, 2),
        "total_oos_pnl": round(running_capital - LIVE_CONFIG.initial_capital, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. REPORTING — Quarterly Robustness Matrix
# ═══════════════════════════════════════════════════════════════════════════════

def print_robustness_matrix(results: Dict[str, Any]) -> None:
    windows = results["windows"]
    profitable = sum(1 for w in windows if w["profitable"])
    total = len(windows)

    if total == 0:
        print("\n  ❌ NO WINDOWS PROCESSED")
        print("  Cause: Insufficient historical data.")
        print("  Fix:   Use Alpaca or Polygon for 2yr history:")
        print("         python3 deep_backtest.py --source alpaca")
        print("\n  Setup:")
        print("  1. Sign up free at https://alpaca.markets")
        print("  2. Generate Paper Trading API keys")
        print("  3. Add to .env file:")
        print("     ALPACA_API_KEY=your_key_id")
        print("     ALPACA_SECRET_KEY=your_secret_key")
        print("  4. Run: python3 deep_backtest.py --source alpaca")
        return

    robust = profitable / total >= 0.625  # 5/8 threshold

    print("\n" + "═" * 75)
    print("  QUARTERLY ROBUSTNESS MATRIX — QQQ 15m Continuation (2yr Deep Backtest)")
    print("═" * 75)
    print(f"  {'Window':>8} {'Period':>22} {'Params':>18} {'Gross':>10} {'Net':>10} {'WR':>7} {'Trades':>7} {'Status':>8}")
    print(f"  {'─' * 73}")

    for w in windows:
        p = w["best_params"]
        param_str = f"ATR={p['atr_sl']} RR={p['rr']}"
        status = "✅ PASS" if w["profitable"] else "❌ FAIL"
        print(f"  {w['window']:>8} {w['label']:>22} {param_str:>18} "
              f"${w['oos_gross_pnl']:>+9.2f} ${w['oos_net_pnl']:>+9.2f} "
              f"{w['oos_win_rate']:>6.1f}% {w['oos_trades']:>7} {status:>8}")

    print(f"  {'─' * 73}")

    total_gross = sum(w["oos_gross_pnl"] for w in windows)
    total_net = sum(w["oos_net_pnl"] for w in windows)
    total_comm = sum(w["oos_commission"] for w in windows)
    total_slip = sum(w["oos_slippage"] for w in windows)
    total_trades = sum(w["oos_trades"] for w in windows)

    print(f"\n  SUMMARY:")
    print(f"    Profitable Windows: {profitable}/{total} ({profitable/total*100:.0f}%)")
    print(f"    Gross P&L (2yr):   ${total_gross:+.2f}")
    print(f"    Commission (2yr):  -${total_comm:.2f}")
    print(f"    Slippage (2yr):    -${total_slip:.2f}")
    print(f"    Net P&L (2yr):     ${total_net:+.2f}")
    print(f"    Total Trades:       {total_trades}")
    print(f"    Final Capital:     ${results['final_capital']:,.2f} (started $10,000)")
    print(f"    Total Return:      {results['total_oos_pnl'] / LIVE_CONFIG.initial_capital * 100:+.2f}%")

    print(f"\n  VERDICT: {'✅ ROBUST — INSTITUTIONALLY VALIDATED' if robust else '⚠️  FRAGILE — NOT LIVE-READY'}")
    if robust:
        # Median params from winning windows
        winning = [w["best_params"] for w in windows if w["profitable"]]
        live_atr = round(float(np.median([p["atr_sl"] for p in winning])), 2)
        live_rr = round(float(np.median([p["rr"] for p in winning])), 2)
        live_disp = round(float(np.median([p["disp"] for p in winning])), 2)
        print(f"\n  📊 2YR LIVE-READY PARAMS:")
        print(f"     ATR SL: {live_atr}x | Risk/Reward: {live_rr}:1 | Displacement: {live_disp}x")
        print(f"     (Median of {len(winning)} profitable quarters)")

    print("═" * 75)


def save_equity_curve_png(oos_equity: List[float]) -> None:
    """Save equity curve as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(oos_equity, color="#00d4ff", linewidth=1.5, label="OOS Equity")
        ax.axhline(y=LIVE_CONFIG.initial_capital, color="gray", linestyle="--", alpha=0.5, label="Starting Capital")

        # Shade wins/losses
        ax.fill_between(range(len(oos_equity)), LIVE_CONFIG.initial_capital, oos_equity,
                        where=[e >= LIVE_CONFIG.initial_capital for e in oos_equity],
                        alpha=0.15, color="#00ff88")
        ax.fill_between(range(len(oos_equity)), LIVE_CONFIG.initial_capital, oos_equity,
                        where=[e < LIVE_CONFIG.initial_capital for e in oos_equity],
                        alpha=0.15, color="#ff4444")

        ax.set_title("2-Year Walk-Forward OOS Equity — QQQ 15m Continuation Strategy",
                     fontsize=14, color="white", pad=15)
        ax.set_xlabel("Trade #", color="gray")
        ax.set_ylabel("Account Equity ($)", color="gray")
        ax.set_facecolor("#0a0e1a")
        fig.set_facecolor("#0a0e1a")
        ax.tick_params(colors="gray")
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(facecolor="#0a0e1a", labelcolor="gray")

        png_path = os.path.join(RESULTS_DIR, "deep_backtest_equity_QQQ.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\n  📈 Equity curve saved: {png_path}")
    except ImportError:
        print("  ⚠️  matplotlib not installed — skipping equity curve PNG")


def save_results_json(results: Dict[str, Any]) -> None:
    """Save full results to JSON."""
    out = {k: v for k, v in results.items() if k != "oos_equity"}
    path = os.path.join(RESULTS_DIR, "deep_backtest_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  💾 Results saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Deep 2-Year QQQ Backtest Engine")
    parser.add_argument("--source", choices=["yfinance", "alpaca", "polygon", "massive"],
                        default="yfinance", help="Data source (default: yfinance)")
    parser.add_argument("--no-download", action="store_true",
                        help="Use cached CSV only, skip download check")
    parser.add_argument("--force-download", action="store_true",
                        help="Force fresh download even if cache exists")
    parser.add_argument("--atr-sl", type=float, default=None, help="Override ATR SL multiplier")
    parser.add_argument("--rr", type=float, default=None, help="Override Risk/Reward ratio")
    parser.add_argument("--windows", type=int, default=8, help="Number of walk-forward windows (default: 8)")
    args = parser.parse_args()

    cfg = LIVE_CONFIG
    if args.atr_sl:
        cfg.atr_multiplier_sl = args.atr_sl
    if args.rr:
        cfg.rr_ratio = args.rr
    cfg.wf_windows = args.windows

    print("\n" + "═" * 75)
    print("  DEEP HISTORICAL BACKTEST ENGINE — QQQ 15m (2 Years)")
    print("  Institutional Continuation Strategy | Walk-Forward Validated")
    print("═" * 75)
    print(f"  Data Source:    {args.source}")
    print(f"  ATR SL:         {cfg.atr_multiplier_sl}x")
    print(f"  Risk/Reward:    {cfg.rr_ratio}:1")
    print(f"  Displacement:   {cfg.displacement_threshold}x ATR")
    print(f"  Filters:        1H EMA | ADX>18 | RVOL>1.2x | Session windows")
    print(f"  Friction:       ${cfg.commission_round_trip}/RT + {cfg.etf_slippage_pct*100:.3f}% slip")
    print(f"  Walk-Forward:   {cfg.wf_windows} windows | {cfg.wf_train_days}d train / {cfg.wf_test_days}d test")

    # ── Load data ──
    print()
    if args.no_download:
        if not os.path.exists(DEEP_CACHE):
            print(f"  ❌ No cache found at {DEEP_CACHE}. Run without --no-download first.")
            sys.exit(1)
        df = pd.read_csv(DEEP_CACHE, index_col=0, parse_dates=True)
        if df.index.tz is None:
            try:
                df.index = df.index.tz_localize("US/Eastern")
            except Exception:
                df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
        print(f"  [cache] Loaded {len(df)} bars from {DEEP_CACHE}")
    else:
        df = load_deep_data(source=args.source, force_download=args.force_download)

    print(f"\n  Dataset: {len(df)} bars | {df.index[0].date()} → {df.index[-1].date()}")

    # ── Run quarterly walk-forward ──
    results = quarterly_walk_forward(df, cfg)

    # ── Print robustness matrix ──
    print_robustness_matrix(results)

    # ── Save outputs ──
    save_equity_curve_png(results["oos_equity"])
    save_results_json(results)

    print("\n" + "═" * 75)
    print("  DONE.")
    print("═" * 75 + "\n")


if __name__ == "__main__":
    main()
