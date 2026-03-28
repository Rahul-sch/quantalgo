#!/usr/bin/env python3
"""
INSTITUTIONAL-GRADE CONTINUATION ENGINE
Vectorized execution, multi-timeframe confluence, anti-chop filters,
walk-forward validation, and realistic friction modeling.

Usage:
    python3 quant_engine.py                    # Run backtest with defaults
    python3 quant_engine.py --optimize         # Walk-forward optimization
    python3 quant_engine.py --symbol QQQ       # Single symbol
    python3 quant_engine.py --no-filter        # Disable session filter
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
from data_fetcher import download_data, INSTRUMENTS

# Ensure QQQ is available
if "QQQ" not in INSTRUMENTS:
    INSTRUMENTS["QQQ"] = "QQQ"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 0. CENTRALIZED CONFIG — NO HARDCODED VALUES IN LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """All tunable parameters in one place."""
    # ── Tickers ──
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ"])
    is_futures: Dict[str, bool] = field(default_factory=lambda: {
        "SPY": False, "QQQ": False, "ES": True, "NQ": True,
    })

    # ── Capital & Risk ──
    initial_capital: float = 10000.0
    risk_pct: float = 0.01  # 1% risk per trade

    # ── Strategy Parameters ──
    atr_period: int = 14
    atr_multiplier_sl: float = 0.5
    rr_ratio: float = 3.0  # must overcome friction — need home runs
    displacement_threshold: float = 1.0
    swing_lookback: int = 10
    fvg_max_age: int = 20       # max bars old an FVG can be
    trend_lookback: int = 20

    # ── Session Filter (EST) ──
    session_filter: bool = True
    am_start: int = 570         # 9:30 in minutes
    am_end: int = 690           # 11:30
    pm_start: int = 810         # 13:30
    pm_end: int = 930           # 15:30

    # ── Confluence: 1H EMA ──
    ema_period: int = 20
    use_htf_filter: bool = True

    # ── Anti-Chop ──
    adx_period: int = 14
    adx_threshold: float = 18.0  # relaxed — still filters chop
    use_adx_filter: bool = True
    rvol_multiplier: float = 1.2  # relaxed from 1.5 — still confirms volume
    rvol_period: int = 10
    use_rvol_filter: bool = True

    # ── Friction ──
    futures_slippage_pts: float = 0.5
    etf_slippage_pct: float = 0.00005  # 0.005% (realistic for liquid ETFs)
    commission_round_trip: float = 2.40

    # ── Walk-Forward ──
    wf_windows: int = 4
    wf_train_days: int = 10
    wf_test_days: int = 5

    # ── Optimizer Grid ──
    opt_atr_sl: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.25])
    opt_rr: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0])
    opt_disp: List[float] = field(default_factory=lambda: [1.0, 1.3])


CFG = Config()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FRICTION MODULE — Realistic Execution Costs
# ═══════════════════════════════════════════════════════════════════════════════

def apply_friction(
    entry: float, exit_price: float, direction: str, symbol: str, cfg: Config
) -> Tuple[float, float, float]:
    """
    Apply slippage + commission. Returns (adj_entry, adj_exit, commission).
    """
    is_fut = cfg.is_futures.get(symbol, False)

    if is_fut:
        slip = cfg.futures_slippage_pts
        if direction == "buy":
            adj_entry = entry + slip
            adj_exit = exit_price - slip
        else:
            adj_entry = entry - slip
            adj_exit = exit_price + slip
    else:
        slip_pct = cfg.etf_slippage_pct
        if direction == "buy":
            adj_entry = entry * (1 + slip_pct)
            adj_exit = exit_price * (1 - slip_pct)
        else:
            adj_entry = entry * (1 - slip_pct)
            adj_exit = exit_price * (1 + slip_pct)

    return adj_entry, adj_exit, cfg.commission_round_trip


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONFLUENCE MODULE — 1H EMA (Lookahead-Bias Free)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_htf_ema_signal(df_5m: pd.DataFrame, ema_period: int = 20) -> pd.Series:
    """
    Resample 5m data to 1H, compute EMA, then map back to 5m bars.
    CRITICAL: Uses .shift(1) on 1H to prevent lookahead bias —
    5m bars only see the PREVIOUS CLOSED 1H candle's EMA.

    Returns: pd.Series aligned to df_5m index with values: 1 (bullish), -1 (bearish), 0 (neutral)
    """
    df_work = df_5m.copy()

    # Ensure DatetimeIndex — bulletproof conversion
    if not isinstance(df_work.index, pd.DatetimeIndex):
        df_work.index = pd.to_datetime(df_work.index, utc=True)
    if df_work.index.tz is None:
        try:
            df_work.index = df_work.index.tz_localize("US/Eastern")
        except Exception:
            df_work.index = df_work.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        df_work.index = df_work.index.tz_convert("US/Eastern")

    # Resample to 1H
    df_1h = df_work["Close"].resample("1h").last().dropna()
    ema_1h = df_1h.ewm(span=ema_period, adjust=False).mean()

    # EMA slope: positive = bullish, negative = bearish
    ema_slope = ema_1h.diff()

    # SHIFT by 1 to prevent lookahead: 5m bar reads PREVIOUS closed 1H candle
    ema_slope_shifted = ema_slope.shift(1)

    # Map signal to 5m bars using forward-fill (each 5m bar gets the last known 1H signal)
    signal_1h = pd.Series(0, index=ema_slope_shifted.index, dtype=int)
    signal_1h[ema_slope_shifted > 0] = 1
    signal_1h[ema_slope_shifted < 0] = -1

    # Reindex to 5m and forward-fill — ensure no NaN gaps
    signal_5m = signal_1h.reindex(df_work.index, method="ffill")
    signal_5m = signal_5m.ffill().fillna(0).astype(int)

    return signal_5m


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANTI-CHOP MODULE — ADX & RVOL Filters (Vectorized)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Vectorized ADX calculation."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()

    return adx.fillna(0)


def compute_rvol(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Relative Volume: current volume / SMA of volume."""
    vol_sma = df["Volume"].rolling(period).mean()
    return (df["Volume"] / vol_sma.replace(0, np.nan)).fillna(0)


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION — Vectorized Pre-computation + Targeted Loop
# ═══════════════════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Pre-compute all indicators in vectorized fashion. No loops."""
    df_ind = df.copy()

    # ATR
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df_ind["atr"] = tr.rolling(cfg.atr_period).mean()

    # Trend: SMA short vs long
    df_ind["sma_short"] = df["Close"].rolling(5).mean()
    df_ind["sma_long"] = df["Close"].rolling(cfg.trend_lookback).mean()
    df_ind["trend"] = 0
    df_ind.loc[df_ind["sma_short"] > df_ind["sma_long"], "trend"] = 1
    df_ind.loc[df_ind["sma_short"] < df_ind["sma_long"], "trend"] = -1

    # ADX
    if cfg.use_adx_filter:
        df_ind["adx"] = compute_adx(df, cfg.adx_period)
    else:
        df_ind["adx"] = 100.0  # always pass

    # RVOL
    if cfg.use_rvol_filter:
        df_ind["rvol"] = compute_rvol(df, cfg.rvol_period)
    else:
        df_ind["rvol"] = 10.0  # always pass

    # Candle range
    df_ind["candle_range"] = df["High"] - df["Low"]

    # Displacement flag
    df_ind["is_displacement"] = df_ind["candle_range"] > (df_ind["atr"] * cfg.displacement_threshold)

    # Session filter (EST) — bulletproof timezone handling
    if cfg.session_filter:
        try:
            idx = df.index
            # Step 1: Force to DatetimeIndex no matter what
            if not isinstance(idx, pd.DatetimeIndex):
                idx = pd.to_datetime(idx, utc=True)
            # Step 2: If no timezone, assume US/Eastern (yfinance US stocks)
            if idx.tz is None:
                try:
                    idx = idx.tz_localize("US/Eastern")
                except Exception:
                    idx = idx.tz_localize("UTC").tz_convert("US/Eastern")
            else:
                # Convert whatever timezone to Eastern
                idx = idx.tz_convert("US/Eastern")

            minutes = idx.hour * 60 + idx.minute
            in_am = (minutes >= cfg.am_start) & (minutes <= cfg.am_end)
            in_pm = (minutes >= cfg.pm_start) & (minutes <= cfg.pm_end)
            df_ind["in_session"] = (in_am | in_pm).values
        except Exception:
            # If all else fails, let everything through
            df_ind["in_session"] = True
    else:
        df_ind["in_session"] = True

    # HTF EMA signal
    if cfg.use_htf_filter:
        df_ind["htf_signal"] = compute_htf_ema_signal(df, cfg.ema_period)
    else:
        df_ind["htf_signal"] = 0

    return df_ind


def detect_fvgs_vectorized(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Vectorized FVG detection.
    Returns: bull_fvg_top, bull_fvg_bot, bear_fvg_top, bear_fvg_bot (Series, NaN where no FVG)
    """
    h = df["High"].values
    l = df["Low"].values
    n = len(df)

    bull_top = np.full(n, np.nan)
    bull_bot = np.full(n, np.nan)
    bear_top = np.full(n, np.nan)
    bear_bot = np.full(n, np.nan)

    for i in range(2, n):
        # Bullish FVG: candle[i-2] high < candle[i] low
        if h[i-2] < l[i]:
            bull_top[i] = l[i]
            bull_bot[i] = h[i-2]
        # Bearish FVG: candle[i-2] low > candle[i] high
        if l[i-2] > h[i]:
            bear_top[i] = l[i-2]
            bear_bot[i] = h[i]

    return (pd.Series(bull_top, index=df.index), pd.Series(bull_bot, index=df.index),
            pd.Series(bear_top, index=df.index), pd.Series(bear_bot, index=df.index))


def find_swings_vectorized(df: pd.DataFrame, lookback: int = 10) -> Tuple[pd.Series, pd.Series]:
    """Vectorized swing high/low detection using rolling max/min."""
    window = 2 * lookback + 1
    roll_max = df["High"].rolling(window, center=True).max()
    roll_min = df["Low"].rolling(window, center=True).min()

    swing_highs = pd.Series(np.nan, index=df.index)
    swing_lows = pd.Series(np.nan, index=df.index)

    swing_highs[df["High"] == roll_max] = df["High"][df["High"] == roll_max]
    swing_lows[df["Low"] == roll_min] = df["Low"][df["Low"] == roll_min]

    return swing_highs, swing_lows


def generate_signals(df: pd.DataFrame, cfg: Config) -> List[Dict[str, Any]]:
    """
    Generate continuation signals with all filters applied.
    Pre-computes everything vectorized, then does a single pass for FVG matching.
    """
    if len(df) < 50:
        return []

    # Vectorized pre-computation
    ind = compute_indicators(df, cfg)
    bull_fvg_top, bull_fvg_bot, bear_fvg_top, bear_fvg_bot = detect_fvgs_vectorized(df)
    swing_highs, swing_lows = find_swings_vectorized(df, cfg.swing_lookback)

    signals = []
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    atr = ind["atr"].values
    trend = ind["trend"].values
    adx = ind["adx"].values
    rvol = ind["rvol"].values
    is_disp = ind["is_displacement"].values
    in_session = ind["in_session"].values
    htf = ind["htf_signal"].values

    # Diagnostics
    _skip = {"atr": 0, "session": 0, "adx": 0, "disp": 0, "htf": 0, "rvol": 0, "no_fvg": 0, "passed": 0}

    # Active FVG tracking (ring buffer for efficiency)
    active_bull_fvgs = []  # (bar, top, bot)
    active_bear_fvgs = []

    for i in range(30, len(df)):
        # ── Pre-filter: skip quickly ──
        if np.isnan(atr[i]) or atr[i] <= 0:
            _skip["atr"] += 1
            continue
        if not in_session[i]:
            _skip["session"] += 1
            # Still track FVGs even outside session
            if not np.isnan(bull_fvg_top.iloc[i]):
                active_bull_fvgs.append((i, float(bull_fvg_top.iloc[i]), float(bull_fvg_bot.iloc[i])))
            if not np.isnan(bear_fvg_top.iloc[i]):
                active_bear_fvgs.append((i, float(bear_fvg_top.iloc[i]), float(bear_fvg_bot.iloc[i])))
            active_bull_fvgs = [(b, t, bt) for b, t, bt in active_bull_fvgs if i - b <= cfg.fvg_max_age]
            active_bear_fvgs = [(b, t, bt) for b, t, bt in active_bear_fvgs if i - b <= cfg.fvg_max_age]
            continue
        if adx[i] < cfg.adx_threshold:
            _skip["adx"] += 1
            # Still track FVGs
            if not np.isnan(bull_fvg_top.iloc[i]):
                active_bull_fvgs.append((i, float(bull_fvg_top.iloc[i]), float(bull_fvg_bot.iloc[i])))
            if not np.isnan(bear_fvg_top.iloc[i]):
                active_bear_fvgs.append((i, float(bear_fvg_top.iloc[i]), float(bear_fvg_bot.iloc[i])))
            active_bull_fvgs = [(b, t, bt) for b, t, bt in active_bull_fvgs if i - b <= cfg.fvg_max_age]
            active_bear_fvgs = [(b, t, bt) for b, t, bt in active_bear_fvgs if i - b <= cfg.fvg_max_age]
            continue

        _skip["passed"] += 1

        # Track FVGs (use .iloc to avoid FutureWarning)
        if not np.isnan(bull_fvg_top.iloc[i]):
            active_bull_fvgs.append((i, float(bull_fvg_top.iloc[i]), float(bull_fvg_bot.iloc[i])))
        if not np.isnan(bear_fvg_top.iloc[i]):
            active_bear_fvgs.append((i, float(bear_fvg_top.iloc[i]), float(bear_fvg_bot.iloc[i])))

        # Prune old FVGs
        active_bull_fvgs = [(b, t, bt) for b, t, bt in active_bull_fvgs if i - b <= cfg.fvg_max_age]
        active_bear_fvgs = [(b, t, bt) for b, t, bt in active_bear_fvgs if i - b <= cfg.fvg_max_age]

        # ── BULLISH CONTINUATION ──
        if trend[i] == 1:
            # HTF filter: only long if 1H EMA bullish (or filter disabled)
            if cfg.use_htf_filter and htf[i] < 0:
                continue

            # Find drawn liquidity (nearest swing high above)
            drawn_liq = None
            for j in range(i - 1, max(i - 50, 0), -1):
                sh = swing_highs.iloc[j]
                if not np.isnan(sh) and sh > close[i]:
                    drawn_liq = sh
                    break
            if drawn_liq is None:
                recent_high = np.max(high[max(0, i-20):i])
                if recent_high > close[i] * 1.001:
                    drawn_liq = recent_high
                else:
                    continue

            # Check FVG reaction
            for fvg_bar, fvg_top, fvg_bot in active_bull_fvgs:
                if fvg_bar >= i - 3:
                    continue
                if low[i] <= fvg_top and low[i] >= fvg_bot and close[i] > fvg_top:
                    # RVOL check on signal candle
                    if rvol[i] < cfg.rvol_multiplier:
                        continue

                    entry = close[i]
                    sl = entry - atr[i] * cfg.atr_multiplier_sl
                    risk = entry - sl
                    tp = entry + risk * cfg.rr_ratio
                    if drawn_liq < tp:
                        tp = drawn_liq
                    reward = tp - entry
                    if risk <= 0 or reward <= 0 or reward / risk < 1.0:
                        continue

                    signals.append({
                        "bar": i, "direction": "buy", "entry": entry,
                        "sl": sl, "tp": tp, "strategy": "continuation",
                        "reason": f"Bull cont DL={drawn_liq:.2f} RR={reward/risk:.1f} ADX={adx[i]:.0f}",
                    })
                    active_bull_fvgs = [(b, t, bt) for b, t, bt in active_bull_fvgs if b != fvg_bar]
                    break

        # ── BEARISH CONTINUATION ──
        elif trend[i] == -1:
            if cfg.use_htf_filter and htf[i] > 0:
                continue

            drawn_liq = None
            for j in range(i - 1, max(i - 50, 0), -1):
                sl_val = swing_lows.iloc[j]
                if not np.isnan(sl_val) and sl_val < close[i]:
                    drawn_liq = sl_val
                    break
            if drawn_liq is None:
                recent_low = np.min(low[max(0, i-20):i])
                if recent_low < close[i] * 0.999:
                    drawn_liq = recent_low
                else:
                    continue

            for fvg_bar, fvg_top, fvg_bot in active_bear_fvgs:
                if fvg_bar >= i - 3:
                    continue
                if high[i] >= fvg_bot and high[i] <= fvg_top and close[i] < fvg_bot:
                    if rvol[i] < cfg.rvol_multiplier:
                        continue

                    entry = close[i]
                    sl = entry + atr[i] * cfg.atr_multiplier_sl
                    risk = sl - entry
                    tp = entry - risk * cfg.rr_ratio
                    if drawn_liq > tp:
                        tp = drawn_liq
                    reward = entry - tp
                    if risk <= 0 or reward <= 0 or reward / risk < 1.0:
                        continue

                    signals.append({
                        "bar": i, "direction": "sell", "entry": entry,
                        "sl": sl, "tp": tp, "strategy": "continuation",
                        "reason": f"Bear cont DL={drawn_liq:.2f} RR={reward/risk:.1f} ADX={adx[i]:.0f}",
                    })
                    active_bear_fvgs = [(b, t, bt) for b, t, bt in active_bear_fvgs if b != fvg_bar]
                    break

    # Print diagnostics
    total_bars = len(df) - 30
    print(f"    Filter diagnostics ({total_bars} bars scanned):")
    print(f"      Skipped — ATR invalid: {_skip['atr']} | Session: {_skip['session']} | ADX<{cfg.adx_threshold}: {_skip['adx']}")
    print(f"      Signals generated: {len(signals)}")

    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION ENGINE — With Friction
# ═══════════════════════════════════════════════════════════════════════════════

def execute_backtest(
    df: pd.DataFrame, signals: List[Dict], symbol: str, cfg: Config
) -> Dict[str, Any]:
    """Run backtest with full friction modeling. Returns gross and net P&L."""
    if not signals:
        return _empty_result(symbol, cfg.initial_capital)

    trades = []
    capital = cfg.initial_capital
    equity_curve = [capital]
    gross_pnl_total = 0.0
    net_pnl_total = 0.0
    total_commission = 0.0
    total_slippage = 0.0

    for sig in signals:
        entry = sig["entry"]
        sl = sig["sl"]
        tp = sig["tp"]
        direction = sig["direction"]
        bar_idx = sig["bar"]
        risk_per_trade = capital * cfg.risk_pct

        sl_dist = abs(entry - sl)
        if sl_dist <= 0:
            continue

        lot_size = risk_per_trade / sl_dist

        # Simulate trade forward
        outcome = _simulate_trade(df, bar_idx, direction, entry, sl, tp)
        if outcome is None:
            continue

        raw_exit = outcome["exit_price"]

        # Apply friction
        adj_entry, adj_exit, commission = apply_friction(entry, raw_exit, direction, symbol, cfg)

        if direction == "buy":
            gross_pnl = (raw_exit - entry) * lot_size
            net_pnl = (adj_exit - adj_entry) * lot_size - commission
        else:
            gross_pnl = (entry - raw_exit) * lot_size
            net_pnl = (adj_entry - adj_exit) * lot_size - commission

        slip_cost = abs(gross_pnl - (net_pnl + commission))
        total_slippage += slip_cost
        total_commission += commission
        gross_pnl_total += gross_pnl
        net_pnl_total += net_pnl

        capital += net_pnl
        equity_curve.append(capital)

        trades.append({
            "direction": direction,
            "entry": round(entry, 4), "exit": round(raw_exit, 4),
            "adj_entry": round(adj_entry, 4), "adj_exit": round(adj_exit, 4),
            "gross_pnl": round(gross_pnl, 2), "net_pnl": round(net_pnl, 2),
            "commission": round(commission, 2),
            "win": net_pnl > 0, "outcome": outcome["type"],
            "bar": bar_idx,
        })

    return _compute_metrics(trades, equity_curve, symbol, cfg.initial_capital,
                           gross_pnl_total, net_pnl_total, total_commission, total_slippage)


def _simulate_trade(df, start_bar, direction, entry, sl, tp, max_bars=50):
    """Simulate trade forward from entry bar."""
    for i in range(start_bar + 1, min(start_bar + max_bars, len(df))):
        h = df["High"].iloc[i]
        l = df["Low"].iloc[i]
        if direction == "buy":
            if l <= sl:
                return {"exit_price": sl, "type": "sl_hit", "bars": i - start_bar}
            if h >= tp:
                return {"exit_price": tp, "type": "tp_hit", "bars": i - start_bar}
        else:
            if h >= sl:
                return {"exit_price": sl, "type": "sl_hit", "bars": i - start_bar}
            if l <= tp:
                return {"exit_price": tp, "type": "tp_hit", "bars": i - start_bar}
    exit_price = df["Close"].iloc[min(start_bar + max_bars - 1, len(df) - 1)]
    return {"exit_price": exit_price, "type": "timeout", "bars": max_bars}


def _compute_metrics(trades, equity_curve, symbol, initial_capital,
                     gross_pnl, net_pnl, total_commission, total_slippage):
    if not trades:
        return _empty_result(symbol, initial_capital)

    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    net_pnls = [t["net_pnl"] for t in trades]
    wr = len(wins) / len(trades) * 100
    avg_win = np.mean([t["net_pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t["net_pnl"]) for t in losses]) if losses else 0
    pf_denom = sum(abs(t["net_pnl"]) for t in losses)
    pf = sum(t["net_pnl"] for t in wins) / pf_denom if pf_denom > 0 else float("inf")

    peak = initial_capital
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    sharpe = 0
    if len(net_pnls) > 10 and np.std(net_pnls) > 0:
        sharpe = np.mean(net_pnls) / np.std(net_pnls) * np.sqrt(252)
        sharpe = max(min(sharpe, 100), -100)

    return {
        "symbol": symbol,
        "total_trades": len(trades),
        "wins": len(wins), "losses": len(losses),
        "win_rate": round(wr, 1),
        "gross_pnl": round(gross_pnl, 2),
        "net_pnl": round(net_pnl, 2),
        "total_commission": round(total_commission, 2),
        "total_slippage": round(total_slippage, 2),
        "total_return_pct": round(net_pnl / initial_capital * 100, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(pf, 2) if pf != float("inf") else "inf",
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "equity_curve": equity_curve,
        "trades": trades,
    }


def _empty_result(symbol, capital):
    return {
        "symbol": symbol, "total_trades": 0, "wins": 0, "losses": 0,
        "win_rate": 0, "gross_pnl": 0, "net_pnl": 0,
        "total_commission": 0, "total_slippage": 0, "total_return_pct": 0,
        "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
        "max_drawdown_pct": 0, "sharpe_ratio": 0,
        "equity_curve": [capital], "trades": [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. WALK-FORWARD VALIDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward_validate(df: pd.DataFrame, symbol: str, cfg: Config) -> Dict[str, Any]:
    """
    Walk-forward: split 60 days into 4 windows (15 days each).
    Train on first 10 days, test on last 5 days per window.
    """
    # Ensure DatetimeIndex for date splitting
    df_work = df.copy()
    if not isinstance(df_work.index, pd.DatetimeIndex):
        df_work.index = pd.to_datetime(df_work.index, utc=True)

    # Get unique trading dates
    if df_work.index.tz is not None:
        dates = df_work.index.tz_convert("US/Eastern").date
    else:
        dates = df_work.index.date
    unique_dates = sorted(set(dates))

    total_days = len(unique_dates)
    days_per_window = total_days // cfg.wf_windows

    print(f"\n  Walk-Forward: {total_days} trading days, {cfg.wf_windows} windows × {days_per_window} days")

    # Parameter grid
    combos = list(itertools.product(cfg.opt_atr_sl, cfg.opt_rr, cfg.opt_disp))
    print(f"  Testing {len(combos)} parameter combos per window\n")

    windows = []
    oos_equity = [cfg.initial_capital]  # Continuous OOS equity curve
    running_capital = cfg.initial_capital

    for w in range(cfg.wf_windows):
        w_start = w * days_per_window
        w_end = w_start + days_per_window if w < cfg.wf_windows - 1 else total_days

        train_end = w_start + cfg.wf_train_days
        test_start = train_end
        test_end = w_end

        if train_end >= total_days or test_start >= total_days:
            continue

        train_dates = set(unique_dates[w_start:train_end])
        test_dates = set(unique_dates[test_start:test_end])

        # Slice data by dates
        train_mask = pd.Series(dates, index=df_work.index).isin(train_dates)
        test_mask = pd.Series(dates, index=df_work.index).isin(test_dates)

        df_train = df_work[train_mask].copy()
        df_test = df_work[test_mask].copy()

        if len(df_train) < 50 or len(df_test) < 20:
            continue

        # ── OPTIMIZE on train ──
        best_score = -999999
        best_params = {}
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
                futures_slippage_pts=cfg.futures_slippage_pts,
                initial_capital=cfg.initial_capital, risk_pct=cfg.risk_pct,
            )
            sigs = generate_signals(df_train, test_cfg)
            if len(sigs) < 3:
                continue
            m = execute_backtest(df_train, sigs, symbol, test_cfg)
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

        # ── TEST on out-of-sample ──
        oos_cfg = Config(
            atr_multiplier_sl=best_params.get("atr_sl", 0.5),
            rr_ratio=best_params.get("rr", 2.5),
            displacement_threshold=best_params.get("disp", 1.0),
            session_filter=cfg.session_filter, use_htf_filter=cfg.use_htf_filter,
            use_adx_filter=cfg.use_adx_filter, use_rvol_filter=cfg.use_rvol_filter,
            adx_threshold=cfg.adx_threshold, rvol_multiplier=cfg.rvol_multiplier,
            ema_period=cfg.ema_period, adx_period=cfg.adx_period,
            rvol_period=cfg.rvol_period, etf_slippage_pct=cfg.etf_slippage_pct,
            commission_round_trip=cfg.commission_round_trip,
            futures_slippage_pts=cfg.futures_slippage_pts,
            initial_capital=running_capital, risk_pct=cfg.risk_pct,
        )
        oos_sigs = generate_signals(df_test, oos_cfg)
        oos_metrics = execute_backtest(df_test, oos_sigs, symbol, oos_cfg)

        oos_pnl = oos_metrics.get("net_pnl", 0)
        oos_trades = oos_metrics.get("total_trades", 0)
        oos_wr = oos_metrics.get("win_rate", 0)
        running_capital += oos_pnl

        # Append OOS equity points
        for eq in oos_metrics.get("equity_curve", [])[1:]:
            oos_equity.append(eq)

        win_str = "✅" if oos_pnl > 0 else "❌"
        print(f"  Window {w+1}/{cfg.wf_windows}: "
              f"train [{unique_dates[w_start]}→{unique_dates[min(train_end-1, total_days-1)]}] "
              f"test [{unique_dates[test_start]}→{unique_dates[min(test_end-1, total_days-1)]}]")
        print(f"    Best params: ATR={best_params.get('atr_sl','?')} RR={best_params.get('rr','?')} "
              f"Disp={best_params.get('disp','?')}")
        print(f"    Train P&L: ${best_train_pnl:+.2f} | {win_str} OOS P&L: ${oos_pnl:+.2f} | "
              f"WR: {oos_wr:.1f}% | Trades: {oos_trades}")

        windows.append({
            "window": w + 1, "best_params": best_params,
            "train_pnl": round(best_train_pnl, 2),
            "oos_pnl": round(oos_pnl, 2), "oos_trades": oos_trades,
            "oos_wr": round(oos_wr, 1), "profitable": oos_pnl > 0,
        })

    # ── ROBUSTNESS MATRIX ──
    profitable_windows = sum(1 for w in windows if w["profitable"])
    robust = profitable_windows >= 3

    print(f"\n  {'='*55}")
    print(f"  ROBUSTNESS MATRIX — {symbol}")
    print(f"  {'='*55}")
    print(f"  {'Window':>8} {'Params':>20} {'Train P&L':>12} {'OOS P&L':>12} {'Status':>8}")
    print(f"  {'─'*62}")
    for w in windows:
        p = w["best_params"]
        param_str = f"ATR={p.get('atr_sl','-')} RR={p.get('rr','-')}"
        status = "✅ PASS" if w["profitable"] else "❌ FAIL"
        print(f"  {w['window']:>8} {param_str:>20} {w['train_pnl']:>+12.2f} {w['oos_pnl']:>+12.2f} {status:>8}")

    print(f"\n  Profitable OOS windows: {profitable_windows}/{len(windows)}")
    print(f"  {'✅ ROBUST — LIVE-READY' if robust else '⚠️  FRAGILE — NOT LIVE-READY'}")

    # Find most common winning params → "Live-Ready" set
    winning_params = [w["best_params"] for w in windows if w["profitable"]]
    live_params = {}
    if winning_params:
        for key in winning_params[0]:
            vals = [p[key] for p in winning_params]
            live_params[key] = round(float(np.median(vals)), 2)
        print(f"\n  📊 LIVE-READY PARAMS: {live_params}")

    total_oos_pnl = sum(w["oos_pnl"] for w in windows)
    print(f"  📊 TOTAL OOS NET P&L: ${total_oos_pnl:+.2f}")

    # ── EQUITY CURVE PNG ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(oos_equity, color="#00d4ff", linewidth=1.5)
        ax.axhline(y=cfg.initial_capital, color="gray", linestyle="--", alpha=0.5)
        ax.fill_between(range(len(oos_equity)), cfg.initial_capital, oos_equity,
                        where=[e >= cfg.initial_capital for e in oos_equity],
                        alpha=0.15, color="#00ff88")
        ax.fill_between(range(len(oos_equity)), cfg.initial_capital, oos_equity,
                        where=[e < cfg.initial_capital for e in oos_equity],
                        alpha=0.15, color="#ff4444")
        ax.set_title(f"Walk-Forward OOS Equity — {symbol} Continuation", fontsize=14, color="white")
        ax.set_xlabel("Trade #", color="gray")
        ax.set_ylabel("Equity ($)", color="gray")
        ax.set_facecolor("#0a0e1a")
        fig.set_facecolor("#0a0e1a")
        ax.tick_params(colors="gray")
        ax.spines["bottom"].set_color("gray")
        ax.spines["left"].set_color("gray")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        png_path = os.path.join(RESULTS_DIR, f"walk_forward_equity_{symbol}.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  📈 Equity curve saved: {png_path}")
    except ImportError:
        print("  ⚠️  matplotlib not installed — skipping equity curve PNG")

    return {
        "symbol": symbol, "windows": windows,
        "profitable_windows": profitable_windows,
        "is_robust": robust, "live_params": live_params,
        "total_oos_pnl": round(total_oos_pnl, 2),
        "oos_equity": oos_equity,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_backtest_report(m: Dict[str, Any], label: str) -> None:
    pf = m.get("profit_factor", 0)
    pf_str = f"{pf:.2f}" if isinstance(pf, (int, float)) else str(pf)
    print(f"\n  {label}:")
    print(f"    Trades: {m.get('total_trades', 0)} | WR: {m.get('win_rate', 0):.1f}%")
    print(f"    Gross P&L: ${m.get('gross_pnl', 0):+.2f}")
    print(f"    Net P&L:   ${m.get('net_pnl', 0):+.2f}")
    print(f"    Commission: ${m.get('total_commission', 0):.2f} | Slippage: ${m.get('total_slippage', 0):.2f}")
    print(f"    PF: {pf_str} | Sharpe: {m.get('sharpe_ratio', 0):.2f} | Max DD: {m.get('max_drawdown_pct', 0):.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Institutional Continuation Engine")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--optimize", action="store_true", help="Run walk-forward optimization")
    parser.add_argument("--no-filter", action="store_true", help="Disable NY session filter")
    parser.add_argument("--no-htf", action="store_true", help="Disable 1H EMA filter")
    parser.add_argument("--no-adx", action="store_true", help="Disable ADX filter")
    parser.add_argument("--no-rvol", action="store_true", help="Disable RVOL filter")
    parser.add_argument("--atr-sl", type=float, default=None)
    parser.add_argument("--rr", type=float, default=None)
    parser.add_argument("--tf", type=str, default="15m", help="Timeframe: 5m, 15m, 1h (default: 15m)")
    args = parser.parse_args()

    cfg = Config()
    if args.no_filter:
        cfg.session_filter = False
    if args.no_htf:
        cfg.use_htf_filter = False
    if args.no_adx:
        cfg.use_adx_filter = False
    if args.no_rvol:
        cfg.use_rvol_filter = False
    if args.atr_sl:
        cfg.atr_multiplier_sl = args.atr_sl
    if args.rr:
        cfg.rr_ratio = args.rr

    symbols = [args.symbol] if args.symbol else cfg.symbols

    print("\n" + "=" * 60)
    print("  INSTITUTIONAL CONTINUATION ENGINE v2.0")
    print("  Friction · Confluence · Anti-Chop · Walk-Forward")
    print("=" * 60)
    print(f"  Session Filter: {'ON' if cfg.session_filter else 'OFF'}")
    print(f"  1H EMA Filter:  {'ON' if cfg.use_htf_filter else 'OFF'}")
    adx_str = f"ON (>{cfg.adx_threshold})" if cfg.use_adx_filter else "OFF"
    rvol_str = f"ON (>{cfg.rvol_multiplier}x)" if cfg.use_rvol_filter else "OFF"
    print(f"  ADX Filter:     {adx_str}")
    print(f"  RVOL Filter:    {rvol_str}")
    slip_display = cfg.etf_slippage_pct * 100
    print(f"  Friction:       ETF slip {slip_display:.3f}% | Comm ${cfg.commission_round_trip}/RT")

    tf = args.tf
    print(f"  Timeframe:      {tf}")

    if args.optimize:
        # Walk-forward validation
        all_wf = {}
        for sym in symbols:
            print(f"\n{'═'*60}")
            print(f"  {sym} — Walk-Forward Validation ({tf})")
            print(f"{'═'*60}")

            df = download_data(sym, period="60d", interval=tf)
            if df.empty:
                continue

            result = walk_forward_validate(df, sym, cfg)
            all_wf[sym] = result

        # Save
        out = {k: {kk: vv for kk, vv in v.items() if kk != "oos_equity"}
               for k, v in all_wf.items()}
        with open(os.path.join(RESULTS_DIR, "wf_institutional_results.json"), "w") as f:
            json.dump(out, f, indent=2, default=str)

    else:
        # Standard backtest
        for sym in symbols:
            print(f"\n{'═'*60}")
            print(f"  {sym} — Backtest ({tf})")

            df = download_data(sym, period="60d", interval=tf)
            if df.empty:
                continue

            print(f"  {len(df)} bars loaded")
            signals = generate_signals(df, cfg)
            metrics = execute_backtest(df, signals, sym, cfg)
            print_backtest_report(metrics, f"{sym} (all filters ON)")

            # Also run without filters for comparison
            cfg_raw = Config(
                atr_multiplier_sl=cfg.atr_multiplier_sl, rr_ratio=cfg.rr_ratio,
                displacement_threshold=cfg.displacement_threshold,
                session_filter=False, use_htf_filter=False,
                use_adx_filter=False, use_rvol_filter=False,
            )
            signals_raw = generate_signals(df, cfg_raw)
            metrics_raw = execute_backtest(df, signals_raw, sym, cfg_raw)
            print_backtest_report(metrics_raw, f"{sym} (NO filters — raw)")

    print("\n" + "=" * 60)
    print("  DONE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
