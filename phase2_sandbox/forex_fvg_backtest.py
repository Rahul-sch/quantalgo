#!/usr/bin/env python3
"""
Forex FVG Continuation Backtest
================================
Cross-asset test of the ICT FVG Continuation model (V2) on Forex majors.

Model: 15m/1h FVG detection → retest entry → Break-Even at IRL pivot exit.

Forex Killzones:
  London Open:  03:00–05:00 ET
  NY Open:      08:00–11:00 ET
  (Asian + late NY are blocked)

Pairs tested: EUR/USD, GBP/USD
Risk:  $100 per trade (1% of $10K)
SL:    0.5x ATR
RR:    2.0:1
RVOL:  Optional — tick volume filter (1.2x 20-bar SMA)

Usage:
  python3 forex_fvg_backtest.py              # 15m (60 days)
  python3 forex_fvg_backtest.py --timeframe 1h   # 1h (2 years)
  python3 forex_fvg_backtest.py --rvol       # enable RVOL filter
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

PAIRS = ["EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X"]
PAIR_LABELS = {"EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD",
               "JPY=X": "USD/JPY", "AUDUSD=X": "AUD/USD"}

# Risk management
RISK_PER_TRADE = 100.0   # $100 per trade
ATR_SL_MULT    = 0.5     # SL = 0.5x ATR
RR_RATIO       = 2.0     # TP = 2.0x risk
POSITION_SIZE  = 10000   # notional for P&L calc (will be derived from risk)

# FVG parameters
RETEST_MAX_BARS = 20     # max bars to wait for FVG retest
MAX_HOLD_BARS   = 40     # max bars in a trade before forced exit

# IRL pivot detection
IRL_PIVOT_BARS  = 5      # local pivot lookback/lookahead

# Forex killzones (minutes since midnight ET)
LONDON_START =  180  # 03:00 ET
LONDON_END   =  300  # 05:00 ET
NY_START     =  480  # 08:00 ET
NY_END       =  660  # 11:00 ET

# RVOL parameters
RVOL_PERIOD = 20
RVOL_MIN    = 1.2

# Transaction costs (forex) — REALISTIC BROKER CONDITIONS
# Spread: 1.5 pips applied at entry (worsens fill price)
# Commission: $3.00 per standard lot (100K units) per round-trip
SPREAD_PIPS     = 1.5     # spread penalty injected at entry
PIP_VALUE_EUR   = 0.0001  # 1 pip for EUR/USD, GBP/USD, etc.
PIP_VALUE_JPY   = 0.01    # 1 pip for JPY pairs
COMMISSION_PER_LOT = 3.0  # $3.00 per 100K lot round-trip
LOT_SIZE        = 100_000 # standard lot

# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def fetch_forex_data(pair: str, timeframe: str = "15m") -> pd.DataFrame:
    """Fetch forex data from yfinance."""
    import yfinance as yf

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{pair.replace('=X','')}_{timeframe}_cache.csv")

    # Check cache freshness (use cache if < 1 hour old)
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        import time
        if time.time() - mtime < 3600:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if len(df) > 100:
                print(f"  [cached] {PAIR_LABELS.get(pair, pair)} {timeframe}: {len(df)} bars")
                return df

    if timeframe == "15m":
        period = "60d"
    else:
        period = "2y"

    print(f"  [downloading] {PAIR_LABELS.get(pair, pair)} {timeframe} {period}...")
    df = yf.download(pair, period=period, interval=timeframe, progress=False)

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Save cache
    df.to_csv(cache_file)
    print(f"  [ok] {PAIR_LABELS.get(pair, pair)}: {len(df)} bars")
    return df


# ════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ════════════════════════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rvol(df: pd.DataFrame, period: int = RVOL_PERIOD) -> pd.Series:
    """Relative Volume = volume / SMA(volume, period)."""
    if "Volume" not in df.columns or df["Volume"].sum() == 0:
        return pd.Series(np.nan, index=df.index)
    vol_sma = df["Volume"].rolling(period).mean()
    return df["Volume"] / vol_sma


# ════════════════════════════════════════════════════════════════════════════
# FVG DETECTION
# ════════════════════════════════════════════════════════════════════════════

def detect_fvgs(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Detect Fair Value Gaps (3-candle imbalance).
    
    Bullish FVG: candle[i-2].High < candle[i].Low (gap up)
    Bearish FVG: candle[i-2].Low > candle[i].High (gap down)
    
    Returns:
        bull_top, bull_bot, bear_top, bear_bot — NaN where no FVG exists
    """
    high = df["High"].values
    low = df["Low"].values

    n = len(df)
    bull_top = np.full(n, np.nan)
    bull_bot = np.full(n, np.nan)
    bear_top = np.full(n, np.nan)
    bear_bot = np.full(n, np.nan)

    for i in range(2, n):
        # Bullish FVG: candle[i-2] high < candle[i] low
        if high[i-2] < low[i]:
            bull_top[i-1] = low[i]       # top of gap = candle 3 low
            bull_bot[i-1] = high[i-2]    # bottom of gap = candle 1 high

        # Bearish FVG: candle[i-2] low > candle[i] high
        if low[i-2] > high[i]:
            bear_top[i-1] = low[i-2]     # top of gap = candle 1 low
            bear_bot[i-1] = high[i]      # bottom of gap = candle 3 high

    return (
        pd.Series(bull_top, index=df.index),
        pd.Series(bull_bot, index=df.index),
        pd.Series(bear_top, index=df.index),
        pd.Series(bear_bot, index=df.index),
    )


# ════════════════════════════════════════════════════════════════════════════
# IRL PIVOT DETECTION
# ════════════════════════════════════════════════════════════════════════════

def find_local_pivots(high: np.ndarray, low: np.ndarray,
                      lookback: int = IRL_PIVOT_BARS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find local swing highs and lows for IRL targeting.
    pivot_highs[i] = high[i] if it's a local max within ±lookback bars
    pivot_lows[i]  = low[i] if it's a local min within ±lookback bars
    """
    n = len(high)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        window_h = high[max(0, i-lookback):i+lookback+1]
        if high[i] == window_h.max():
            pivot_highs[i] = high[i]

        window_l = low[max(0, i-lookback):i+lookback+1]
        if low[i] == window_l.min():
            pivot_lows[i] = low[i]

    return pivot_highs, pivot_lows


def find_irl_target(direction: str, entry: float, tp: float,
                    bar_idx: int, pivot_highs: np.ndarray,
                    pivot_lows: np.ndarray) -> Optional[float]:
    """
    Find the nearest IRL (Internal Range Liquidity) target between entry and TP.
    For longs: nearest pivot high above entry but below TP
    For shorts: nearest pivot low below entry but above TP
    """
    look_start = max(0, bar_idx - 100)
    look_end = bar_idx

    if direction == "long":
        candidates = []
        for j in range(look_start, look_end):
            ph = pivot_highs[j]
            if not np.isnan(ph) and entry < ph < tp:
                candidates.append(ph)
        return min(candidates) if candidates else None
    else:
        candidates = []
        for j in range(look_start, look_end):
            pl = pivot_lows[j]
            if not np.isnan(pl) and tp < pl < entry:
                candidates.append(pl)
        return max(candidates) if candidates else None


# ════════════════════════════════════════════════════════════════════════════
# SESSION / KILLZONE FILTERS
# ════════════════════════════════════════════════════════════════════════════

def _to_et_minutes(idx: pd.DatetimeIndex) -> np.ndarray:
    """Convert DatetimeIndex to minutes-since-midnight in US/Eastern."""
    try:
        et = idx.tz_convert("US/Eastern")
    except TypeError:
        et = idx.tz_localize("UTC").tz_convert("US/Eastern")
    return np.array(et.hour * 60 + et.minute)


def is_in_forex_killzone(idx: pd.DatetimeIndex) -> pd.Series:
    """
    Forex Killzone filter:
      London Open: 03:00–05:00 ET
      NY Open:     08:00–11:00 ET
    """
    mins = _to_et_minutes(idx)
    mask = (((mins >= LONDON_START) & (mins < LONDON_END)) |
            ((mins >= NY_START) & (mins < NY_END)))
    return pd.Series(mask, index=idx)


def is_in_any_session(idx: pd.DatetimeIndex) -> pd.Series:
    """Broad session filter (no killzone, just London + NY combined 03:00-17:00 ET)."""
    mins = _to_et_minutes(idx)
    mask = (mins >= 180) & (mins < 1020)  # 3am to 5pm ET
    return pd.Series(mask, index=idx)


# ════════════════════════════════════════════════════════════════════════════
# TRADE SIMULATION — BREAK-EVEN AT IRL
# ════════════════════════════════════════════════════════════════════════════

def simulate_breakeven(signals: List[Dict], df: pd.DataFrame,
                       pivot_highs: np.ndarray, pivot_lows: np.ndarray,
                       risk_per_trade: float = RISK_PER_TRADE) -> List[Dict]:
    """
    Simulate trades with Break-Even at IRL:
    1. Enter at FVG retest
    2. If price hits IRL target, move SL to breakeven
    3. Then either TP or scratch (exit at BE)
    """
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    results = []

    for sig in signals:
        entry = sig["entry"]
        sl = sig["sl"]
        tp = sig["tp"]
        direction = sig["direction"]
        bar_idx = sig["bar_idx"]
        risk = sig["risk"]

        if risk <= 0:
            continue

        # Calculate position size based on $risk
        # For forex: units = risk / risk_in_price
        units = risk_per_trade / risk

        # Find IRL target
        irl = find_irl_target(direction, entry, tp, bar_idx, pivot_highs, pivot_lows)

        be_triggered = False
        exit_price = None
        exit_bar = None
        outcome = None

        for j in range(bar_idx + 1, min(bar_idx + MAX_HOLD_BARS + 1, len(df))):
            h = high[j]
            l = low[j]

            if direction == "long":
                # Check SL first
                current_sl = entry if be_triggered else sl
                if l <= current_sl:
                    exit_price = current_sl
                    exit_bar = j
                    outcome = "scratch" if be_triggered else "loss"
                    break

                # Check IRL for BE trigger
                if irl is not None and not be_triggered and h >= irl:
                    be_triggered = True

                # Check TP
                if h >= tp:
                    exit_price = tp
                    exit_bar = j
                    outcome = "win"
                    break
            else:
                # Short
                current_sl = entry if be_triggered else sl
                if h >= current_sl:
                    exit_price = current_sl
                    exit_bar = j
                    outcome = "scratch" if be_triggered else "loss"
                    break

                # Check IRL for BE trigger
                if irl is not None and not be_triggered and l <= irl:
                    be_triggered = True

                # Check TP
                if l <= tp:
                    exit_price = tp
                    exit_bar = j
                    outcome = "win"
                    break

        # Forced exit at max hold
        if exit_price is None:
            exit_bar = min(bar_idx + MAX_HOLD_BARS, len(df) - 1)
            exit_price = close[exit_bar]
            if direction == "long":
                outcome = "win" if exit_price > entry else "loss"
            else:
                outcome = "win" if exit_price < entry else "loss"

        # Calculate P&L
        if direction == "long":
            gross_pnl = (exit_price - entry) * units
        else:
            gross_pnl = (entry - exit_price) * units

        # Commission: $3.00 per standard lot (100K units) round-trip
        lots = units / LOT_SIZE
        commission = lots * COMMISSION_PER_LOT
        net_pnl = gross_pnl - commission

        # Adjust outcome based on net P&L for scratches
        if outcome == "scratch":
            net_pnl = -commission  # scratch = lose the commission only

        results.append({
            "bar_idx": bar_idx,
            "time": sig["time"],
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "exit_price": exit_price,
            "exit_bar": exit_bar,
            "outcome": outcome,
            "be_triggered": be_triggered,
            "irl_target": irl,
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "risk": risk,
            "fvg_rvol": sig.get("fvg_rvol", np.nan),
        })

    return results


# ════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ════════════════════════════════════════════════════════════════════════════

def generate_signals(df: pd.DataFrame, use_killzone: bool = True,
                     use_rvol: bool = False) -> List[Dict]:
    """
    Generate FVG continuation signals for forex.
    
    Parameters
    ----------
    use_killzone : If True, only arm FVGs where the displacement candle is in
                   London/NY killzone. If False, broad session filter.
    use_rvol     : If True, require RVOL >= 1.2x on FVG-creating bar.
    """
    atr = compute_atr(df)
    bull_top, bull_bot, bear_top, bear_bot = detect_fvgs(df)
    rvol = compute_rvol(df)

    if use_killzone:
        sess = is_in_forex_killzone(df.index)
    else:
        sess = is_in_any_session(df.index)

    high = df["High"].values
    low = df["Low"].values
    rvol_arr = rvol.values if rvol is not None else None

    signals: List[Dict] = []
    armed: Dict[int, tuple] = {}

    for i in range(30, len(df)):
        atr_val = atr.iloc[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        # Displacement check: the FVG candle body must be >= 1.0x ATR
        body = abs(df["Close"].iloc[i] - df["Open"].iloc[i])
        if body < atr_val * 1.0:
            continue

        # RVOL of FVG-creating bar
        fvg_rvol = np.nan
        if rvol_arr is not None and i < len(rvol_arr):
            rv = float(rvol_arr[i])
            if not np.isnan(rv):
                fvg_rvol = rv

        # RVOL gate
        rvol_ok = True
        if use_rvol:
            rvol_ok = (not np.isnan(fvg_rvol)) and (fvg_rvol >= RVOL_MIN)

        # Session gate on FVG creation (displacement must be in killzone)
        if not sess.iloc[i]:
            continue

        # Arm new FVGs
        if not np.isnan(bull_top.iloc[i]) and rvol_ok:
            armed[i] = ("long", float(bull_top.iloc[i]), float(bull_bot.iloc[i]),
                        float(atr_val), i, fvg_rvol)
        if not np.isnan(bear_top.iloc[i]) and rvol_ok:
            armed[i] = ("short", float(bear_bot.iloc[i]), float(bear_top.iloc[i]),
                        float(atr_val), i, fvg_rvol)

        # Scan armed FVGs for retest
        to_remove = []
        for fvg_bar, (direction, limit_price, other_edge,
                      atr_arm, armed_bar, sig_rvol) in armed.items():
            bars_elapsed = i - armed_bar
            if bars_elapsed > RETEST_MAX_BARS:
                to_remove.append(fvg_bar)
                continue
            if bars_elapsed == 0:
                continue

            # Invalidation check
            if direction == "long" and low[i] < other_edge:
                to_remove.append(fvg_bar)
                continue
            if direction == "short" and high[i] > other_edge:
                to_remove.append(fvg_bar)
                continue

            # Retest check
            if not (low[i] <= limit_price <= high[i]):
                continue

            # Build trade — inject 1.5 pip spread penalty at entry
            # For longs: fill is WORSE (higher), for shorts: fill is WORSE (lower)
            pip_val = PIP_VALUE_JPY if "JPY" in str(df.index.name) else PIP_VALUE_EUR
            spread_penalty = SPREAD_PIPS * pip_val

            if direction == "long":
                entry = limit_price + spread_penalty  # worse fill for buy
                sl = entry - atr_arm * ATR_SL_MULT
                risk = entry - sl
                if risk <= 0:
                    continue
                tp = entry + risk * RR_RATIO
            else:
                entry = limit_price - spread_penalty  # worse fill for sell
                sl = entry + atr_arm * ATR_SL_MULT
                risk = sl - entry
                if risk <= 0:
                    continue
                tp = entry - risk * RR_RATIO

            signals.append({
                "bar_idx": i,
                "time": df.index[i],
                "direction": direction,
                "entry": round(entry, 6),
                "sl": round(sl, 6),
                "tp": round(tp, 6),
                "risk": round(risk, 6),
                "fvg_rvol": round(sig_rvol, 4) if not np.isnan(sig_rvol) else np.nan,
                "spread_applied": round(spread_penalty, 6),
            })
            to_remove.append(fvg_bar)

        for k in to_remove:
            armed.pop(k, None)

    return signals


# ════════════════════════════════════════════════════════════════════════════
# STATISTICS & REPORTING
# ════════════════════════════════════════════════════════════════════════════

def max_drawdown(trades: List[Dict]) -> Tuple[float, float]:
    """Calculate max drawdown from trade results."""
    if not trades:
        return 0.0, 0.0
    equity = 10000.0
    peak = equity
    max_dd = 0.0
    for t in trades:
        equity += t["net_pnl"]
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = max_dd / 10000.0 * 100
    return max_dd, max_dd_pct


def print_pair_report(trades: List[Dict], pair_label: str, model_label: str):
    """Print a detailed report for one pair + model combination."""
    if not trades:
        print(f"\n  {pair_label} — {model_label}: No trades")
        return

    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    scratches = [t for t in trades if t["outcome"] == "scratch"]
    non_scratch = [t for t in trades if t["outcome"] != "scratch"]

    wr = len(wins) / len(trades) * 100
    wr_excl = len(wins) / len(non_scratch) * 100 if non_scratch else 0
    pnl = sum(t["net_pnl"] for t in trades)
    aw = np.mean([t["net_pnl"] for t in wins]) if wins else 0.0
    al = np.mean([t["net_pnl"] for t in losses]) if losses else 0.0
    ws = sum(t["net_pnl"] for t in wins)
    ls = sum(t["net_pnl"] for t in losses)
    pf = abs(ws / ls) if ls != 0 else float("inf")
    md, mdp = max_drawdown(trades)

    longs = [t for t in trades if t["direction"] == "long"]
    shorts = [t for t in trades if t["direction"] == "short"]
    lwr = len([t for t in longs if t["outcome"] == "win"]) / len(longs) * 100 if longs else 0.0
    swr = len([t for t in shorts if t["outcome"] == "win"]) / len(shorts) * 100 if shorts else 0.0

    pfs = f"{pf:.2f}" if pf != float("inf") else "inf"

    print(f"\n  {pair_label} — {model_label}")
    print(f"  {'─' * 65}")
    print(f"  Trades:        {len(trades):>6}  |  Win Rate:      {wr:>6.1f}%")
    print(f"  W / L / S:     {len(wins)}W / {len(losses)}L / {len(scratches)}S"
          f"  |  WR (excl S):  {wr_excl:>6.1f}%")
    print(f"  Net P&L:       ${pnl:>10,.2f}  |  Long WR:       {lwr:>6.1f}% ({len(longs)})")
    print(f"  Avg Win:       ${aw:>10,.2f}  |  Short WR:      {swr:>6.1f}% ({len(shorts)})")
    print(f"  Avg Loss:      ${al:>10,.2f}  |  Profit Factor: {pfs:>6}")
    print(f"  Max DD:        ${md:>9,.2f} ({mdp:.1f}%)")

    # Monthly breakdown
    months: Dict[str, list] = {}
    for t in trades:
        m = t["time"].strftime("%Y-%m")
        months.setdefault(m, []).append(t)

    if len(months) > 1:
        print(f"\n  {'Month':<10} {'#':>4} {'W':>3} {'L':>3} {'S':>3} {'WR%':>6} {'P&L':>10} {'PF':>6}")
        print(f"  {'─' * 52}")
        for m in sorted(months.keys()):
            mt = months[m]
            mw = len([t for t in mt if t["outcome"] == "win"])
            ml = len([t for t in mt if t["outcome"] == "loss"])
            ms = len([t for t in mt if t["outcome"] == "scratch"])
            mwr = mw / len(mt) * 100
            mpnl = sum(t["net_pnl"] for t in mt)
            mws = sum(t["net_pnl"] for t in mt if t["outcome"] == "win")
            mls = sum(t["net_pnl"] for t in mt if t["outcome"] == "loss")
            mpf = abs(mws / mls) if mls != 0 else float("inf")
            mpfs = f"{mpf:.2f}" if mpf != float("inf") else "inf"
            mark = "✓" if mpnl > 0 else "✗"
            print(f"  {m:<10} {len(mt):>4} {mw:>3} {ml:>3} {ms:>3}"
                  f" {mwr:>5.1f}% ${mpnl:>8,.2f} {mpfs:>6}  {mark}")


def print_combined_summary(all_results: Dict[str, Dict[str, List[Dict]]]):
    """Print a combined summary table across all pairs and models."""
    print(f"\n  {'=' * 78}")
    print(f"  COMBINED SUMMARY")
    print(f"  {'=' * 78}")
    print(f"  {'Pair':<10} {'Model':<20} {'#':>5} {'WR%':>6} {'P&L':>10} {'PF':>6} {'MaxDD':>9}")
    print(f"  {'─' * 78}")

    total_trades = {"baseline": 0, "killzone": 0, "kz_rvol": 0}
    total_pnl = {"baseline": 0.0, "killzone": 0.0, "kz_rvol": 0.0}

    for pair, models in all_results.items():
        for model_key, trades in models.items():
            if not trades:
                continue
            n = len(trades)
            w = len([t for t in trades if t["outcome"] == "win"])
            wr = w / n * 100
            pnl = sum(t["net_pnl"] for t in trades)
            ws = sum(t["net_pnl"] for t in trades if t["outcome"] == "win")
            ls = sum(t["net_pnl"] for t in trades if t["outcome"] == "loss")
            pf = abs(ws / ls) if ls != 0 else float("inf")
            pfs = f"{pf:.2f}" if pf != float("inf") else "inf"
            md, mdp = max_drawdown(trades)

            label = pair
            model_label = {"baseline": "Broad Session",
                           "killzone": "Killzone",
                           "kz_rvol": "Killzone + RVOL"}[model_key]

            print(f"  {label:<10} {model_label:<20} {n:>5} {wr:>5.1f}%"
                  f" ${pnl:>8,.2f} {pfs:>6} ${md:>7,.2f}")

            total_trades[model_key] += n
            total_pnl[model_key] += pnl

    print(f"  {'─' * 78}")
    for model_key in ["baseline", "killzone", "kz_rvol"]:
        model_label = {"baseline": "Broad Session",
                       "killzone": "Killzone",
                       "kz_rvol": "Killzone + RVOL"}[model_key]
        print(f"  {'TOTAL':<10} {model_label:<20} {total_trades[model_key]:>5}"
              f" {'':>6} ${total_pnl[model_key]:>8,.2f}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Forex FVG Continuation Backtest")
    parser.add_argument("--timeframe", "-t", default="15m", choices=["15m", "1h"],
                        help="Timeframe: 15m (60 days) or 1h (2 years)")
    parser.add_argument("--rvol", action="store_true",
                        help="Enable RVOL filter (tick volume >= 1.2x 20-bar SMA)")
    args = parser.parse_args()

    tf = args.timeframe
    sep = "=" * 72

    print(f"\n  {sep}")
    print(f"  Forex FVG Continuation Backtest")
    print(f"  ICT Model V2 — Break-Even at IRL")
    print(f"  Timeframe: {tf} | Risk: ${RISK_PER_TRADE}/trade | RR: {RR_RATIO}:1")
    print(f"  ATR SL: {ATR_SL_MULT}x | Displacement: >= 1.0x ATR body")
    print(f"  Killzones: London 03:00–05:00 ET | NY 08:00–11:00 ET")
    if args.rvol:
        print(f"  RVOL Filter: ON (>= {RVOL_MIN}x {RVOL_PERIOD}-bar SMA)")
    else:
        print(f"  RVOL Filter: OFF (tick volume unreliable for forex)")
    print(f"  {sep}")

    all_results: Dict[str, Dict[str, List[Dict]]] = {}

    for pair in PAIRS:
        pair_label = PAIR_LABELS.get(pair, pair)
        print(f"\n  {'─' * 72}")
        print(f"  Loading {pair_label}...")

        df = fetch_forex_data(pair, tf)
        if len(df) < 200:
            print(f"  ERROR: Not enough data for {pair_label} ({len(df)} bars)")
            continue

        period_start = df.index[0].strftime("%Y-%m-%d")
        period_end = df.index[-1].strftime("%Y-%m-%d")
        print(f"  Period: {period_start} → {period_end} ({len(df)} bars)")

        # Compute pivots
        pivot_highs, pivot_lows = find_local_pivots(df["High"].values, df["Low"].values)
        n_ph = int(np.count_nonzero(~np.isnan(pivot_highs)))
        n_pl = int(np.count_nonzero(~np.isnan(pivot_lows)))
        print(f"  IRL pivots: {n_ph} highs, {n_pl} lows")

        pair_results: Dict[str, List[Dict]] = {}

        # Model 1: Broad session (baseline — no killzone restriction)
        print(f"\n  Generating signals — Broad Session (03:00–17:00 ET)...")
        sigs_base = generate_signals(df, use_killzone=False, use_rvol=False)
        print(f"  → {len(sigs_base)} signals")
        res_base = simulate_breakeven(sigs_base, df, pivot_highs, pivot_lows)
        pair_results["baseline"] = res_base

        # Model 2: Killzone only
        print(f"  Generating signals — Killzone (London + NY Open)...")
        sigs_kz = generate_signals(df, use_killzone=True, use_rvol=False)
        print(f"  → {len(sigs_kz)} signals")
        res_kz = simulate_breakeven(sigs_kz, df, pivot_highs, pivot_lows)
        pair_results["killzone"] = res_kz

        # Model 3: Killzone + RVOL (if enabled)
        if args.rvol:
            print(f"  Generating signals — Killzone + RVOL >= {RVOL_MIN}x...")
            sigs_rv = generate_signals(df, use_killzone=True, use_rvol=True)
            print(f"  → {len(sigs_rv)} signals")
            res_rv = simulate_breakeven(sigs_rv, df, pivot_highs, pivot_lows)
            pair_results["kz_rvol"] = res_rv

        all_results[pair_label] = pair_results

        # Print per-pair reports
        print_pair_report(res_base, pair_label, "Broad Session (Baseline)")
        print_pair_report(res_kz, pair_label, "Killzone (London + NY Open)")
        if args.rvol and "kz_rvol" in pair_results:
            print_pair_report(pair_results["kz_rvol"], pair_label,
                              "Killzone + RVOL")

    # Combined summary
    print_combined_summary(all_results)

    print(f"\n  {sep}")
    print(f"  Done.")
    print(f"  {sep}\n")


if __name__ == "__main__":
    main()
