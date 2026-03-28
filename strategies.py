"""
Five ICT/Goldbach Trading Strategies
"""
import pandas as pd
import numpy as np
from goldbach import calculate_goldbach_levels, price_in_zone, get_po3_levels, get_nearest_goldbach_level

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False


def _rolling_high_low(df: pd.DataFrame, window: int = 20):
    """Get rolling dealing range."""
    return df["High"].rolling(window).max(), df["Low"].rolling(window).min()


# ─── Signal format: list of dicts ─────────────────────────────────────
# Each signal: {"bar": int, "direction": "buy"|"sell", "entry": float,
#               "sl": float, "tp": float, "strategy": str, "reason": str}


def strategy_goldbach_bounce(df: pd.DataFrame, lookback: int = 20) -> list:
    """
    Strategy 1: Goldbach Bounce
    - Buy at PO3/PO9 level in discount zone
    - Sell at PO3/PO9 level in premium zone
    """
    signals = []
    highs, lows = _rolling_high_low(df, lookback)

    for i in range(lookback + 1, len(df)):
        h, l = highs.iloc[i], lows.iloc[i]
        if pd.isna(h) or pd.isna(l) or h <= l:
            continue

        close = df["Close"].iloc[i]
        gb = calculate_goldbach_levels(h, l)
        zone = price_in_zone(close, h, l)
        rng = h - l

        # Find nearest PO3 or PO9 level
        key_levels = [lv for lv in gb["levels"] if lv["power"] in (3, 9)]
        nearest = get_nearest_goldbach_level(close, key_levels)
        if nearest is None:
            continue

        dist = abs(close - nearest["price"])
        tolerance = rng * 0.01  # within 1% of range

        if dist > tolerance:
            continue

        if zone == "discount" and nearest["zone"] == "discount":
            sl = l - rng * 0.02
            tp = gb["equilibrium"]
            if tp > close and close > sl:
                signals.append({
                    "bar": i, "direction": "buy", "entry": close,
                    "sl": sl, "tp": tp, "strategy": "goldbach_bounce",
                    "reason": f"Buy at {nearest['label']} in discount"
                })
        elif zone == "premium" and nearest["zone"] == "premium":
            sl = h + rng * 0.02
            tp = gb["equilibrium"]
            if tp < close and close < sl:
                signals.append({
                    "bar": i, "direction": "sell", "entry": close,
                    "sl": sl, "tp": tp, "strategy": "goldbach_bounce",
                    "reason": f"Sell at {nearest['label']} in premium"
                })

    return signals


def strategy_amd_cycle(df: pd.DataFrame) -> list:
    """
    Strategy 2: AMD Cycle
    - Identify Asian range (accumulation) using first portion of trading day
    - London/mid-day manipulation breaks the range
    - Trade distribution in opposite direction

    Works with any timezone — uses relative bar positions within each day
    instead of absolute UTC hours (yfinance timezone varies by instrument).
    """
    signals = []

    df_work = df.copy()

    # Ensure we have a usable datetime index
    if not isinstance(df_work.index, pd.DatetimeIndex):
        try:
            df_work.index = pd.to_datetime(df_work.index)
        except Exception:
            return signals

    # Strip timezone so groupby works cleanly
    if df_work.index.tz is not None:
        df_work.index = df_work.index.tz_convert(None)

    df_work["_date"] = df_work.index.date

    for date, day_df in df_work.groupby("_date"):
        n_bars = len(day_df)
        if n_bars < 6:
            continue

        # Split the day into 3 roughly equal sessions by bar count
        # Asian/accumulation = first 1/3, London/manipulation = middle 1/3, NY/distribution = last 1/3
        split1 = n_bars // 3
        split2 = 2 * n_bars // 3

        asian = day_df.iloc[:split1]
        london = day_df.iloc[split1:split2]
        ny = day_df.iloc[split2:]

        if len(asian) < 2 or len(london) < 2 or len(ny) < 1:
            continue

        asian_high = asian["High"].max()
        asian_low = asian["Low"].min()
        asian_range = asian_high - asian_low
        if asian_range <= 0:
            continue

        london_high = london["High"].max()
        london_low = london["Low"].min()

        ny_first_idx = ny.index[0]
        # Map back to position in original df
        if ny_first_idx in df.index:
            bar_idx = df.index.get_loc(ny_first_idx)
        else:
            # Try finding closest match
            try:
                bar_idx = df.index.get_indexer([ny_first_idx], method="nearest")[0]
            except Exception:
                continue

        entry = float(ny["Open"].iloc[0])
        sl_buffer = asian_range * 0.3

        # Bearish manipulation (London swept below Asian lows) → Bullish distribution
        if london_low < asian_low:
            sl = london_low - sl_buffer
            tp = entry + asian_range * 1.5
            if entry > sl and tp > entry:
                signals.append({
                    "bar": int(bar_idx), "direction": "buy", "entry": entry,
                    "sl": sl, "tp": tp, "strategy": "amd_cycle",
                    "reason": "Bullish AMD: manipulation swept Asian lows"
                })

        # Bullish manipulation (London swept above Asian highs) → Bearish distribution
        elif london_high > asian_high:
            sl = london_high + sl_buffer
            tp = entry - asian_range * 1.5
            if entry < sl and tp < entry:
                signals.append({
                    "bar": int(bar_idx), "direction": "sell", "entry": entry,
                    "sl": sl, "tp": tp, "strategy": "amd_cycle",
                    "reason": "Bearish AMD: manipulation swept Asian highs"
                })

    return signals


def strategy_goldbach_momentum(df: pd.DataFrame, lookback: int = 20) -> list:
    """
    Strategy 3: Goldbach + RSI Momentum
    - Buy at discount Goldbach level when RSI < 30
    - Sell at premium Goldbach level when RSI > 70
    """
    signals = []

    if HAS_TA:
        rsi = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    else:
        # Manual RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

    highs, lows = _rolling_high_low(df, lookback)

    for i in range(lookback + 1, len(df)):
        h, l = highs.iloc[i], lows.iloc[i]
        if pd.isna(h) or pd.isna(l) or h <= l:
            continue
        if pd.isna(rsi.iloc[i]):
            continue

        close = df["Close"].iloc[i]
        rsi_val = rsi.iloc[i]
        gb = calculate_goldbach_levels(h, l)
        zone = price_in_zone(close, h, l)
        rng = h - l

        key_levels = [lv for lv in gb["levels"] if lv["power"] in (3, 9, 27)]
        nearest = get_nearest_goldbach_level(close, key_levels)
        if nearest is None:
            continue

        dist = abs(close - nearest["price"])
        if dist > rng * 0.015:
            continue

        if zone == "discount" and rsi_val < 30:
            sl = l - rng * 0.02
            tp = gb["equilibrium"] + rng * 0.1
            if tp > close and close > sl:
                signals.append({
                    "bar": i, "direction": "buy", "entry": close,
                    "sl": sl, "tp": tp, "strategy": "goldbach_momentum",
                    "reason": f"Buy {nearest['label']} RSI={rsi_val:.0f}"
                })
        elif zone == "premium" and rsi_val > 70:
            sl = h + rng * 0.02
            tp = gb["equilibrium"] - rng * 0.1
            if tp < close and close < sl:
                signals.append({
                    "bar": i, "direction": "sell", "entry": close,
                    "sl": sl, "tp": tp, "strategy": "goldbach_momentum",
                    "reason": f"Sell {nearest['label']} RSI={rsi_val:.0f}"
                })

    return signals


def strategy_po3_breakout(df: pd.DataFrame, lookback: int = 20) -> list:
    """
    Strategy 4: PO3 Breakout
    - When price breaks through a PO3 level, trade the continuation
    """
    signals = []
    highs, lows = _rolling_high_low(df, lookback)

    for i in range(lookback + 2, len(df)):
        h, l = highs.iloc[i], lows.iloc[i]
        if pd.isna(h) or pd.isna(l) or h <= l:
            continue

        rng = h - l
        po3 = get_po3_levels(h, l, 3)

        prev_close = df["Close"].iloc[i - 1]
        curr_close = df["Close"].iloc[i]

        for level in po3:
            # Bullish breakout: prev below, curr above
            if prev_close < level and curr_close > level:
                sl = level - rng * 0.03
                next_levels = [p for p in po3 if p > level]
                tp = next_levels[0] if next_levels else h
                if tp > curr_close and curr_close > sl:
                    signals.append({
                        "bar": i, "direction": "buy", "entry": curr_close,
                        "sl": sl, "tp": tp, "strategy": "po3_breakout",
                        "reason": f"Bullish PO3 break at {level:.5f}"
                    })
                break

            # Bearish breakout: prev above, curr below
            if prev_close > level and curr_close < level:
                sl = level + rng * 0.03
                prev_levels = [p for p in po3 if p < level]
                tp = prev_levels[-1] if prev_levels else l
                if tp < curr_close and curr_close < sl:
                    signals.append({
                        "bar": i, "direction": "sell", "entry": curr_close,
                        "sl": sl, "tp": tp, "strategy": "po3_breakout",
                        "reason": f"Bearish PO3 break at {level:.5f}"
                    })
                break

    return signals


def strategy_multi_tf_goldbach(df_daily: pd.DataFrame, df_hourly: pd.DataFrame, lookback_w: int = 5, lookback_d: int = 20) -> list:
    """
    Strategy 5: Multi-Timeframe Goldbach
    - Weekly dealing range for bias
    - Daily dealing range for entry
    - Enter when levels align
    """
    signals = []
    if df_daily.empty or df_hourly.empty:
        return signals

    weekly_high = df_daily["High"].rolling(lookback_w * 5).max()
    weekly_low = df_daily["Low"].rolling(lookback_w * 5).min()

    daily_high = df_daily["High"].rolling(lookback_d).max()
    daily_low = df_daily["Low"].rolling(lookback_d).min()

    for i in range(lookback_w * 5 + 1, len(df_daily)):
        wh, wl = weekly_high.iloc[i], weekly_low.iloc[i]
        dh, dl = daily_high.iloc[i], daily_low.iloc[i]

        if pd.isna(wh) or pd.isna(wl) or pd.isna(dh) or pd.isna(dl):
            continue
        if wh <= wl or dh <= dl:
            continue

        close = df_daily["Close"].iloc[i]

        weekly_zone = price_in_zone(close, wh, wl)
        daily_gb = calculate_goldbach_levels(dh, dl)
        daily_zone = price_in_zone(close, dh, dl)
        rng = dh - dl

        key_levels = [lv for lv in daily_gb["levels"] if lv["power"] in (3, 9)]
        nearest = get_nearest_goldbach_level(close, key_levels)
        if nearest is None:
            continue

        dist = abs(close - nearest["price"])
        if dist > rng * 0.02:
            continue

        # Weekly discount + daily discount = strong buy
        if weekly_zone == "discount" and daily_zone == "discount":
            sl = dl - rng * 0.03
            tp = daily_gb["equilibrium"] + rng * 0.15
            if tp > close and close > sl:
                signals.append({
                    "bar": i, "direction": "buy", "entry": close,
                    "sl": sl, "tp": tp, "strategy": "multi_tf_goldbach",
                    "reason": f"Weekly+Daily discount at {nearest['label']}"
                })
        elif weekly_zone == "premium" and daily_zone == "premium":
            sl = dh + rng * 0.03
            tp = daily_gb["equilibrium"] - rng * 0.15
            if tp < close and close < sl:
                signals.append({
                    "bar": i, "direction": "sell", "entry": close,
                    "sl": sl, "tp": tp, "strategy": "multi_tf_goldbach",
                    "reason": f"Weekly+Daily premium at {nearest['label']}"
                })

    return signals


ALL_STRATEGIES = {
    "goldbach_bounce": strategy_goldbach_bounce,
    "amd_cycle": strategy_amd_cycle,
    "goldbach_momentum": strategy_goldbach_momentum,
    "po3_breakout": strategy_po3_breakout,
}
