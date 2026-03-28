"""
Strategy 6: ICT Continuation Model
Based on the continuation-to-drawn-liquidity approach:
1. Identify drawn liquidity (swing highs/lows, session extremes)
2. Wait for retracement into a FVG
3. Look for displacement/inverse FVG confirmation
4. Enter targeting drawn liquidity
5. Break even at internal structure
"""
import pandas as pd
import numpy as np
from typing import Optional, List


def find_swing_highs(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Identify swing highs (local maxima)."""
    highs = df["High"]
    swing_highs = pd.Series(np.nan, index=df.index)
    for i in range(lookback, len(df) - lookback):
        if highs.iloc[i] == highs.iloc[i - lookback:i + lookback + 1].max():
            swing_highs.iloc[i] = highs.iloc[i]
    return swing_highs


def find_swing_lows(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Identify swing lows (local minima)."""
    lows = df["Low"]
    swing_lows = pd.Series(np.nan, index=df.index)
    for i in range(lookback, len(df) - lookback):
        if lows.iloc[i] == lows.iloc[i - lookback:i + lookback + 1].min():
            swing_lows.iloc[i] = lows.iloc[i]
    return swing_lows


def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Fair Value Gaps (3-candle pattern where wicks don't overlap).
    Returns DataFrame with FVG info for each bar.
    """
    fvgs = []
    for i in range(2, len(df)):
        # Bullish FVG: candle1 high < candle3 low (gap up)
        if df["High"].iloc[i - 2] < df["Low"].iloc[i]:
            fvgs.append({
                "bar": i,
                "type": "bullish",
                "top": df["Low"].iloc[i],
                "bottom": df["High"].iloc[i - 2],
                "mid": (df["Low"].iloc[i] + df["High"].iloc[i - 2]) / 2,
            })
        # Bearish FVG: candle1 low > candle3 high (gap down)
        elif df["Low"].iloc[i - 2] > df["High"].iloc[i]:
            fvgs.append({
                "bar": i,
                "type": "bearish",
                "top": df["Low"].iloc[i - 2],
                "bottom": df["High"].iloc[i],
                "mid": (df["Low"].iloc[i - 2] + df["High"].iloc[i]) / 2,
            })
    return fvgs


def is_displacement(candle_range: float, avg_range: float, threshold: float = 1.5) -> bool:
    """Check if a candle is displacing (significantly larger than average)."""
    return candle_range > avg_range * threshold


def get_trend(df: pd.DataFrame, i: int, lookback: int = 20) -> str:
    """Determine trend based on higher highs/lows or lower highs/lows."""
    if i < lookback:
        return "neutral"
    segment = df.iloc[i - lookback:i]
    sma_short = segment["Close"].iloc[-5:].mean()
    sma_long = segment["Close"].mean()
    if sma_short > sma_long:
        return "bullish"
    elif sma_short < sma_long:
        return "bearish"
    return "neutral"


def strategy_continuation(df: pd.DataFrame, swing_lookback: int = 10) -> list:
    """
    ICT Continuation Model:
    - Find drawn liquidity (swing highs in uptrend, swing lows in downtrend)
    - Wait for retracement into a FVG
    - Confirm with displacement candle
    - Enter targeting drawn liquidity
    """
    signals = []

    if len(df) < 50:
        return signals

    # Pre-compute
    swing_highs = find_swing_highs(df, swing_lookback)
    swing_lows = find_swing_lows(df, swing_lookback)
    fvgs = detect_fvg(df)
    avg_range = (df["High"] - df["Low"]).rolling(20).mean()

    # Track active FVGs (unfilled)
    active_bullish_fvgs = []
    active_bearish_fvgs = []

    for i in range(30, len(df)):
        close = df["Close"].iloc[i]
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        candle_range = high - low
        prev_close = df["Close"].iloc[i - 1]

        if pd.isna(avg_range.iloc[i]) or avg_range.iloc[i] <= 0:
            continue

        trend = get_trend(df, i)

        # Add new FVGs from this bar
        for fvg in fvgs:
            if fvg["bar"] == i:
                if fvg["type"] == "bullish":
                    active_bullish_fvgs.append(fvg)
                else:
                    active_bearish_fvgs.append(fvg)

        # Clean up old FVGs (keep last 20)
        active_bullish_fvgs = active_bullish_fvgs[-20:]
        active_bearish_fvgs = active_bearish_fvgs[-20:]

        # === BULLISH CONTINUATION ===
        if trend == "bullish":
            # Find drawn liquidity: nearest unfilled swing high above
            drawn_liq = None
            for j in range(i - 1, max(i - 50, 0), -1):
                if not pd.isna(swing_highs.iloc[j]) and swing_highs.iloc[j] > close:
                    drawn_liq = swing_highs.iloc[j]
                    break

            if drawn_liq is None:
                # Use recent high as target
                recent_high = df["High"].iloc[max(0, i - 20):i].max()
                if recent_high > close * 1.001:
                    drawn_liq = recent_high
                else:
                    continue

            # Check if price is retracing into a bullish FVG
            for fvg in active_bullish_fvgs:
                if fvg["bar"] >= i - 3:
                    continue  # Too recent

                # Price entering the FVG zone
                if low <= fvg["top"] and low >= fvg["bottom"]:
                    # Confirmation: displacement candle (bullish close above FVG)
                    if close > fvg["top"] and is_displacement(candle_range, avg_range.iloc[i], 1.3):
                        # Check for inverse FVG (previous bearish FVG now broken)
                        has_ifvg = False
                        for bfvg in active_bearish_fvgs:
                            if bfvg["bar"] > fvg["bar"] and bfvg["bar"] < i:
                                if close > bfvg["top"]:
                                    has_ifvg = True
                                    break

                        entry = close
                        sl = fvg["bottom"] - candle_range * 0.1
                        tp = drawn_liq

                        # Validate R:R
                        risk = entry - sl
                        reward = tp - entry
                        if risk <= 0 or reward <= 0:
                            continue
                        rr = reward / risk
                        if rr < 1.5:
                            continue

                        signals.append({
                            "bar": i,
                            "direction": "buy",
                            "entry": entry,
                            "sl": sl,
                            "tp": tp,
                            "strategy": "continuation",
                            "reason": f"Bullish continuation to DL={drawn_liq:.2f}, FVG reaction{' +IFVG' if has_ifvg else ''}, R:R={rr:.1f}"
                        })
                        # Remove used FVG
                        if fvg in active_bullish_fvgs:
                            active_bullish_fvgs.remove(fvg)
                        break

        # === BEARISH CONTINUATION ===
        elif trend == "bearish":
            # Find drawn liquidity: nearest unfilled swing low below
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

            # Check if price is retracing into a bearish FVG
            for fvg in active_bearish_fvgs:
                if fvg["bar"] >= i - 3:
                    continue

                # Price entering the FVG zone
                if high >= fvg["bottom"] and high <= fvg["top"]:
                    # Confirmation: displacement candle (bearish close below FVG)
                    if close < fvg["bottom"] and is_displacement(candle_range, avg_range.iloc[i], 1.3):
                        has_ifvg = False
                        for bfvg in active_bullish_fvgs:
                            if bfvg["bar"] > fvg["bar"] and bfvg["bar"] < i:
                                if close < bfvg["bottom"]:
                                    has_ifvg = True
                                    break

                        entry = close
                        sl = fvg["top"] + candle_range * 0.1
                        tp = drawn_liq

                        risk = sl - entry
                        reward = entry - tp
                        if risk <= 0 or reward <= 0:
                            continue
                        rr = reward / risk
                        if rr < 1.5:
                            continue

                        signals.append({
                            "bar": i,
                            "direction": "sell",
                            "entry": entry,
                            "sl": sl,
                            "tp": tp,
                            "strategy": "continuation",
                            "reason": f"Bearish continuation to DL={drawn_liq:.2f}, FVG reaction{' +IFVG' if has_ifvg else ''}, R:R={rr:.1f}"
                        })
                        if fvg in active_bearish_fvgs:
                            active_bearish_fvgs.remove(fvg)
                        break

    return signals


def strategy_range_breakout_retest(df: pd.DataFrame, range_period: int = 15) -> list:
    """
    ICT Range Breakout + Retest Model:
    - Identify consolidation range
    - Wait for breakout with displacement
    - Enter on retest of FVG created by breakout
    - Target drawn liquidity
    """
    signals = []

    if len(df) < range_period + 20:
        return signals

    avg_range = (df["High"] - df["Low"]).rolling(20).mean()

    for i in range(range_period + 10, len(df)):
        if pd.isna(avg_range.iloc[i]):
            continue

        # Detect consolidation: look for tight range over last N bars
        segment = df.iloc[i - range_period:i]
        range_high = segment["High"].max()
        range_low = segment["Low"].min()
        total_range = range_high - range_low
        avg_candle = (segment["High"] - segment["Low"]).mean()

        # Consolidation = range is less than 4x average candle
        if total_range > avg_candle * 5:
            continue

        close = df["Close"].iloc[i]
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        candle_range = high - low

        # === BULLISH BREAKOUT ===
        if close > range_high and is_displacement(candle_range, avg_range.iloc[i], 1.3):
            # Look for FVG created by breakout
            if i >= 2 and df["High"].iloc[i - 2] < df["Low"].iloc[i]:
                fvg_top = df["Low"].iloc[i]
                fvg_bottom = df["High"].iloc[i - 2]

                # Check next bars for retest
                for j in range(i + 1, min(i + 10, len(df))):
                    if df["Low"].iloc[j] <= fvg_top and df["Close"].iloc[j] > fvg_bottom:
                        # Retest happened, look for drawn liquidity above
                        drawn_liq = None
                        for k in range(i - range_period - 30, i - range_period):
                            if k >= 0 and k < len(df):
                                if df["High"].iloc[k] > range_high * 1.002:
                                    if drawn_liq is None or df["High"].iloc[k] > drawn_liq:
                                        drawn_liq = df["High"].iloc[k]

                        if drawn_liq is None:
                            drawn_liq = range_high + total_range * 2

                        entry = df["Close"].iloc[j]
                        sl = fvg_bottom - candle_range * 0.1
                        tp = drawn_liq

                        risk = entry - sl
                        reward = tp - entry
                        if risk <= 0 or reward <= 0:
                            break
                        rr = reward / risk
                        if rr < 1.5:
                            break

                        signals.append({
                            "bar": j,
                            "direction": "buy",
                            "entry": entry,
                            "sl": sl,
                            "tp": tp,
                            "strategy": "range_breakout_retest",
                            "reason": f"Bullish range breakout retest, FVG={fvg_bottom:.2f}-{fvg_top:.2f}, R:R={rr:.1f}"
                        })
                        break

        # === BEARISH BREAKOUT ===
        elif close < range_low and is_displacement(candle_range, avg_range.iloc[i], 1.3):
            if i >= 2 and df["Low"].iloc[i - 2] > df["High"].iloc[i]:
                fvg_top = df["Low"].iloc[i - 2]
                fvg_bottom = df["High"].iloc[i]

                for j in range(i + 1, min(i + 10, len(df))):
                    if df["High"].iloc[j] >= fvg_bottom and df["Close"].iloc[j] < fvg_top:
                        drawn_liq = None
                        for k in range(i - range_period - 30, i - range_period):
                            if k >= 0 and k < len(df):
                                if df["Low"].iloc[k] < range_low * 0.998:
                                    if drawn_liq is None or df["Low"].iloc[k] < drawn_liq:
                                        drawn_liq = df["Low"].iloc[k]

                        if drawn_liq is None:
                            drawn_liq = range_low - total_range * 2

                        entry = df["Close"].iloc[j]
                        sl = fvg_top + candle_range * 0.1
                        tp = drawn_liq

                        risk = sl - entry
                        reward = entry - tp
                        if risk <= 0 or reward <= 0:
                            break
                        rr = reward / risk
                        if rr < 1.5:
                            break

                        signals.append({
                            "bar": j,
                            "direction": "sell",
                            "entry": entry,
                            "sl": sl,
                            "tp": tp,
                            "strategy": "range_breakout_retest",
                            "reason": f"Bearish range breakout retest, FVG={fvg_bottom:.2f}-{fvg_top:.2f}, R:R={rr:.1f}"
                        })
                        break

    return signals
