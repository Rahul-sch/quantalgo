#!/usr/bin/env python3
"""
QQQ Algo Trading Dashboard
Streamlit + Plotly live visual dashboard showing algorithm decisions on an
interactive candlestick chart with FVG overlays, regime cards, and trade log.

Usage:
    streamlit run dashboard.py
"""

import os
import json
import sys
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QQQ Algo · Live",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — dark terminal aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'Inter', 'SF Pro Display', -apple-system, sans-serif;
  }
  .stApp { background: #080c10; }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1rem; padding-bottom: 0.5rem; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2329;
  }
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label { color: #8b9299 !important; }

  /* ── Metric cards ── */
  .regime-card {
    background: #0d1117;
    border: 1px solid #1e2329;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .regime-card:hover { border-color: #30363d; }
  .regime-card .label { font-size: 0.75em; color: #6e7681; text-transform: uppercase; letter-spacing: 0.08em; }
  .regime-card .value { font-size: 2.1em; font-weight: 700; color: #e6edf3; margin: 4px 0; line-height: 1.1; }
  .regime-card .sub   { font-size: 0.78em; color: #8b9299; }
  .regime-card .badge {
    display: inline-block; margin-top: 8px;
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.75em; font-weight: 600; letter-spacing: 0.04em;
  }
  .badge-green { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid rgba(63,185,80,0.3); }
  .badge-red   { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid rgba(248,81,73,0.3); }
  .badge-yellow{ background: rgba(210,153,34,0.15); color: #d2993c; border: 1px solid rgba(210,153,34,0.3); }
  .badge-gray  { background: rgba(110,118,129,0.15);color: #8b9299; border: 1px solid rgba(110,118,129,0.3); }

  /* ── Session banner ── */
  .session-banner {
    border-radius: 8px; padding: 10px 18px;
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 4px;
  }
  .session-banner .dot {
    width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0;
    animation: pulse 1.8s infinite;
  }
  @keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(63,185,80, 0.6); }
    70%  { box-shadow: 0 0 0 7px rgba(63,185,80, 0); }
    100% { box-shadow: 0 0 0 0 rgba(63,185,80, 0); }
  }

  /* ── Trade table ── */
  .stDataFrame { background: #0d1117 !important; }
  .stDataFrame td, .stDataFrame th {
    border-color: #1e2329 !important;
    background: #0d1117 !important;
    color: #c9d1d9 !important;
  }

  /* ── Section headers ── */
  h2, h3 { color: #e6edf3 !important; font-weight: 600 !important; }

  /* ── Divider ── */
  hr { border-color: #1e2329 !important; }

  /* ── Stat row ── */
  .stat-row {
    display: flex; gap: 16px; flex-wrap: wrap; margin: 6px 0;
  }
  .stat-item {
    background: #0d1117; border: 1px solid #1e2329;
    border-radius: 8px; padding: 10px 16px; flex: 1; min-width: 90px;
  }
  .stat-item .s-label { font-size: 0.72em; color: #6e7681; text-transform: uppercase; letter-spacing: 0.06em; }
  .stat-item .s-value { font-size: 1.35em; font-weight: 700; color: #e6edf3; }
  .stat-item .s-value.green { color: #3fb950; }
  .stat-item .s-value.red   { color: #f85149; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh support (graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ─────────────────────────────────────────────────────────────────────────────
# Optional imports — dashboard degrades gracefully if missing
# ─────────────────────────────────────────────────────────────────────────────
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    sys.path.insert(0, os.path.dirname(__file__))
    from quant_engine import Config, generate_signals, compute_indicators
    HAS_ENGINE = True
except Exception as _e:
    HAS_ENGINE = False
    _ENGINE_ERROR = str(_e)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

CSV_FILES = {
    "QQQ": {
        "15m": os.path.join(DATA_DIR, "QQQ_15m_2yr.csv"),
        "5m":  os.path.join(DATA_DIR, "QQQ_5m.csv"),
        "1h":  os.path.join(DATA_DIR, "QQQ_1h.csv"),
    },
    "SPY": {
        "15m": os.path.join(DATA_DIR, "SPY_15m_2yr.csv"),
        "5m":  os.path.join(DATA_DIR, "SPY_5m.csv"),
        "1h":  os.path.join(DATA_DIR, "SPY_1h.csv"),
    },
}
# Primary fallback — always available
CSV_FILES["QQQ"]["15m_primary"] = os.path.join(DATA_DIR, "QQQ_15m_2yr.csv")

BLACKOUT_FILE = os.path.join(DATA_DIR, "blackout_windows.json")
PAPER_TRADES_JSON = os.path.join(RESULTS_DIR, "paper_trades.json")
PAPER_TRADES_CSV = os.path.join(RESULTS_DIR, "paper_trades_log.csv")

# ─────────────────────────────────────────────────────────────────────────────
# ── Sidebar ──
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ QQQ Algo")
    st.markdown("<div style='color:#6e7681;font-size:0.8em;margin-bottom:16px'>FVG Retest · 15m · AM Session</div>", unsafe_allow_html=True)

    symbol = st.selectbox("Symbol", ["QQQ", "SPY"], index=0)
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "1h"], index=1)
    lookback_days = st.slider("Lookback (days)", min_value=1, max_value=30, value=5)

    st.markdown("---")
    auto_refresh = st.toggle("Auto-Refresh", value=False)
    refresh_interval_map = {"30s": 30_000, "60s": 60_000, "5m": 300_000}
    refresh_interval_label = st.selectbox(
        "Interval", list(refresh_interval_map.keys()), index=1,
        disabled=not auto_refresh
    )
    refresh_ms = refresh_interval_map[refresh_interval_label]

    refresh_btn = st.button("🔄 Refresh Now", use_container_width=True)

    st.markdown("---")
    engine_status = "🟢 Active" if HAS_ENGINE else "🔴 Offline"
    st.markdown(f"<div style='color:#6e7681;font-size:0.8em'>Engine: <span style='color:#e6edf3'>{engine_status}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#6e7681;font-size:0.8em'>Data: <span style='color:#e6edf3'>{'✅' if HAS_YFINANCE else '❌'}</span></div>", unsafe_allow_html=True)

# Trigger auto-refresh if enabled
if auto_refresh and HAS_AUTOREFRESH:
    st_autorefresh(interval=refresh_ms, key="auto_refresh_counter")

# ─────────────────────────────────────────────────────────────────────────────
# ── Data loading (cached) ──
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_ohlcv(sym: str, tf: str) -> pd.DataFrame:
    """Load OHLCV from CSV file. Returns empty DataFrame on failure."""
    # Try the exact file first, then fallback to QQQ 15m primary
    candidates = []
    if sym in CSV_FILES and tf in CSV_FILES[sym]:
        candidates.append(CSV_FILES[sym][tf])
    # Fallback for QQQ 15m
    candidates.append(os.path.join(DATA_DIR, f"{sym}_{tf}_2yr.csv"))
    candidates.append(os.path.join(DATA_DIR, f"{sym}_{tf}.csv"))
    candidates.append(CSV_FILES["QQQ"]["15m_primary"])  # last resort

    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                # Normalize column names
                df.columns = [c.strip().title() for c in df.columns]
                if "Timestamp" in df.columns:
                    df = df.rename(columns={"Timestamp": "timestamp"})
                # Ensure standard OHLCV columns exist
                required = {"Open", "High", "Low", "Close"}
                if not required.issubset(df.columns):
                    continue
                # Ensure DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, utc=True)
                # Normalize timezone to US/Eastern
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
                else:
                    df.index = df.index.tz_convert("US/Eastern")
                df = df.sort_index()
                return df
            except Exception:
                continue

    return pd.DataFrame()


@st.cache_data(ttl=60)
def fetch_vix_data() -> dict:
    """Fetch latest VIX from yfinance."""
    if not HAS_YFINANCE:
        return {"error": "yfinance not installed"}
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        if hist.empty:
            return {"error": "No VIX data"}
        current = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current
        return {"current": current, "prev": prev, "delta": current - prev}
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300)
def fetch_macro_trend(sym: str) -> dict:
    """Fetch daily data and compute 20-Month SMA for macro trend card."""
    if not HAS_YFINANCE:
        return {"error": "yfinance not installed"}
    try:
        ticker = yf.Ticker(sym)
        # Get ~3 years of monthly data
        hist = ticker.history(period="3y", interval="1mo")
        if hist.empty or len(hist) < 5:
            return {"error": "Insufficient data"}
        closes = hist["Close"]
        sma20 = closes.rolling(20, min_periods=5).mean()
        current_price = float(closes.iloc[-1])
        current_sma = float(sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else None
        if current_sma and current_sma > 0:
            pct_diff = (current_price - current_sma) / current_sma * 100
            above = current_price > current_sma
            return {
                "price": current_price,
                "sma20": current_sma,
                "pct_diff": pct_diff,
                "above": above,
            }
        return {"error": "SMA not computable"}
    except Exception as e:
        return {"error": str(e)}


def load_blackout_windows() -> list:
    """Load blackout windows from JSON. Returns list of [start, end] strings."""
    if not os.path.exists(BLACKOUT_FILE):
        return []
    try:
        with open(BLACKOUT_FILE) as f:
            data = json.load(f)
        return data.get("windows", [])
    except Exception:
        return []


def get_next_blackout(windows: list) -> Optional[dict]:
    """Find the next upcoming blackout window from now (US/Eastern)."""
    try:
        import pytz
        et = pytz.timezone("US/Eastern")
        now = datetime.now(et)
    except ImportError:
        now = datetime.now()

    upcoming = []
    for w in windows:
        try:
            start = pd.to_datetime(w[0])
            end = pd.to_datetime(w[1])
            if start.tzinfo is None:
                try:
                    import pytz
                    start = pytz.timezone("US/Eastern").localize(start)
                    end = pytz.timezone("US/Eastern").localize(end)
                except ImportError:
                    pass
            if start >= now:
                upcoming.append({"start": start, "end": end})
        except Exception:
            continue

    if not upcoming:
        return None

    upcoming.sort(key=lambda x: x["start"])
    nxt = upcoming[0]

    # Check if today
    today = now.date()
    start_date = nxt["start"].date() if hasattr(nxt["start"], "date") else today
    if start_date == today:
        day_label = "Today"
    elif (start_date - today).days == 1:
        day_label = "Tomorrow"
    else:
        day_label = nxt["start"].strftime("%b %d") if hasattr(nxt["start"], "strftime") else str(start_date)

    try:
        start_str = nxt["start"].strftime("%H:%M")
        end_str = nxt["end"].strftime("%H:%M")
    except Exception:
        start_str = "??"
        end_str = "??"

    return {
        "label": f"{day_label} {start_str}–{end_str} ET",
        "start": nxt["start"],
        "end": nxt["end"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# ── FVG Detection (dashboard-local vectorized version) ──
# ─────────────────────────────────────────────────────────────────────────────

def detect_fvgs(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Detect Fair Value Gaps (3-candle pattern).
    Bull FVG: candle[i-2].high < candle[i].low   (gap between them)
    Bear FVG: candle[i-2].low > candle[i].high
    Returns (bull_top, bull_bot, bear_top, bear_bot) as Series indexed like df.
    """
    high = df["High"]
    low = df["Low"]

    bull_top = pd.Series(np.nan, index=df.index)
    bull_bot = pd.Series(np.nan, index=df.index)
    bear_top = pd.Series(np.nan, index=df.index)
    bear_bot = pd.Series(np.nan, index=df.index)

    h2 = high.shift(2)
    l2 = low.shift(2)

    # Bullish FVG: gap[i-2].high < gap[i].low
    bull_mask = h2 < low
    bull_top[bull_mask] = low[bull_mask]          # top of gap = current bar low
    bull_bot[bull_mask] = h2[bull_mask]           # bot of gap = candle[i-2] high

    # Bearish FVG: gap[i-2].low > gap[i].high
    bear_mask = l2 > high
    bear_top[bear_mask] = l2[bear_mask]           # top of gap = candle[i-2] low
    bear_bot[bear_mask] = high[bear_mask]         # bot of gap = current bar high

    return bull_top, bull_bot, bear_top, bear_bot


# ─────────────────────────────────────────────────────────────────────────────
# ── Chart Building ──
# ─────────────────────────────────────────────────────────────────────────────

def build_chart(df: pd.DataFrame, signals: list, fvg_max_age: int = 20) -> go.Figure:
    """
    Build the main Plotly candlestick chart with:
    - FVG shaded rectangles
    - Buy/sell signal markers
    - TP/SL horizontal lines
    - AM session shading
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.80, 0.20],
    )

    # ── Candlestick ──
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
        ),
        row=1, col=1,
    )

    # ── Volume bars ──
    if "Volume" in df.columns:
        colors = [
            "#26a69a" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ef5350"
            for i in range(len(df))
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7,
                showlegend=False,
            ),
            row=2, col=1,
        )

    # ── AM Session shading (9:30–11:30 ET) ──
    try:
        dates_in_df = pd.Series(df.index.date).unique()
        for day in dates_in_df:
            am_start = pd.Timestamp(day).tz_localize("US/Eastern").replace(hour=9, minute=30)
            am_end = pd.Timestamp(day).tz_localize("US/Eastern").replace(hour=11, minute=30)
            # Only shade if within chart range
            if am_start > df.index[-1] or am_end < df.index[0]:
                continue
            fig.add_vrect(
                x0=am_start,
                x1=am_end,
                fillcolor="rgba(100, 149, 237, 0.06)",
                line_width=0,
                layer="below",
            )
    except Exception:
        pass  # Don't break chart over shading errors

    # ── FVG rectangles ──
    bull_top, bull_bot, bear_top, bear_bot = detect_fvgs(df)
    n = len(df)
    last_50_start = max(0, n - 50)

    for i in range(last_50_start, n):
        bar_time = df.index[i]
        end_idx = min(i + fvg_max_age, n - 1)
        end_time = df.index[end_idx]

        # Bullish FVG
        if not np.isnan(bull_top.iloc[i]):
            fig.add_shape(
                type="rect",
                x0=bar_time, x1=end_time,
                y0=float(bull_bot.iloc[i]), y1=float(bull_top.iloc[i]),
                fillcolor="rgba(38, 166, 154, 0.2)",
                line=dict(width=0),
                layer="below",
            )

        # Bearish FVG
        if not np.isnan(bear_top.iloc[i]):
            fig.add_shape(
                type="rect",
                x0=bar_time, x1=end_time,
                y0=float(bear_bot.iloc[i]), y1=float(bear_top.iloc[i]),
                fillcolor="rgba(239, 83, 80, 0.2)",
                line=dict(width=0),
                layer="below",
            )

    # ── Signal markers + TP/SL lines ──
    if signals:
        buy_x, buy_y, buy_text = [], [], []
        sell_x, sell_y, sell_text = [], [], []

        for sig in signals:
            try:
                bar_time = sig.get("bar_time") or sig.get("time") or sig.get("entry_time")
                entry = float(sig.get("entry", 0))
                sl = float(sig.get("stop", sig.get("sl", 0)))
                tp = float(sig.get("target", sig.get("tp", 0)))
                direction = sig.get("direction", "")

                if not bar_time or not entry:
                    continue

                # Convert bar_time to Timestamp if needed
                if not isinstance(bar_time, pd.Timestamp):
                    bar_time = pd.Timestamp(bar_time)

                label = f"Entry: {entry:.2f}<br>SL: {sl:.2f}<br>TP: {tp:.2f}"

                if direction == "buy":
                    buy_x.append(bar_time)
                    buy_y.append(entry * 0.9995)  # slightly below candle
                    buy_text.append(label)
                    # TP line
                    if tp:
                        fig.add_shape(type="line",
                            x0=bar_time, x1=df.index[-1],
                            y0=tp, y1=tp,
                            line=dict(color="rgba(38,166,154,0.7)", width=1, dash="dash"),
                            layer="above")
                    # SL line
                    if sl:
                        fig.add_shape(type="line",
                            x0=bar_time, x1=df.index[-1],
                            y0=sl, y1=sl,
                            line=dict(color="rgba(239,83,80,0.7)", width=1, dash="dash"),
                            layer="above")

                elif direction == "sell":
                    sell_x.append(bar_time)
                    sell_y.append(entry * 1.0005)  # slightly above candle
                    sell_text.append(label)
                    if tp:
                        fig.add_shape(type="line",
                            x0=bar_time, x1=df.index[-1],
                            y0=tp, y1=tp,
                            line=dict(color="rgba(38,166,154,0.7)", width=1, dash="dash"),
                            layer="above")
                    if sl:
                        fig.add_shape(type="line",
                            x0=bar_time, x1=df.index[-1],
                            y0=sl, y1=sl,
                            line=dict(color="rgba(239,83,80,0.7)", width=1, dash="dash"),
                            layer="above")

            except Exception:
                continue

        if buy_x:
            fig.add_trace(go.Scatter(
                x=buy_x, y=buy_y,
                mode="markers",
                name="Buy Signal",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="#26a69a",
                    line=dict(color="white", width=1),
                ),
                hovertext=buy_text,
                hoverinfo="text",
            ), row=1, col=1)

        if sell_x:
            fig.add_trace(go.Scatter(
                x=sell_x, y=sell_y,
                mode="markers",
                name="Sell Signal",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color="#ef5350",
                    line=dict(color="white", width=1),
                ),
                hovertext=sell_text,
                hoverinfo="text",
            ), row=1, col=1)

    # ── Layout ──
    fig.update_layout(
        title=dict(
            text=f"{symbol} · {timeframe} · Last {lookback_days}d",
            font=dict(size=16),
        ),
        xaxis_rangeslider_visible=False,
        height=620,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(
        gridcolor="#1e2329",
        zeroline=False,
        showspikes=True,
        spikecolor="#555",
    )
    fig.update_yaxes(
        gridcolor="#1e2329",
        zeroline=False,
        showspikes=True,
        spikecolor="#555",
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ── Main render ──
# ─────────────────────────────────────────────────────────────────────────────

try:
    import pytz
    et = pytz.timezone("US/Eastern")
    now_et = datetime.now(et)
except ImportError:
    now_et = datetime.now()
now_str = now_et.strftime("%H:%M:%S ET")
now_h = now_et.hour
now_m = now_et.minute
day_of_week = now_et.weekday()  # 0=Mon, 6=Sun

# ── Session status banner ──
is_weekend = day_of_week >= 5
is_am_session = (not is_weekend) and ((now_h == 9 and now_m >= 30) or (now_h == 10) or (now_h == 11 and now_m < 30))
is_pre_market = (not is_weekend) and (now_h < 9 or (now_h == 9 and now_m < 30))
is_post_market = (not is_weekend) and (now_h >= 16)

if is_am_session:
    banner_bg = "rgba(63,185,80,0.08)"
    banner_border = "rgba(63,185,80,0.3)"
    dot_color = "#3fb950"
    session_text = "🟢 AM SESSION LIVE — Engine scanning every 15 min"
elif is_pre_market:
    banner_bg = "rgba(210,153,34,0.08)"
    banner_border = "rgba(210,153,34,0.3)"
    dot_color = "#d2993c"
    session_text = f"⏳ Pre-market — AM session opens at 9:30 ET"
elif is_weekend:
    banner_bg = "rgba(110,118,129,0.08)"
    banner_border = "rgba(110,118,129,0.2)"
    dot_color = "#8b9299"
    session_text = "💤 Weekend — Market closed · Crons resume Monday 9:30 ET"
else:
    banner_bg = "rgba(110,118,129,0.08)"
    banner_border = "rgba(110,118,129,0.2)"
    dot_color = "#8b9299"
    session_text = "📴 Market closed — AM session window passed"

# Header row
hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown(
        f"<h1 style='color:#e6edf3;font-weight:700;margin:0;font-size:1.6em;'>⚡ QQQ Algo</h1>"
        f"<div style='color:#6e7681;font-size:0.85em;margin-top:2px'>FVG Retest · 15m · Institutional Paper Trading</div>",
        unsafe_allow_html=True
    )
with hcol2:
    st.markdown(
        f"<div style='text-align:right;color:#6e7681;font-size:0.8em;padding-top:10px'>{now_str}</div>",
        unsafe_allow_html=True
    )

st.markdown(
    f"""<div style="background:{banner_bg};border:1px solid {banner_border};border-radius:8px;
    padding:10px 16px;margin:12px 0 4px 0;color:#e6edf3;font-size:0.88em;">
    <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
    background:{dot_color};margin-right:8px;vertical-align:middle;"></span>
    {session_text}
    </div>""",
    unsafe_allow_html=True
)

# ─── Load OHLCV ───
with st.spinner("Loading price data…"):
    df_full = load_ohlcv(symbol, timeframe)

if df_full.empty:
    st.error(
        f"⚠️ No data found for {symbol} {timeframe}. "
        "Expected file: `data/QQQ_15m_2yr.csv`. "
        "Run `python3 data_fetcher.py` to download data."
    )
    st.stop()

# Filter to lookback window
cutoff = df_full.index[-1] - pd.Timedelta(days=lookback_days)
df = df_full[df_full.index >= cutoff].copy()

if df.empty:
    st.warning("No data in selected date range. Try increasing lookback days.")
    st.stop()

# ─── Generate signals ───
signals = []
if HAS_ENGINE:
    try:
        cfg = Config(
            use_vix_filter=False,
            use_blackout_filter=False,
            use_macro_veto=False,
        )
        # Generate signals on the full slice (need enough lookback for indicators)
        # Use a larger window for indicator warmup but only show signals in display range
        warmup_cutoff = df_full.index[-1] - pd.Timedelta(days=lookback_days + 30)
        df_warmup = df_full[df_full.index >= warmup_cutoff].copy()
        signals_all = generate_signals(df_warmup, cfg)
        # Filter to display range
        display_start = df.index[0]
        for sig in signals_all:
            try:
                bt = sig.get("bar_time") or sig.get("time") or sig.get("entry_time")
                if bt:
                    bt = pd.Timestamp(bt)
                    if bt >= display_start:
                        signals.append(sig)
            except Exception:
                pass
    except Exception as e:
        st.sidebar.warning(f"Signal generation error: {e}")
else:
    st.sidebar.warning(f"⚠️ Engine not loaded — signals disabled.\n{_ENGINE_ERROR if 'HAS_ENGINE' in dir() else ''}")

# ─── Regime Status + P&L Stats Row ───
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# Load trades for stats
def _load_trades_stats():
    path = os.path.join(os.path.dirname(__file__), "results", "paper_trades.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except Exception:
        return []

all_trades = _load_trades_stats()
closed = [t for t in all_trades if t.get("status", "").startswith("closed") or t.get("net_pnl") is not None]
open_trades = [t for t in all_trades if t.get("status") in ("open", "pending")]
wins = [t for t in closed if (t.get("net_pnl") or 0) > 0]
total_net = sum(t.get("net_pnl", 0) or 0 for t in closed)
win_rate = len(wins) / len(closed) * 100 if closed else 0
net_color = "green" if total_net >= 0 else "red"
wr_color = "green" if win_rate >= 40 else ("yellow" if win_rate >= 30 else "red")
pnl_sign = "+" if total_net >= 0 else ""

# Load daily state
def _load_daily():
    path = os.path.join(os.path.dirname(__file__), "results", "paper_daily_state.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

daily = _load_daily()
daily_pnl = daily.get("daily_pnl", 0.0)
halted = daily.get("halted", False)
daily_color = "green" if daily_pnl >= 0 else "red"
daily_sign = "+" if daily_pnl >= 0 else ""

# ── Stats strip ──
sc1, sc2, sc3, sc4, sc5 = st.columns(5)
stats = [
    (sc1, "NET P&L", f"{pnl_sign}${total_net:.2f}", net_color),
    (sc2, "TODAY",   f"{daily_sign}${daily_pnl:.2f}", daily_color),
    (sc3, "WIN RATE", f"{win_rate:.0f}%", wr_color),
    (sc4, "TRADES",  f"{len(closed)} closed", ""),
    (sc5, "OPEN",    f"{'🛑 HALTED' if halted else str(len(open_trades)) + ' pending'}", "red" if halted else ""),
]
for col, label, val, color in stats:
    with col:
        color_style = f"color:{'#3fb950' if color=='green' else '#f85149' if color=='red' else '#d2993c' if color=='yellow' else '#e6edf3'}"
        st.markdown(f"""
        <div class="stat-item">
            <div class="s-label">{label}</div>
            <div class="s-value" style="{color_style}">{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Regime cards ──
col1, col2, col3 = st.columns(3)

with col1:
    vix_data = fetch_vix_data()
    if "error" in vix_data:
        st.markdown("""<div class="regime-card">
            <div class="label">VIX Shield</div>
            <div class="value" style="font-size:1.2em;color:#6e7681">—</div>
            <div class="sub">Data unavailable</div>
            <span class="badge badge-gray">UNKNOWN</span>
        </div>""", unsafe_allow_html=True)
    else:
        vix_val = vix_data["current"]
        vix_delta = vix_data["delta"]
        badge = 'badge-green">🟢 CLEAR' if vix_val < 25 else 'badge-red">🔴 BLOCKED'
        delta_str = f"{vix_delta:+.2f} vs yesterday"
        st.markdown(f"""<div class="regime-card">
            <div class="label">VIX Shield · Threshold 25</div>
            <div class="value">{vix_val:.1f}</div>
            <div class="sub">{delta_str}</div>
            <span class="badge {badge}</span>
        </div>""", unsafe_allow_html=True)

with col2:
    macro_data = fetch_macro_trend(symbol)
    if "error" in macro_data:
        st.markdown("""<div class="regime-card">
            <div class="label">Macro Trend · 20M SMA</div>
            <div class="value" style="font-size:1.2em;color:#6e7681">—</div>
            <div class="sub">Data unavailable</div>
            <span class="badge badge-gray">UNKNOWN</span>
        </div>""", unsafe_allow_html=True)
    else:
        pct = macro_data["pct_diff"]
        above = macro_data["above"]
        sign = "+" if pct >= 0 else ""
        badge = 'badge-green">🟢 BULLISH' if above else 'badge-red">🔴 BEARISH'
        direction = "above" if above else "below"
        st.markdown(f"""<div class="regime-card">
            <div class="label">Macro Trend · 20M SMA</div>
            <div class="value">{sign}{pct:.1f}%</div>
            <div class="sub">{direction} 20-Month SMA</div>
            <span class="badge {badge}</span>
        </div>""", unsafe_allow_html=True)

with col3:
    blackout_windows = load_blackout_windows()
    if not blackout_windows:
        st.markdown("""<div class="regime-card">
            <div class="label">Calendar Blackout</div>
            <div class="value" style="font-size:1.2em;color:#6e7681">—</div>
            <div class="sub">Run paper_trader.py first</div>
            <span class="badge badge-gray">NOT LOADED</span>
        </div>""", unsafe_allow_html=True)
    else:
        next_bo = get_next_blackout(blackout_windows)
        if next_bo is None:
            val, sub, badge = "Clear", "No events today", 'badge-green">✅ ALL CLEAR'
        else:
            val = next_bo["label"][:12]
            sub = next_bo["label"]
            badge = 'badge-yellow">⚠️ EVENT AHEAD'
        st.markdown(f"""<div class="regime-card">
            <div class="label">Calendar Blackout</div>
            <div class="value" style="font-size:1.1em">{val}</div>
            <div class="sub">{sub}</div>
            <span class="badge {badge}</span>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ─── Chart ───
st.markdown("<div style='color:#e6edf3;font-weight:600;font-size:1.05em;margin-bottom:4px'>🕯️ Price Chart</div>", unsafe_allow_html=True)

if df.empty:
    st.warning("No chart data available.")
else:
    try:
        fig = build_chart(df, signals)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Showing {len(df):,} bars · "
            f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')} · "            f"{len(signals)} signals"
        )
    except Exception as e:
        st.error(f"Chart rendering error: {e}")

# ─── Equity Curve ───
if closed:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#e6edf3;font-weight:600;font-size:1.05em;margin-bottom:4px'>📈 Equity Curve</div>", unsafe_allow_html=True)

    INITIAL_CAPITAL = 50_000.0
    equity = [INITIAL_CAPITAL]
    sorted_closed = sorted(closed, key=lambda x: x.get("exit_timestamp") or x.get("timestamp") or "")
    trade_labels = ["Start"]
    for t in sorted_closed:
        pnl = t.get("net_pnl", 0) or 0
        equity.append(equity[-1] + pnl)
        ts = (t.get("exit_timestamp") or t.get("timestamp") or "")[:16]
        trade_labels.append(f"#{len(equity)-1} {ts}")

    colors = []
    for i in range(1, len(equity)):
        colors.append("#3fb950" if equity[i] >= equity[i-1] else "#f85149")

    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(
        x=list(range(len(equity))),
        y=equity,
        mode="lines+markers",
        line=dict(color="#58a6ff", width=2),
        marker=dict(
            color=["#8b9299"] + colors,
            size=7,
            line=dict(color="#080c10", width=1.5),
        ),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.06)",
        hovertext=trade_labels,
        hoverinfo="text+y",
        name="Equity",
    ))
    # Breakeven line
    eq_fig.add_hline(
        y=INITIAL_CAPITAL,
        line_dash="dot",
        line_color="rgba(110,118,129,0.5)",
        line_width=1,
    )
    peak = max(equity)
    max_dd = 0.0
    peak_eq = INITIAL_CAPITAL
    for e in equity:
        peak_eq = max(peak_eq, e)
        dd = (peak_eq - e) / peak_eq * 100 if peak_eq > 0 else 0
        max_dd = max(max_dd, dd)

    eq_fig.update_layout(
        height=200,
        paper_bgcolor="#080c10",
        plot_bgcolor="#080c10",
        font=dict(color="#8b9299", size=11),
        margin=dict(l=8, r=8, t=8, b=8),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(gridcolor="#1e2329", zeroline=False, tickprefix="$", tickformat=",.0f"),
    )
    st.plotly_chart(eq_fig, use_container_width=True)
    st.caption(f"Peak: ${peak:,.2f} · Max Drawdown: {max_dd:.1f}% · {len(wins)}W / {len(closed)-len(wins)}L")

# ─── Trade Log ───
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.markdown("<div style='color:#e6edf3;font-weight:600;font-size:1.05em;margin-bottom:4px'>📋 Trade Log</div>", unsafe_allow_html=True)

def style_trade_row(row):
    """Color trades by status/PnL."""
    pnl = row.get("P&L", row.get("pnl", 0))
    status = str(row.get("Status", row.get("status", ""))).lower()
    if status in ("open", "pending"):
        return ["background-color: rgba(255,200,0,0.15)"] * len(row)
    try:
        pnl_float = float(pnl)
        if pnl_float > 0:
            return ["background-color: rgba(38,166,154,0.15)"] * len(row)
        elif pnl_float < 0:
            return ["background-color: rgba(239,83,80,0.15)"] * len(row)
    except (TypeError, ValueError):
        pass
    return [""] * len(row)


trades_loaded = False

# Try JSON trades first
if os.path.exists(PAPER_TRADES_JSON):
    try:
        with open(PAPER_TRADES_JSON) as f:
            trades_raw = json.load(f)

        if isinstance(trades_raw, dict):
            open_trades = trades_raw.get("open", [])
            closed_trades = trades_raw.get("closed", [])
            all_trades = [{"Status": "open", **t} for t in open_trades] + \
                         [{"Status": "closed", **t} for t in closed_trades]
        elif isinstance(trades_raw, list):
            all_trades = trades_raw
        else:
            all_trades = []

        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            # Normalize columns
            col_map = {
                "time": "Time", "entry_time": "Time",
                "direction": "Direction",
                "entry": "Entry",
                "stop": "SL", "sl": "SL",
                "target": "TP", "tp": "TP",
                "status": "Status",
                "pnl": "P&L",
            }
            trades_df = trades_df.rename(columns={k: v for k, v in col_map.items() if k in trades_df.columns})
            display_cols = [c for c in ["Time", "Direction", "Entry", "SL", "TP", "Status", "P&L"] if c in trades_df.columns]
            trades_df = trades_df[display_cols]

            try:
                styled = trades_df.style.apply(style_trade_row, axis=1)
                st.dataframe(styled, use_container_width=True)
            except Exception:
                st.dataframe(trades_df, use_container_width=True)
            trades_loaded = True
    except Exception as e:
        st.warning(f"Could not load paper_trades.json: {e}")

# Try CSV log
if not trades_loaded and os.path.exists(PAPER_TRADES_CSV):
    try:
        trades_df = pd.read_csv(PAPER_TRADES_CSV)
        col_map = {
            "time": "Time", "entry_time": "Time",
            "direction": "Direction",
            "entry": "Entry",
            "stop": "SL", "sl": "SL",
            "target": "TP", "tp": "TP",
            "status": "Status",
            "pnl": "P&L",
        }
        trades_df = trades_df.rename(columns={k: v for k, v in col_map.items() if k in trades_df.columns})
        display_cols = [c for c in ["Time", "Direction", "Entry", "SL", "TP", "Status", "P&L"] if c in trades_df.columns]
        if display_cols:
            trades_df = trades_df[display_cols]

        try:
            styled = trades_df.style.apply(style_trade_row, axis=1)
            st.dataframe(styled, use_container_width=True)
        except Exception:
            st.dataframe(trades_df, use_container_width=True)
        trades_loaded = True
    except Exception as e:
        st.warning(f"Could not load paper_trades_log.csv: {e}")

if not trades_loaded:
    st.info("📭 No paper trades yet — run `paper_trader.py` during market hours")

# ─── Manual refresh button triggers rerun ───
if refresh_btn:
    # Clear caches so fresh data is fetched
    load_ohlcv.clear()
    fetch_vix_data.clear()
    fetch_macro_trend.clear()
    st.rerun()

# ─── Footer ───
st.markdown("<hr style='border-color:#1e2329;margin-top:24px'>", unsafe_allow_html=True)
st.markdown(
    f"<div style='color:#6e7681;font-size:0.75em;text-align:center'>"
    f"⚡ QQQ Algo · FVG Retest · "
    f"Engine {'🟢' if HAS_ENGINE else '🔴'} · "
    f"{len(df_full):,} bars · "
    f"Updated {now_str}"
    f"</div>",
    unsafe_allow_html=True
)
