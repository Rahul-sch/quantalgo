"""
Backtesting Engine - Runs strategies and tracks performance.
"""
import numpy as np
import pandas as pd
from data_fetcher import is_forex


FOREX_SPREAD_PIPS = 2.5
STOCK_COMMISSION = 0.005  # per share
SLIPPAGE_PCT = 0.0001

# Standard lot = 100,000 units; 1 pip movement on a standard lot =
#   $10 for USD-quoted pairs (GBPUSD, EURUSD, etc.)
#   varies for JPY pairs (~$6.67 at USDJPY 150)
# We simplify: pip_dollar_value = approximate $ per pip per standard lot
FOREX_PIP_DOLLAR = {
    "GBPUSD": 10.0, "EURUSD": 10.0,   # USD is quote currency → $10/pip/lot
    "GBPJPY": 6.67, "EURJPY": 6.67, "USDJPY": 6.67,  # ~$10 / USDJPY rate
}
DEFAULT_PIP_DOLLAR = 10.0  # fallback


def run_backtest(
    df: pd.DataFrame,
    signals: list[dict],
    symbol: str,
    initial_capital: float = 10000.0,
    risk_pct: float = 0.01,
) -> dict:
    """Execute signals against price data and compute metrics."""
    if not signals:
        return _empty_result(symbol, initial_capital)

    forex = is_forex(symbol)
    pip_size = 0.01 if "JPY" in symbol else 0.0001 if forex else 1.0
    spread = FOREX_SPREAD_PIPS * pip_size if forex else 0.0

    # Dollar value per pip per standard lot for this pair
    pip_dollar_value = FOREX_PIP_DOLLAR.get(symbol, DEFAULT_PIP_DOLLAR) if forex else 0.0

    trades = []
    capital = initial_capital
    equity_curve = [initial_capital]

    for sig in signals:
        entry = sig["entry"]
        sl = sig["sl"]
        tp = sig["tp"]
        direction = sig["direction"]
        risk_per_trade = capital * risk_pct

        # Apply spread + slippage
        if direction == "buy":
            entry += spread / 2 + entry * SLIPPAGE_PCT
            sl_dist = abs(entry - sl)
        else:
            entry -= spread / 2 - entry * SLIPPAGE_PCT
            sl_dist = abs(sl - entry)

        if sl_dist <= 0:
            continue

        # Position sizing: ensure forex risks the same $ amount as stocks
        if forex:
            # sl_dist is in price units; convert to pips
            sl_pips = sl_dist / pip_size
            # $ risk per pip needed = risk_per_trade / sl_pips
            dollar_per_pip_needed = risk_per_trade / sl_pips if sl_pips > 0 else 0
            # lot_size in standard lots = dollar_per_pip_needed / pip_dollar_value
            lot_size_lots = dollar_per_pip_needed / pip_dollar_value if pip_dollar_value > 0 else 0
        else:
            # Stocks: lot_size = number of shares
            lot_size = risk_per_trade / sl_dist

        # Simulate trade outcome using future bars
        bar_idx = sig["bar"]
        outcome = _simulate_trade(df, bar_idx, direction, entry, sl, tp)

        if outcome is None:
            continue

        exit_price = outcome["exit_price"]
        if direction == "buy":
            pnl_raw = exit_price - entry
        else:
            pnl_raw = entry - exit_price

        if forex:
            # P&L = pips moved × $ per pip per lot × number of lots
            pnl_pips = pnl_raw / pip_size
            pnl = pnl_pips * pip_dollar_value * lot_size_lots
        else:
            pnl = pnl_raw * lot_size - STOCK_COMMISSION * lot_size * 2
            pnl_pips = pnl_raw

        capital += pnl
        equity_curve.append(capital)

        rr = abs(tp - entry) / sl_dist if sl_dist > 0 else 0

        trades.append({
            "symbol": symbol,
            "strategy": sig["strategy"],
            "direction": direction,
            "entry": entry,
            "exit": exit_price,
            "sl": sl,
            "tp": tp,
            "pnl": round(pnl, 2),
            "pnl_pips": round(pnl_pips, 1) if forex else None,
            "rr_target": round(rr, 2),
            "rr_actual": round(pnl / risk_per_trade, 2) if risk_per_trade > 0 else 0,
            "win": pnl > 0,
            "reason": sig["reason"],
            "bar": bar_idx,
            "outcome": outcome["type"],
        })

    return _compute_metrics(trades, equity_curve, symbol, initial_capital)


def _simulate_trade(df, start_bar, direction, entry, sl, tp, max_bars=50):
    """Simulate a trade forward from entry bar."""
    for i in range(start_bar + 1, min(start_bar + max_bars, len(df))):
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]

        if direction == "buy":
            if low <= sl:
                return {"exit_price": sl, "type": "sl_hit", "bars": i - start_bar}
            if high >= tp:
                return {"exit_price": tp, "type": "tp_hit", "bars": i - start_bar}
        else:
            if high >= sl:
                return {"exit_price": sl, "type": "sl_hit", "bars": i - start_bar}
            if low <= tp:
                return {"exit_price": tp, "type": "tp_hit", "bars": i - start_bar}

    # Timeout: close at last bar
    exit_price = df["Close"].iloc[min(start_bar + max_bars - 1, len(df) - 1)]
    return {"exit_price": exit_price, "type": "timeout", "bars": max_bars}


def _compute_metrics(trades, equity_curve, symbol, initial_capital) -> dict:
    """Compute performance metrics from completed trades."""
    if not trades:
        return _empty_result(symbol, initial_capital)

    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    pnls = [t["pnl"] for t in trades]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t["pnl"]) for t in losses]) if losses else 0
    profit_factor = sum(t["pnl"] for t in wins) / sum(abs(t["pnl"]) for t in losses) if losses and sum(abs(t["pnl"]) for t in losses) > 0 else float("inf")

    # Max drawdown
    peak = initial_capital
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (simplified) — guard against tiny sample sizes
    if len(pnls) > 10 and np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
        sharpe = max(min(sharpe, 100.0), -100.0)  # clamp to prevent absurd values
    else:
        sharpe = 0

    total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100

    return {
        "symbol": symbol,
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "equity_curve": equity_curve,
        "trades": trades,
    }


def _empty_result(symbol: str, initial_capital: float) -> dict:
    return {
        "symbol": symbol,
        "total_trades": 0, "wins": 0, "losses": 0,
        "win_rate": 0, "total_pnl": 0, "total_return_pct": 0,
        "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
        "max_drawdown_pct": 0, "sharpe_ratio": 0,
        "equity_curve": [initial_capital], "trades": [],
    }
