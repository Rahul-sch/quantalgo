#!/usr/bin/env python3
"""
Telegram Notifier for QQQ Paper Trader
Sends trade alerts and session summary reports via OpenClaw's message API.

Usage (standalone):
    python3 notifier.py --test          # send a test message
    python3 notifier.py --summary       # send AM session summary now

Called automatically by paper_trader.py when signals fire.
"""
import os
import sys
import json
import subprocess
import argparse
from datetime import datetime, date
from zoneinfo import ZoneInfo
from typing import Optional

EST = ZoneInfo("US/Eastern")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TRADES_FILE  = os.path.join(RESULTS_DIR, "paper_trades.json")
DAILY_STATE  = os.path.join(RESULTS_DIR, "paper_daily_state.json")

# ── Telegram delivery via openclaw CLI ──────────────────────────────────────

def send_telegram(message: str) -> bool:
    """Send a Telegram message via the openclaw CLI."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    # Get the Telegram chat ID to send to (Rex's chat)
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "6515146575")
    try:
        result = subprocess.run(
            ["openclaw", "message", "send",
             "-t", chat_id,
             "--channel", "telegram",
             "-m", message],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            return True
        print(f"  [notifier] openclaw send failed: {result.stderr.strip()}")
        return _send_via_requests(message)
    except Exception as e:
        print(f"  [notifier] openclaw CLI error: {e}")
        return _send_via_requests(message)


def _send_via_requests(message: str) -> bool:
    """Fallback: send via Telegram Bot API directly."""
    try:
        import urllib.request
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            print("  [notifier] No TELEGRAM_BOT_TOKEN/CHAT_ID in .env — skipping")
            return False
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = json.dumps({
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }).encode()
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"  [notifier] Telegram API error: {e}")
        return False


# ── Message Formatters ───────────────────────────────────────────────────────

def format_trade_alert(trade: dict, vix: Optional[float] = None,
                        macro_pct: Optional[float] = None) -> str:
    """
    Format a new signal/trade alert message.
    
    Example output:
    🚨 PAPER TRADE ARMED — QQQ
    ─────────────────────────
    Direction:   🟢 LONG
    Entry Limit: $478.50
    Stop Loss:   $476.20  (-$2.30 / 0.48%)
    Take Profit: $484.25  (+$5.75 / 1.20%)
    Risk/Reward: 2.5:1
    Size:        ~42 shares ($100 risk)
    
    🛡️ Regime Status
    VIX:    19.4  ✅ CLEAR
    Macro:  +8.2% above 20M SMA  ✅ BULL
    
    ⏰ 09:34 ET | QQQ 15m
    """
    direction = trade.get("direction", "buy").upper()
    emoji = "🟢 LONG" if direction == "BUY" else "🔴 SHORT"
    entry = trade.get("entry_price", 0)
    sl    = trade.get("stop_loss", 0)
    tp    = trade.get("take_profit", 0)
    rr    = trade.get("rr_ratio", 0)
    risk  = trade.get("risk_amount", 0)
    size  = trade.get("position_size", 0)
    reason = trade.get("reason", "FVG Continuation")

    sl_dist = abs(entry - sl)
    sl_pct  = sl_dist / entry * 100 if entry else 0
    tp_dist = abs(tp - entry)
    tp_pct  = tp_dist / entry * 100 if entry else 0
    sl_sign = "-" if sl < entry else "+"
    tp_sign = "+" if (direction == "BUY" and tp > entry) or (direction == "SELL" and tp < entry) else "-"

    now = datetime.now(EST).strftime("%H:%M ET")

    # Regime block
    regime_lines = []
    if vix is not None:
        vix_status = "✅ CLEAR" if vix < 25 else "🔴 BLOCKED"
        regime_lines.append(f"VIX:    {vix:.1f}  {vix_status}")
    if macro_pct is not None:
        macro_dir = "BULL" if macro_pct >= 0 else "BEAR"
        macro_emoji = "✅" if macro_pct >= 0 else "⚠️"
        regime_lines.append(f"Macro:  {macro_pct:+.1f}% vs 20M SMA  {macro_emoji} {macro_dir}")

    regime_block = ""
    if regime_lines:
        regime_block = "\n🛡️ <b>Regime Status</b>\n" + "\n".join(regime_lines) + "\n"

    msg = (
        f"🚨 <b>PAPER TRADE ARMED — QQQ</b>\n"
        f"─────────────────────────\n"
        f"Direction:   <b>{emoji}</b>\n"
        f"Entry Limit: <b>${entry:.2f}</b>\n"
        f"Stop Loss:   ${sl:.2f}  ({sl_sign}${sl_dist:.2f} / {sl_pct:.2f}%)\n"
        f"Take Profit: ${tp:.2f}  ({tp_sign}${tp_dist:.2f} / {tp_pct:.2f}%)\n"
        f"Risk/Reward: {rr:.1f}:1\n"
        f"Size:        ~{size:.0f} shares (${risk:.0f} risk)\n"
        f"Setup:       {reason}\n"
        f"{regime_block}"
        f"\n⏰ {now} | QQQ 15m"
    )
    return msg


def format_session_summary() -> str:
    """
    Format the 11:45 AM end-of-AM-session summary.
    
    Example output:
    📊 AM SESSION COMPLETE — QQQ
    ─────────────────────────
    Time:     9:30–11:30 ET
    Trades:   3 taken  (2 closed, 1 open)
    Wins:     1W / 1L  (50% WR)
    
    Net P&L (AM):  +$142.50
    Open P&L:      ~+$67.20 (unrealized)
    
    Status: ✅ ACTIVE  |  Daily limit: $150
    """
    now = datetime.now(EST).strftime("%H:%M ET")
    today = date.today().isoformat()

    # Load trades
    trades = []
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE) as f:
                trades = json.load(f)
        except Exception:
            pass

    # Filter to today only
    today_trades = [t for t in trades
                    if t.get("timestamp", "")[:10] == today]

    closed_statuses = ("closed_sl", "closed_tp", "closed_eod", "closed_manual")
    closed = [t for t in today_trades if t.get("status") in closed_statuses]
    open_t = [t for t in today_trades if t.get("status") == "open"]

    total_net = sum(t.get("net_pnl", 0) or 0 for t in closed)
    wins   = [t for t in closed if (t.get("net_pnl") or 0) > 0]
    losses = [t for t in closed if (t.get("net_pnl") or 0) <= 0]
    wr = len(wins) / len(closed) * 100 if closed else 0

    # Unrealized P&L on open trades
    unrealized = 0.0
    open_lines = []
    for t in open_t:
        cp = t.get("current_price") or t.get("entry_price", 0)
        ep = t.get("entry_price", 0)
        sz = t.get("position_size", 0)
        if t.get("direction") == "buy":
            unr = (cp - ep) * sz
        else:
            unr = (ep - cp) * sz
        unrealized += unr
        dir_emoji = "🟢" if t.get("direction") == "buy" else "🔴"
        open_lines.append(f"  {dir_emoji} {t.get('direction','').upper()} @ ${ep:.2f}  ~${unr:+.2f}")

    # Daily state
    halted = False
    if os.path.exists(DAILY_STATE):
        try:
            with open(DAILY_STATE) as f:
                state = json.load(f)
                halted = state.get("halted", False)
        except Exception:
            pass

    status_str = "🛑 HALTED (loss limit hit)" if halted else "✅ ACTIVE"

    # Build closed trade lines
    closed_lines = []
    for t in closed:
        result = "✅" if (t.get("net_pnl") or 0) > 0 else "❌"
        how = t.get("status", "").replace("closed_", "").upper()
        closed_lines.append(
            f"  {result} {t.get('direction','').upper():5} {how:4}  "
            f"${t.get('net_pnl', 0):+.2f}"
        )

    closed_block = "\n".join(closed_lines) if closed_lines else "  (none)"
    open_block   = "\n".join(open_lines)   if open_lines   else "  (none)"

    pnl_emoji = "🟢" if total_net >= 0 else "🔴"
    unr_emoji = "🟡" if open_t else ""

    msg = (
        f"📊 <b>AM SESSION COMPLETE — QQQ</b>\n"
        f"─────────────────────────\n"
        f"Window:    9:30–11:30 ET\n"
        f"Trades:    {len(today_trades)} taken  "
        f"({len(closed)} closed, {len(open_t)} open)\n"
    )

    if closed:
        msg += f"Win Rate:  {len(wins)}W / {len(losses)}L  ({wr:.0f}%)\n"

    msg += f"\n{pnl_emoji} <b>Net P&L (AM):  ${total_net:+.2f}</b>\n"

    if open_t:
        msg += f"{unr_emoji} Open P&L:     ~${unrealized:+.2f} (unrealized)\n"

    if closed_lines:
        msg += f"\n<b>Closed trades:</b>\n{closed_block}\n"

    if open_lines:
        msg += f"\n<b>Open positions:</b>\n{open_block}\n"

    msg += (
        f"\n<b>Status:</b> {status_str}\n"
        f"Daily loss limit: $150\n"
        f"⏰ {now}"
    )
    return msg


def format_daily_limit_alert() -> str:
    """Alert when daily loss limit is hit."""
    now = datetime.now(EST).strftime("%H:%M ET")
    return (
        f"🛑 <b>DAILY LOSS LIMIT HIT — TRADING HALTED</b>\n"
        f"─────────────────────────\n"
        f"The $150 daily loss limit has been reached.\n"
        f"All new signals are blocked for today.\n"
        f"Open positions will be force-closed at 3:30 PM ET.\n"
        f"⏰ {now}"
    )


# ── Convenience functions called by paper_trader.py ─────────────────────────

def alert_new_trade(trade: dict, vix: Optional[float] = None,
                     macro_pct: Optional[float] = None) -> None:
    """Called when a new paper trade is armed."""
    msg = format_trade_alert(trade, vix=vix, macro_pct=macro_pct)
    ok = send_telegram(msg)
    status = "✅ sent" if ok else "⚠️  failed"
    print(f"  [notifier] Trade alert {status}")


def alert_session_summary() -> None:
    """Called at 11:45 AM to summarize the AM session."""
    msg = format_session_summary()
    ok = send_telegram(msg)
    status = "✅ sent" if ok else "⚠️  failed"
    print(f"  [notifier] Session summary {status}")


def alert_daily_limit() -> None:
    """Called when the $150 daily loss limit is hit."""
    msg = format_daily_limit_alert()
    ok = send_telegram(msg)
    print(f"  [notifier] Daily limit alert {'✅ sent' if ok else '⚠️  failed'}")


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QQQ Paper Trader Notifier")
    parser.add_argument("--test",    action="store_true", help="Send test message")
    parser.add_argument("--summary", action="store_true", help="Send AM session summary")
    args = parser.parse_args()

    if args.test:
        now = datetime.now(EST).strftime("%Y-%m-%d %H:%M %Z")
        ok = send_telegram(
            f"🤖 <b>QQQ Paper Trader — System Check</b>\n"
            f"─────────────────────────\n"
            f"✅ Telegram alerts are LIVE\n"
            f"✅ Engine ready for Monday 9:30 AM ET\n"
            f"📊 Monitoring: QQQ 15m AM session\n"
            f"⏰ {now}"
        )
        print("Test message sent ✅" if ok else "Test message FAILED ⚠️")

    elif args.summary:
        alert_session_summary()

    else:
        parser.print_help()
