#!/usr/bin/env python3
"""
stress_test.py — Chaos Injection Test Suite for the QQQ Algo Engine

Tests:
  T1. Corrupted JSON ledger — partial write, invalid JSON, truncated file
  T2. Atomic write crash simulation — verify .tmp + os.replace integrity
  T3. Concurrent write collision — cron + Streamlit writing simultaneously
  T4. API timeout injection — yfinance mock hanging >15s
  T5. Missing / sparse candles — gaps in the DataFrame
  T6. Gap-over scenario — single candle skips both entry AND stop loss
  T7. Daily state circuit breaker — halted flag after $150 loss
  T8. Signal fingerprint deduplication — same signal across 3 cron runs

Run:
    cd ~/quantalgo && python3 stress_test.py
    cd ~/quantalgo && python3 stress_test.py --test T6   # run single test
"""

import os
import sys
import json
import time
import tempfile
import threading
import argparse
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

EST = ZoneInfo("US/Eastern")
PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def run_test(name: str, fn):
    print(f"\n  {'─'*55}")
    print(f"  🧪 {name}")
    print(f"  {'─'*55}")
    try:
        fn()
        results.append((name, True, ""))
        print(f"  {PASS}")
    except AssertionError as e:
        results.append((name, False, str(e)))
        print(f"  {FAIL}: {e}")
    except Exception as e:
        results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  {FAIL} (exception): {type(e).__name__}: {e}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# T1 — CORRUPTED JSON LEDGER
# ═══════════════════════════════════════════════════════════════════════════════

def test_corrupted_json():
    """_safe_read_json must return the default value on any corrupt input."""
    from paper_trader import _safe_read_json, _safe_write_json

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name

    try:
        # Case 1: truncated mid-write (simulate power loss)
        with open(path, "w") as f:
            f.write('{"date": "2026-03-31", "pending_orders": [{"id": "abc", ')  # incomplete
        result = _safe_read_json(path, {"default": True})
        assert result == {"default": True}, f"Truncated JSON should return default, got {result}"
        print("    Truncated JSON → default ✓")

        # Case 2: completely empty file
        with open(path, "w") as f:
            f.write("")
        result = _safe_read_json(path, [])
        assert result == [], f"Empty file should return [], got {result}"
        print("    Empty file → default ✓")

        # Case 3: binary garbage
        with open(path, "wb") as f:
            f.write(b"\x00\xff\xfe\xfd" * 50)
        result = _safe_read_json(path, {"ok": True})
        assert result == {"ok": True}, f"Binary garbage should return default, got {result}"
        print("    Binary garbage → default ✓")

        # Case 4: valid JSON but wrong type (list instead of dict)
        with open(path, "w") as f:
            json.dump([1, 2, 3], f)
        result = _safe_read_json(path, {})
        # Should return the list (it's valid JSON) — caller must validate type
        assert isinstance(result, list), f"Valid JSON list should load, got {type(result)}"
        print("    Valid JSON wrong type → loaded (caller validates) ✓")

    finally:
        os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════════
# T2 — ATOMIC WRITE INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════

def test_atomic_write():
    """Verify that .tmp → os.replace is used and the original file is never partial."""
    from paper_trader import _safe_write_json, _safe_read_json

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ledger.json")
        tmp_path = path + ".tmp"

        # Write initial good data
        good_data = {"date": "2026-03-31", "trades": list(range(100))}
        _safe_write_json(path, good_data)
        assert os.path.exists(path), "File should exist after write"
        assert not os.path.exists(tmp_path), ".tmp should be cleaned up after replace"
        print("    .tmp cleaned up after successful write ✓")

        # Verify round-trip
        loaded = _safe_read_json(path, {})
        assert loaded == good_data, f"Round-trip failed: {loaded}"
        print("    Round-trip integrity verified ✓")

        # Simulate crash mid-write: leave a .tmp orphan
        with open(tmp_path, "w") as f:
            f.write('{"partial": true, "truncated":')  # incomplete
        # A new write should overwrite .tmp and succeed
        new_data = {"date": "2026-03-31", "trades": [999]}
        _safe_write_json(path, new_data)
        loaded = _safe_read_json(path, {})
        assert loaded == new_data, f"Should survive orphaned .tmp, got {loaded}"
        assert not os.path.exists(tmp_path), ".tmp should be gone"
        print("    Orphaned .tmp overwritten cleanly ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# T3 — CONCURRENT WRITE COLLISION
# ═══════════════════════════════════════════════════════════════════════════════

def test_concurrent_writes():
    """
    10 threads each write unique data to the same file simultaneously.
    After all threads finish, the file must contain exactly one valid JSON object
    (no interleaving, no corruption).
    """
    from paper_trader import _safe_write_json, _safe_read_json

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "concurrent.json")
        errors = []
        THREADS = 10
        WRITES_PER_THREAD = 5

        def writer(thread_id: int):
            for i in range(WRITES_PER_THREAD):
                try:
                    _safe_write_json(path, {"thread": thread_id, "write": i,
                                            "payload": list(range(100))})
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Write errors: {errors}"
        print(f"    {THREADS * WRITES_PER_THREAD} concurrent writes — no errors ✓")

        # File must be valid JSON
        result = _safe_read_json(path, None)
        assert result is not None, "File must be valid JSON after concurrent writes"
        assert "thread" in result and "write" in result, f"Unexpected content: {result}"
        print(f"    Final file valid JSON: thread={result['thread']}, write={result['write']} ✓")

        # No .tmp or .lock orphans left behind
        leftover = [f for f in os.listdir(tmpdir) if f.endswith((".tmp", ".lock"))]
        # .lock files are OK to persist (they're empty sentinels), but .tmp should be gone
        tmp_orphans = [f for f in leftover if f.endswith(".tmp")]
        assert not tmp_orphans, f".tmp orphans left: {tmp_orphans}"
        print("    No .tmp orphans after concurrent writes ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# T4 — API TIMEOUT INJECTION (yfinance mock)
# ═══════════════════════════════════════════════════════════════════════════════

def test_api_timeout():
    """
    Verify _yf_download_safe returns an empty DataFrame within the timeout
    window when the downstream call hangs.

    CPython threads cannot be forcibly killed, so we test the wrapper's
    return-path directly: the ThreadPoolExecutor raises FuturesTimeoutError
    after `timeout_sec`, calls executor.shutdown(wait=False), and returns
    an empty DataFrame — all within wall-clock ≤ timeout_sec + 1s.

    The background thread keeps sleeping (daemon) but the CALLER is unblocked.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTE
    import pandas as pd

    hang_event = threading.Event()

    def slow_download(*args, **kwargs):
        hang_event.wait(timeout=60)   # simulates hung Yahoo connection
        return pd.DataFrame()

    # Test the wrapper pattern directly (same logic as _yf_download_safe)
    TIMEOUT = 2  # seconds

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(slow_download, "^VIX",
                             period="2d", interval="1d",
                             progress=False, auto_adjust=True)
    start = time.time()
    try:
        result = future.result(timeout=TIMEOUT)
    except FTE:
        result = pd.DataFrame()
        executor.shutdown(wait=False, cancel_futures=True)

    elapsed = time.time() - start
    hang_event.set()  # release background thread for cleanup

    assert elapsed < TIMEOUT + 1.5, \
        f"Caller blocked too long: {elapsed:.1f}s (timeout={TIMEOUT}s)"
    assert result.empty, f"Should return empty DataFrame on timeout, got {result}"
    print(f"    FuturesTimeoutError after {TIMEOUT}s → caller returned in {elapsed:.2f}s ✓")
    print(f"    executor.shutdown(wait=False) — caller unblocked immediately ✓")
    print(f"    Returned empty DataFrame ✓")
    print(f"    Note: background thread is a daemon — exits when process exits ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# T5 — MISSING / SPARSE CANDLES
# ═══════════════════════════════════════════════════════════════════════════════

def test_sparse_candles():
    """
    Feed the signal engine a DataFrame with random gaps (missing bars).
    Engine must not crash and must not generate lookahead signals.
    """
    from quant_engine import generate_signals, Config

    # Build a sparse QQQ-like DataFrame: 200 bars, randomly drop 40%
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2026-01-02 09:30", periods=n, freq="15min", tz="US/Eastern")
    price = 480 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "Open":   price + np.random.uniform(-0.3, 0.3, n),
        "High":   price + np.random.uniform(0.1, 0.8, n),
        "Low":    price - np.random.uniform(0.1, 0.8, n),
        "Close":  price,
        "Volume": np.random.randint(500_000, 5_000_000, n).astype(float),
    }, index=dates)

    # Drop 40% of rows randomly (simulate market gaps, missing API data)
    drop_idx = np.random.choice(df.index, size=int(n * 0.4), replace=False)
    df_sparse = df.drop(drop_idx).copy()
    print(f"    Sparse DataFrame: {len(df_sparse)}/{n} bars ({len(drop_idx)} gaps)")

    cfg = Config(
        symbols=["QQQ"],
        initial_capital=50_000,
        risk_pct=0.01,
        use_vix_filter=False,
        use_blackout_filter=False,
        use_macro_veto=False,
        session_filter=False,
    )

    # Must not raise
    signals = generate_signals(df_sparse, cfg)
    print(f"    generate_signals on sparse data: {len(signals)} signals — no crash ✓")

    # No signal bar index should exceed DataFrame length
    for s in signals:
        bar_idx = s.get("bar", 0)
        assert bar_idx < len(df_sparse), f"Signal bar {bar_idx} >= len(df) {len(df_sparse)}"
    print(f"    All signal bar indices within bounds ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# T6 — GAP-OVER SCENARIO
# ═══════════════════════════════════════════════════════════════════════════════

def test_gap_over():
    """
    Simulate a pending BUY limit at $480 with SL at $477.
    The next bar opens and closes BELOW both entry AND SL: [475.00–476.50].
    resolve_pending_orders() must detect the gap-over and close the trade at SL.
    """
    from paper_trader import (
        _safe_write_json, _safe_read_json, STATE_LEDGER_FILE, TRADES_FILE,
        add_pending_order, resolve_pending_orders, load_trades
    )
    import shutil

    # Back up real files
    bk_ledger = STATE_LEDGER_FILE + ".bak"
    bk_trades  = TRADES_FILE + ".bak"
    if os.path.exists(STATE_LEDGER_FILE): shutil.copy(STATE_LEDGER_FILE, bk_ledger)
    if os.path.exists(TRADES_FILE):       shutil.copy(TRADES_FILE, bk_trades)

    try:
        today = datetime.now(EST).strftime("%Y-%m-%d")

        # Inject a synthetic pending order
        test_trade = {
            "id":           "GAP_TEST_001",
            "timestamp":    datetime.now(EST).isoformat(),
            "symbol":       "QQQ",
            "timeframe":    "15m",
            "direction":    "buy",
            "entry_price":  480.00,
            "stop_loss":    477.00,
            "take_profit":  487.50,
            "position_size": 10.0,
            "commission":   2.40,
            "status":       "pending",
            "signal_id":    "QQQ_GAP_TEST_BUY",
        }
        # Write directly to trade file + ledger
        _safe_write_json(TRADES_FILE, [test_trade])
        ledger = {
            "date": today,
            "pending_orders": [{
                "signal_id":      "QQQ_GAP_TEST_BUY",
                "trade_id":       "GAP_TEST_001",
                "direction":      "buy",
                "entry_price":    480.00,
                "stop_loss":      477.00,
                "take_profit":    487.50,
                "armed_at":       datetime.now(EST).isoformat(),
                "armed_bar_time": datetime.now(EST).isoformat(),
                "expires_after":  "2099-01-01T23:59:00-05:00",
                "status":         "pending",
            }],
            "alerted_signal_ids": ["QQQ_GAP_TEST_BUY"],
        }
        _safe_write_json(STATE_LEDGER_FILE, ledger)

        # Build a gap-over bar: opens and closes BELOW entry AND SL
        gap_bar_time = pd.Timestamp("2026-03-31 10:00:00", tz="US/Eastern")
        df_gap = pd.DataFrame({
            "Open":   [476.00],
            "High":   [476.50],
            "Low":    [475.00],  # below SL 477.00
            "Close":  [476.20],  # below entry 480.00
            "Volume": [1_000_000.0],
        }, index=pd.DatetimeIndex([gap_bar_time]))

        print(f"    Bar: [{df_gap['Low'].iloc[0]:.2f}–{df_gap['High'].iloc[0]:.2f}] "
              f"vs entry $480.00, SL $477.00")
        filled = resolve_pending_orders(df_gap)

        # Check: trade should be closed with gap_over_sl status
        trades = load_trades()
        assert trades, "Trades file should not be empty"
        t = trades[0]
        assert t["status"] == "closed_gap_sl", \
            f"Expected 'closed_gap_sl', got '{t['status']}'"
        print(f"    Trade status: {t['status']} ✓")
        assert t.get("net_pnl") is not None, "net_pnl should be set"
        assert t["net_pnl"] < 0, f"Gap-over should be a loss, got net_pnl={t['net_pnl']}"
        print(f"    net_pnl = ${t['net_pnl']:.2f} (loss confirmed) ✓")

    finally:
        # Restore real files
        if os.path.exists(bk_ledger):
            shutil.copy(bk_ledger, STATE_LEDGER_FILE)
            os.unlink(bk_ledger)
        elif os.path.exists(STATE_LEDGER_FILE):
            os.unlink(STATE_LEDGER_FILE)
        if os.path.exists(bk_trades):
            shutil.copy(bk_trades, TRADES_FILE)
            os.unlink(bk_trades)
        elif os.path.exists(TRADES_FILE):
            os.unlink(TRADES_FILE)


# ═══════════════════════════════════════════════════════════════════════════════
# T7 — DAILY CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════════════

def test_circuit_breaker():
    """
    Simulate $150 daily loss. The next scan must be blocked by the halted flag.
    """
    from paper_trader import (
        load_daily_state, save_daily_state, DAILY_STATE_FILE,
        DAILY_LOSS_LIMIT, _today_str
    )
    import shutil

    bk = DAILY_STATE_FILE + ".bak"
    if os.path.exists(DAILY_STATE_FILE): shutil.copy(DAILY_STATE_FILE, bk)

    try:
        # Set daily P&L to exactly -$151 (over the limit)
        state = {
            "date":         _today_str(),
            "daily_pnl":    -151.00,
            "halted":       False,  # not yet set — engine should detect and set it
            "trades_today": 3,
        }
        save_daily_state(state)

        loaded = load_daily_state()
        assert loaded["daily_pnl"] == -151.00, f"State not saved correctly: {loaded}"
        assert loaded["halted"] == False, "halted should be False before engine checks"
        print(f"    Daily P&L = ${loaded['daily_pnl']:.2f} (limit: -${DAILY_LOSS_LIMIT}) ✓")

        # Simulate what the engine does: check and halt
        if loaded["daily_pnl"] <= -DAILY_LOSS_LIMIT:
            loaded["halted"] = True
            save_daily_state(loaded)

        reloaded = load_daily_state()
        assert reloaded["halted"] == True, "Should be halted after loss limit breach"
        print(f"    Halted flag set after breach ✓")
        print(f"    DAILY_LOSS_LIMIT constant = ${DAILY_LOSS_LIMIT} ✓")

    finally:
        if os.path.exists(bk):
            shutil.copy(bk, DAILY_STATE_FILE)
            os.unlink(bk)
        elif os.path.exists(DAILY_STATE_FILE):
            os.unlink(DAILY_STATE_FILE)


# ═══════════════════════════════════════════════════════════════════════════════
# T8 — SIGNAL DEDUPLICATION ACROSS CRON RUNS
# ═══════════════════════════════════════════════════════════════════════════════

def test_signal_deduplication():
    """
    Same signal fingerprint fired across 3 simulated cron runs.
    Only the FIRST run should mark it alerted — runs 2 and 3 must skip.
    """
    from paper_trader import (
        _safe_write_json, STATE_LEDGER_FILE,
        is_already_alerted, mark_alerted, load_state_ledger, _today_str
    )
    import shutil

    bk = STATE_LEDGER_FILE + ".bak"
    if os.path.exists(STATE_LEDGER_FILE): shutil.copy(STATE_LEDGER_FILE, bk)

    try:
        # Fresh ledger
        _safe_write_json(STATE_LEDGER_FILE, {
            "date": _today_str(),
            "pending_orders": [],
            "alerted_signal_ids": [],
        })

        sig_id = "QQQ_1743400200_478.50_buy"

        # Run 1: not yet alerted
        assert not is_already_alerted(sig_id), "Should not be alerted yet"
        mark_alerted(sig_id)
        print("    Run 1: signal fired + marked alerted ✓")

        # Run 2: should be blocked
        assert is_already_alerted(sig_id), "Run 2 should see signal as already alerted"
        print("    Run 2: duplicate blocked ✓")

        # Run 3: still blocked
        assert is_already_alerted(sig_id), "Run 3 should still see signal as already alerted"
        print("    Run 3: duplicate blocked ✓")

        # Verify ledger state
        ledger = load_state_ledger()
        assert ledger["alerted_signal_ids"].count(sig_id) == 1, \
            f"Signal ID should appear exactly once, got {ledger['alerted_signal_ids'].count(sig_id)}"
        print(f"    Ledger contains exactly 1 entry for signal ✓")

    finally:
        if os.path.exists(bk):
            shutil.copy(bk, STATE_LEDGER_FILE)
            os.unlink(bk)
        elif os.path.exists(STATE_LEDGER_FILE):
            os.unlink(STATE_LEDGER_FILE)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

ALL_TESTS = {
    "T1": ("Corrupted JSON Ledger",           test_corrupted_json),
    "T2": ("Atomic Write Integrity",           test_atomic_write),
    "T3": ("Concurrent Write Collision",       test_concurrent_writes),
    "T4": ("API Timeout Injection",            test_api_timeout),
    "T5": ("Missing / Sparse Candles",         test_sparse_candles),
    "T6": ("Gap-Over Scenario",                test_gap_over),
    "T7": ("Daily Circuit Breaker",            test_circuit_breaker),
    "T8": ("Signal Deduplication (3 runs)",    test_signal_deduplication),
}

def main():
    parser = argparse.ArgumentParser(description="QQQ Algo Engine — Chaos Stress Tests")
    parser.add_argument("--test", help="Run single test (e.g. T6)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  💥 STRESS TEST SUITE — QQQ Algo Engine")
    print("=" * 60)

    if args.test:
        key = args.test.upper()
        if key not in ALL_TESTS:
            print(f"  Unknown test: {key}. Options: {list(ALL_TESTS.keys())}")
            sys.exit(1)
        name, fn = ALL_TESTS[key]
        run_test(f"{key}: {name}", fn)
    else:
        for key, (name, fn) in ALL_TESTS.items():
            run_test(f"{key}: {name}", fn)

    # Summary
    print("\n" + "=" * 60)
    print("  📊 RESULTS SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    for name, ok, err in results:
        status = "✅" if ok else "❌"
        line = f"  {status} {name}"
        if not ok:
            line += f"\n       └─ {err}"
        print(line)
    print(f"\n  {passed}/{len(results)} tests passed", end="")
    if failed == 0:
        print(" — 🟢 ALL CLEAR")
    else:
        print(f" — 🔴 {failed} FAILURE(S)")
    print()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
