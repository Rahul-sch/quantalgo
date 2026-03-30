#!/usr/bin/env python3
"""
HERMES NODE 1 — READER
Ingests paper trade history and computes HermesReport.

SCAFFOLD — implement per HERMES_SPEC.md
DO NOT import from phase1 production files.
"""

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

EST = ZoneInfo("US/Eastern")
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
TRADES_CSV  = os.path.join(BASE_DIR, "results", "paper_trades.csv")
FOREX_JSON  = os.path.join(BASE_DIR, "results", "forex_closed.json")
MIN_TRADES  = 20   # minimum before Hermes runs


# ─── Data Model ──────────────────────────────────────────────────────────────

@dataclass
class Cohort:
    """A slice of trades sharing a common attribute (e.g. direction=buy, zone=discount)."""
    label:        str
    trades:       List[Dict]
    win_rate:     float
    avg_win:      float
    avg_loss:     float
    expectancy:   float
    sample_size:  int

    @classmethod
    def from_trades(cls, label: str, trades: List[Dict]) -> "Cohort":
        if not trades:
            return cls(label, [], 0.0, 0.0, 0.0, 0.0, 0)
        wins   = [t for t in trades if t.get("outcome") == "win" or
                  (t.get("net_pnl") or 0) > 0]
        losses = [t for t in trades if t not in wins]
        wr     = len(wins) / len(trades)
        avg_w  = np.mean([t.get("net_pnl", 0) for t in wins])   if wins   else 0.0
        avg_l  = abs(np.mean([t.get("net_pnl", 0) for t in losses])) if losses else 0.0
        exp    = (wr * avg_w) - ((1 - wr) * avg_l)
        return cls(label, trades, round(wr, 4), round(avg_w, 2),
                   round(avg_l, 2), round(exp, 2), len(trades))


@dataclass
class HermesReport:
    """Full statistical report on paper trading performance."""
    generated_at:   str
    total_trades:   int
    win_rate:       float
    expectancy:     float
    sharpe_approx:  float
    max_consec_loss: int

    # Cohort breakdown
    by_direction:   Dict[str, Cohort] = field(default_factory=dict)
    by_session:     Dict[str, Cohort] = field(default_factory=dict)
    by_zone:        Dict[str, Cohort] = field(default_factory=dict)
    by_dow:         Dict[str, Cohort] = field(default_factory=dict)  # day-of-week
    by_model:       Dict[str, Cohort] = field(default_factory=dict)

    # Losing patterns (cohorts with WR < 30% and n >= 15)
    losing_patterns: List[Cohort] = field(default_factory=list)

    # Raw
    all_trades:     List[Dict] = field(default_factory=list)
    pnl_series:     List[float] = field(default_factory=list)

    # Trigger flags
    should_run:     bool = False
    trigger_reason: str  = ""


# ─── Loader ──────────────────────────────────────────────────────────────────

def load_equity_trades() -> List[Dict]:
    """Load equity paper trades from CSV."""
    if not os.path.exists(TRADES_CSV):
        return []
    trades = []
    with open(TRADES_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize types
            for float_field in ("entry_price", "exit_price", "net_pnl", "gross_pnl",
                                 "sl_distance", "rr_ratio", "atr_at_entry"):
                if row.get(float_field):
                    try:
                        row[float_field] = float(row[float_field])
                    except ValueError:
                        row[float_field] = None
            trades.append(row)
    return trades


def load_forex_trades() -> List[Dict]:
    """Load closed forex trades from JSON."""
    if not os.path.exists(FOREX_JSON):
        return []
    try:
        with open(FOREX_JSON, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def load_all_trades() -> List[Dict]:
    """Merge equity + forex closed trades, normalize schema."""
    equity = load_equity_trades()
    forex  = load_forex_trades()

    # Tag source
    for t in equity:
        t.setdefault("source", "equity")
    for t in forex:
        t.setdefault("source", "forex")
        # Normalize outcome field
        if t.get("status") == "closed_tp":
            t["outcome"] = "win"
        elif t.get("status") == "closed_sl":
            t["outcome"] = "loss"
        # Add session tag based on hour
        try:
            ts = datetime.fromisoformat(t.get("entry_time", t.get("timestamp", "")))
            ts_est = ts.astimezone(EST)
            hour = ts_est.hour
            t["session"] = "AM" if 9 <= hour < 12 else "PM" if 13 <= hour < 16 else "OTHER"
            t["dow"] = ts_est.strftime("%A")
        except Exception:
            t["session"] = "UNKNOWN"
            t["dow"]     = "UNKNOWN"

    return equity + forex


# ─── Analysis ────────────────────────────────────────────────────────────────

def compute_consec_losses(trades: List[Dict]) -> int:
    """Maximum consecutive losing streak."""
    max_streak = cur = 0
    for t in sorted(trades, key=lambda x: x.get("timestamp", "")):
        pnl = t.get("net_pnl", 0) or 0
        if pnl < 0:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0
    return max_streak


def compute_sharpe(pnl_series: List[float]) -> float:
    """Approximate Sharpe ratio: mean(pnl) / std(pnl)."""
    if len(pnl_series) < 2:
        return 0.0
    arr = np.array(pnl_series)
    std = np.std(arr)
    return float(np.mean(arr) / std) if std > 0 else 0.0


def build_cohorts(trades: List[Dict], key: str) -> Dict[str, Cohort]:
    """Group trades by a key field and build a Cohort for each group."""
    groups: Dict[str, List[Dict]] = {}
    for t in trades:
        val = str(t.get(key, "unknown"))
        groups.setdefault(val, []).append(t)
    return {k: Cohort.from_trades(f"{key}={k}", v) for k, v in groups.items()}


def identify_losing_patterns(cohorts_dict: Dict[str, Dict[str, Cohort]]) -> List[Cohort]:
    """Find cohorts with WR < 30% and n >= 15 — these are Hermes targets."""
    losing = []
    for group_name, cohorts in cohorts_dict.items():
        for label, cohort in cohorts.items():
            if cohort.sample_size >= 15 and cohort.win_rate < 0.30:
                losing.append(cohort)
    # Sort by worst expectancy first
    return sorted(losing, key=lambda c: c.expectancy)


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def generate_report() -> HermesReport:
    """
    Full reader pipeline. Call this from hermes.py.
    Returns HermesReport with trigger flag set if Hermes should run.
    """
    trades = load_all_trades()
    closed = [t for t in trades if t.get("outcome") in ("win", "loss")]

    if not closed:
        return HermesReport(
            generated_at=datetime.now(EST).isoformat(),
            total_trades=0,
            win_rate=0.0,
            expectancy=0.0,
            sharpe_approx=0.0,
            max_consec_loss=0,
            should_run=False,
            trigger_reason="No closed trades yet",
        )

    pnl_series = [t.get("net_pnl", 0) or 0 for t in closed]
    wins       = [t for t in closed if (t.get("net_pnl", 0) or 0) > 0]
    wr         = len(wins) / len(closed)
    avg_w      = np.mean([t["net_pnl"] for t in wins]) if wins else 0
    losses_t   = [t for t in closed if t not in wins]
    avg_l      = abs(np.mean([t["net_pnl"] for t in losses_t])) if losses_t else 0
    expectancy = (wr * avg_w) - ((1 - wr) * avg_l)
    consec     = compute_consec_losses(closed)
    sharpe     = compute_sharpe(pnl_series)

    cohort_groups = {
        "direction": build_cohorts(closed, "direction"),
        "session":   build_cohorts(closed, "session"),
        "zone":      build_cohorts(closed, "zone"),
        "dow":       build_cohorts(closed, "dow"),
        "model":     build_cohorts(closed, "model"),
    }
    losing_patterns = identify_losing_patterns(cohort_groups)

    # Trigger logic
    should_run    = False
    trigger_reason = ""
    if len(closed) >= MIN_TRADES:
        should_run    = True
        trigger_reason = f"Sample size {len(closed)} >= {MIN_TRADES}"
    if consec >= 4:
        should_run    = True
        trigger_reason = f"Early warning: {consec} consecutive losses"

    return HermesReport(
        generated_at=datetime.now(EST).isoformat(),
        total_trades=len(closed),
        win_rate=round(wr, 4),
        expectancy=round(expectancy, 2),
        sharpe_approx=round(sharpe, 3),
        max_consec_loss=consec,
        by_direction=cohort_groups["direction"],
        by_session=cohort_groups["session"],
        by_zone=cohort_groups["zone"],
        by_dow=cohort_groups["dow"],
        by_model=cohort_groups["model"],
        losing_patterns=losing_patterns,
        all_trades=closed,
        pnl_series=pnl_series,
        should_run=should_run,
        trigger_reason=trigger_reason,
    )


if __name__ == "__main__":
    report = generate_report()
    print(f"\n  HERMES READER — {report.generated_at}")
    print(f"  Total closed trades: {report.total_trades}")
    print(f"  Win rate:    {report.win_rate:.1%}")
    print(f"  Expectancy:  ${report.expectancy:.2f}")
    print(f"  Sharpe:      {report.sharpe_approx:.2f}")
    print(f"  Max consec loss: {report.max_consec_loss}")
    print(f"  Should run: {report.should_run} — {report.trigger_reason}")
    if report.losing_patterns:
        print(f"\n  LOSING PATTERNS ({len(report.losing_patterns)}):")
        for p in report.losing_patterns:
            print(f"    {p.label}: {p.win_rate:.0%} WR | n={p.sample_size} | E=${p.expectancy:.2f}")
