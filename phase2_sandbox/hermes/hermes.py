#!/usr/bin/env python3
"""
HERMES — Orchestrator
Runs the 4-node loop: Reader → Critic → Writer → Gater

SCAFFOLD — wire nodes together in Week 3 build.

Usage:
    python3 hermes.py          # run if trigger conditions met
    python3 hermes.py --force  # force run regardless of trade count
    python3 hermes.py --report # print HermesReport only, no proposals
"""

import argparse
import sys
import os
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes.reader import generate_report, MIN_TRADES

EST = ZoneInfo("US/Eastern")


def main():
    parser = argparse.ArgumentParser(description="Hermes Self-Improving Critic")
    parser.add_argument("--force",  action="store_true", help="Run even if below min trades")
    parser.add_argument("--report", action="store_true", help="Print report only, no proposals")
    args = parser.parse_args()

    now = datetime.now(EST)
    print(f"\n  {'═'*50}")
    print(f"  🧠 HERMES CRITIC — {now.strftime('%I:%M %p ET · %a %b %d')}")
    print(f"  {'═'*50}")

    # Node 1: Reader
    print("\n  [1/4] READER — ingesting trade history...")
    report = generate_report()

    print(f"        Trades loaded:  {report.total_trades}")
    print(f"        Win rate:       {report.win_rate:.1%}")
    print(f"        Expectancy:     ${report.expectancy:.2f}")
    print(f"        Consec losses:  {report.max_consec_loss}")
    print(f"        Trigger:        {report.should_run} — {report.trigger_reason}")

    if not report.should_run and not args.force:
        print(f"\n  ⏸  Hermes dormant — need {MIN_TRADES} closed trades "
              f"(have {report.total_trades}). Use --force to override.")
        return

    if args.report:
        print("\n  [report-only mode — no proposals generated]")
        return

    # Node 2: Critic
    print("\n  [2/4] CRITIC — analyzing patterns...")
    # TODO: from hermes.critic import generate_hypotheses
    # hypotheses = generate_hypotheses(report)
    print("        [SCAFFOLD] Critic not yet wired — build in Week 2")

    # Node 3: Writer
    print("\n  [3/4] WRITER — generating patch files...")
    print("        [SCAFFOLD] Writer not yet wired — build in Week 2")

    # Node 4: Gater
    print("\n  [4/4] GATER — running stress tests + PR gate...")
    print("        [SCAFFOLD] Gater not yet wired — build in Week 3")

    print(f"\n  Hermes cycle complete.")


if __name__ == "__main__":
    main()
