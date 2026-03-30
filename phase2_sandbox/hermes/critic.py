#!/usr/bin/env python3
"""
HERMES NODE 2 — CRITIC
Pattern analysis → CriticHypothesis generation.

SCAFFOLD — implement per HERMES_SPEC.md
"""

from dataclasses import dataclass
from typing import Any, List
from .reader import HermesReport, Cohort

# ─── Safety rails — params Hermes CANNOT touch ───────────────────────────────
LOCKED_PARAMS = {
    "DAILY_LOSS_LIMIT",
    "commission_round_trip",
    "etf_slippage_pct",
}
MAX_RISK_PCT = 0.02  # 2% hard ceiling


@dataclass
class CriticHypothesis:
    param_name:     str
    current_value:  Any
    proposed_value: Any
    rationale:      str
    confidence:     float   # 0.0–1.0
    risk_level:     str     # "low" | "medium" | "high"
    affected_cohort: str    # label of the losing cohort being fixed
    expected_wr_improvement: float  # percentage points


def _confidence(sample_size: int, signal_strength: float) -> float:
    """Confidence = capped sample size proxy × signal strength."""
    return min(sample_size / 30, 1.0) * signal_strength


def generate_hypotheses(report: HermesReport) -> List[CriticHypothesis]:
    """
    Analyze losing patterns and generate parameter change proposals.

    Rules:
    - Never touch LOCKED_PARAMS
    - Never propose risk_pct > MAX_RISK_PCT
    - Minimum 15 trades in losing cohort
    - Confidence >= 0.40 to even generate (gate will require >= 0.60)

    TODO: implement pattern matching logic in Week 2 build.
    """
    hypotheses: List[CriticHypothesis] = []

    for cohort in report.losing_patterns:
        if cohort.sample_size < 15:
            continue

        # ── Pattern: PM session has poor WR ──
        if "session=PM" in cohort.label and cohort.win_rate < 0.25:
            conf = _confidence(cohort.sample_size, 1.0 - cohort.win_rate)
            if conf >= 0.40:
                hypotheses.append(CriticHypothesis(
                    param_name="pm_session_enabled",
                    current_value=True,
                    proposed_value=False,
                    rationale=(f"PM session WR = {cohort.win_rate:.0%} on "
                               f"{cohort.sample_size} trades. AM-only may improve expectancy."),
                    confidence=conf,
                    risk_level="medium",
                    affected_cohort=cohort.label,
                    expected_wr_improvement=0.0,  # computed by gater
                ))

        # ── Pattern: specific day-of-week is losing ──
        if "dow=" in cohort.label and cohort.win_rate < 0.20:
            dow = cohort.label.split("=")[1]
            conf = _confidence(cohort.sample_size, 1.0 - cohort.win_rate)
            if conf >= 0.40:
                hypotheses.append(CriticHypothesis(
                    param_name="blackout_days",
                    current_value=[],
                    proposed_value=[dow],
                    rationale=(f"{dow} WR = {cohort.win_rate:.0%} on "
                               f"{cohort.sample_size} trades. Adding to blackout calendar."),
                    confidence=conf,
                    risk_level="low",
                    affected_cohort=cohort.label,
                    expected_wr_improvement=0.0,
                ))

        # ── Pattern: ADX filter too loose ──
        if "zone=neutral" in cohort.label and cohort.win_rate < 0.25:
            conf = _confidence(cohort.sample_size, 0.8)
            hypotheses.append(CriticHypothesis(
                param_name="adx_threshold",
                current_value=18.0,
                proposed_value=22.0,
                rationale=(f"Neutral zone entries (ADX borderline) showing "
                           f"{cohort.win_rate:.0%} WR. Raising ADX threshold "
                           f"from 18 → 22 filters choppy conditions."),
                confidence=conf,
                risk_level="low",
                affected_cohort=cohort.label,
                expected_wr_improvement=0.0,
            ))

    # Deduplicate by param_name (keep highest confidence)
    seen = {}
    for h in hypotheses:
        if h.param_name not in seen or h.confidence > seen[h.param_name].confidence:
            seen[h.param_name] = h

    return list(seen.values())
