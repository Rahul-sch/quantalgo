# HERMES — Self-Improving Critic Agent
## Technical Specification v0.1
### Phase 2 Sandbox · `phase2_sandbox/hermes/`

---

## 1. What Hermes Is

Hermes is a **closed-loop agentic system** that reads the live paper trading record,
identifies structural weaknesses in the signal logic, generates candidate parameter
patches, stress-tests them against historical data, and opens a GitHub PR for human
review — all without manual intervention.

It does NOT auto-merge. It proposes. Humans approve.

---

## 2. Architecture: The 4-Node Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                        HERMES LOOP                              │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  READER  │───▶│  CRITIC  │───▶│  WRITER  │───▶│  GATER   │  │
│  │          │    │          │    │          │    │          │  │
│  │ Reads    │    │ Identifies│   │ Generates│    │ Runs     │  │
│  │ paper_   │    │ losing   │    │ patched  │    │ stress   │  │
│  │ trades   │    │ patterns │    │ param    │    │ test +   │  │
│  │ .csv +   │    │ via stat │    │ config   │    │ backtest │  │
│  │ ledger   │    │ analysis │    │ + test   │    │ gates;   │  │
│  │          │    │          │    │ script   │    │ opens PR │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       ▲                                               │         │
│       └───────────── weekly feedback loop ────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Node Specs

### Node 1 — READER (`hermes/reader.py`)
**Input:** `results/paper_trades.csv`, `results/forex_closed.json`
**Output:** `HermesReport` dataclass

Computes:
- Win rate by: session (AM/PM), direction (buy/sell), zone, day-of-week
- Losing pattern fingerprints: "buy + premium zone + Monday = 0% WR"
- Consecutive loss runs and their market context (VIX, macro, RVOL)
- Expectancy: `(WR × avg_win) - ((1-WR) × avg_loss)`
- Sharpe-equivalent: expectancy / std(pnl)

Triggers Hermes run when:
- `len(closed_trades) >= 20` (minimum sample)
- OR `consecutive_losses >= 4` (early warning)
- OR weekly cron fires (Sunday 6 PM ET)

### Node 2 — CRITIC (`hermes/critic.py`)
**Input:** `HermesReport`
**Output:** `List[CriticHypothesis]`

Each hypothesis has:
```python
@dataclass
class CriticHypothesis:
    param_name: str          # e.g. "vix_threshold"
    current_value: Any       # e.g. 25.0
    proposed_value: Any      # e.g. 20.0
    rationale: str           # e.g. "AM trades above VIX 22 have 18% WR"
    confidence: float        # 0.0–1.0 based on sample size
    risk_level: str          # "low" | "medium" | "high"
```

Hard rules:
- NEVER propose changes to: `DAILY_LOSS_LIMIT`, `commission_round_trip`
- NEVER propose `risk_pct > 0.02` (2% max)
- Minimum 15 trades in the losing cohort before proposing a fix
- Confidence = `min(sample_size/30, 1.0) * signal_strength`

### Node 3 — WRITER (`hermes/writer.py`)
**Input:** `List[CriticHypothesis]`
**Output:** patch file + backtest script

For each hypothesis:
1. Writes a candidate `config_patch.py` (overrides only, imports from production)
2. Generates `stress_test_hypothesis_{n}.py` that runs the 60-day backtest
   with the patched params vs baseline and outputs a comparison table
3. Writes a human-readable `PROPOSAL.md` explaining the change in plain English

Uses the **MCP pattern** (see Section 5): each generated script is a
self-contained tool call that Hermes can invoke and parse the output of.

### Node 4 — GATER (`hermes/gater.py`)
**Input:** stress test results
**Output:** GitHub PR or rejection log

Gate criteria (ALL must pass):
- [ ] Proposed params improve win rate by ≥ 3% on backtest holdout
- [ ] Max drawdown does not increase by > 15%
- [ ] Net P&L improves or stays neutral
- [ ] Stress test passes all 8 chaos tests (T1–T8 from `stress_test.py`)
- [ ] `confidence >= 0.60`

If gates pass → opens GitHub PR via `gh pr create` with:
- Title: `[Hermes] {param_name}: {current} → {proposed} (+{wr_improvement}% WR)`
- Body: full `PROPOSAL.md` + comparison table
- Label: `hermes-proposal`, `needs-review`
- Branch: `hermes/{param_name}-{timestamp}`

If gates fail → writes rejection to `memory/hermes-rejected.md` with reason

---

## 4. MCP Integration (Model Context Protocol)

### What MCP enables for Hermes

MCP is a standardized protocol for AI agents to invoke external tools and
receive structured results. For Hermes, this means:

**Instead of:** hardcoding backtest calls as subprocess shells
**We use:** MCP tool definitions that Hermes invokes as first-class actions

```python
# hermes/mcp_tools.py — tool definitions Hermes can call
HERMES_TOOLS = [
    {
        "name": "run_backtest",
        "description": "Run forex_backtester with specific params, return metrics dict",
        "input_schema": {
            "pair": "string",
            "min_rr": "float",
            "max_sl_pips": "int",
            "vix_threshold": "float",
        }
    },
    {
        "name": "read_trade_history",
        "description": "Read paper_trades.csv and forex_closed.json, return stats",
        "input_schema": {
            "last_n_trades": "int",
            "filter_by": "string",  # "buy"|"sell"|"AM"|"PM"|"all"
        }
    },
    {
        "name": "open_github_pr",
        "description": "Create a PR with proposed param changes via gh CLI",
        "input_schema": {
            "branch_name": "string",
            "title": "string",
            "body_md": "string",
            "files": "List[{path, content}]",
        }
    },
    {
        "name": "run_stress_test",
        "description": "Run stress_test.py against a candidate config",
        "input_schema": {
            "config_patch": "dict",
        }
    },
]
```

**Practical implementation path (no MCP server required initially):**
Use the `anthropic` Python SDK with `tools=HERMES_TOOLS` and implement
each tool as a local Python function. MCP server is Phase 2.5+ when we
need multi-agent orchestration across machines.

### Long-Context Caching for Hermes

Claude's prompt caching means Hermes can hold the entire 60-day trade
history + backtest code in context across its multi-step reasoning loop
without re-paying tokens on each node transition:

```python
# Cache the static context (code + historical data) once
# Pay only for the incremental critic/writer/gater reasoning
messages = [
    {"role": "user", "content": [
        {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": backtest_code_str, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": trade_history_str},  # fresh each run
    ]}
]
```

This cuts Hermes token cost by ~60-80% per cycle because the backtest
engine code (largest context block) is cached between node calls.

---

## 5. Autonomous Code Review Pattern

Hermes uses a **3-pass review** before proposing any PR:

```
Pass 1 — SAFETY CHECK
  - Does the patch touch DAILY_LOSS_LIMIT? → REJECT
  - Does the patch increase risk_pct above 2%? → REJECT
  - Does the patch disable any fail-safe (VIX, macro)? → REJECT

Pass 2 — STATISTICAL VALIDITY
  - Is sample size >= 15 trades in the affected cohort? → PASS
  - Is the improvement > noise (p-value proxy: confidence >= 0.6)? → PASS

Pass 3 — BACKTEST GATE
  - Run baseline vs patched on holdout window (last 20% of data)
  - All 8 stress tests must pass
  - Drawdown cannot worsen by > 15%
```

Only patches that clear all 3 passes reach GitHub.

---

## 6. Build Order

```
Week 1:  hermes/reader.py        — trade ingestion + HermesReport
Week 1:  hermes/mcp_tools.py     — tool definitions + local implementations
Week 2:  hermes/critic.py        — pattern analysis + hypothesis generation
Week 2:  hermes/writer.py        — patch file + stress test script generation
Week 3:  hermes/gater.py         — gate logic + gh PR integration
Week 3:  hermes/hermes.py        — orchestrator that runs the 4-node loop
Week 4:  Cron: Sunday 6 PM ET    — weekly autonomous Hermes run
```

---

## 7. Activation Criteria (don't build until)

- [ ] 20+ closed paper trades in `paper_trades.csv`
- [ ] At least 1 full week of dual-session data (AM + PM)
- [ ] Forex manager has logged at least 5 closed forex trades

**Reason:** Hermes needs real data to be useful. Running it on synthetic
backtest data trains it to optimize for the past, not for live edge.

---

*Spec Owner: Principal Quant Architect*
*Status: APPROVED FOR SANDBOX BUILD*
*Phase 1 production files: READ-ONLY*
