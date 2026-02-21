# Cardiac Agent Post-Processing — Developer Guide

## Project Structure

```
cardiac_agent_postproc/
├── agents/                  # Multi-agent system
│   ├── base_agent.py        # BaseAgent ABC with LLM reasoning + memory
│   ├── message_bus.py       # AgentMessage, CaseContext, MessageBus
│   ├── coordinator.py       # CoordinatorAgent — orchestrates the pipeline
│   ├── triage_agent.py      # TriageAgent — fast quality classification
│   ├── diagnosis_agent.py   # DiagnosisAgent — spatial defect ID
│   ├── planner_agent.py     # PlannerAgent — repair operation sequencing
│   ├── executor.py          # ExecutorAgent — applies ops, monitors RQS
│   ├── verifier_agent.py    # VerifierAgent — quality gating
│   └── atlas_builder.py     # AtlasBuilderAgent — shape atlas (legacy)
├── orchestrator.py          # Pipeline entry — sets up bus + agents
├── ops.py                   # 20+ morphological operations (tool library)
├── rqs.py                   # Reward Quality Score
├── vlm_guardrail.py         # VLM overlay generation + scoring
├── quality_model.py         # Trained quality classifier
├── api_client.py            # OpenAI-compatible LLM/VLM client
├── settings.py              # .env config loader
├── io_utils.py, geom.py, view_utils.py, eval_metrics.py
└── run.py                   # CLI entry point
```

## Build & Run

```bash
pip install -e .
python -m cardiac_agent_postproc.run --config config/default.yaml
```

## Architecture

Each agent inherits from `BaseAgent` and has:
- **LLM reasoning** via `think()` / `think_json()` / `think_vision()`
- **Memory** — rolling conversation history for statefulness
- **MessageBus** — inter-agent communication (send/receive/broadcast)

Pipeline per case: **Triage → Diagnosis → Plan → Execute → Verify**, with dynamic re-routing on rejection (up to `max_rounds_per_case` rounds).

## Key Principles
1. **No-GT optimization** — use RQS (proxy metrics) to optimize labels without ground truth
2. **Agent autonomy** — each agent reasons independently before communicating results
3. **Conservative gating** — VerifierAgent rejects fixes if unsure (preserves original)

## Coding Style
- Python ≥ 3.10, type hints everywhere
- `black` + `isort` for formatting
- All agents register with `MessageBus` and communicate via `AgentMessage`

## Workflow Convention
- After every completed task, append a new session entry to `WORKLOG.md`.
- Minimum required fields per entry: `Current goal`, `Done`, `Blocked`, `Next command`, `Key files`, `Notes`.
- If no blocker exists, explicitly write `Blocked: None` to keep logs consistent.
