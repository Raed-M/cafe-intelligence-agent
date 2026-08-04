# Cafe Intelligence Agent

A generic, multi-agent LangGraph system that ingests a cafe's weekly operational
data, runs five parallel specialist analysts with a critic/revision loop,
generates grounded bilingual social content, and produces a bilingual
HTML/WhatsApp report with a human approve/edit/reject gate before delivery.

Implements `implementation_plan_final.md` phases 0–5 (foundation through
report/HITL/persistence). See that document for the full architecture
contract, formulas, and prompt pack.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # or .venv/bin/activate on macOS/Linux
pip install -r requirements.lock   # or: pip install -e . (see pyproject.toml)
cp .env.example .env               # then fill in real keys
```

Required env vars (see `.env.example`):
- `OPENAI_API_KEY` — model calls for analysts/critic/content/email extraction.
  (`LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` also supported; see
  `src/tools/llm_factory.py`.)
- `TAVILY_API_KEY` — optional; local-context search degrades gracefully without it.
- `LANGCHAIN_API_KEY` / `LANGCHAIN_TRACING_V2=true` — optional LangSmith tracing.

## Run

```bash
python scripts/preflight.py --profile data/qahwa_saihat/cafe_profile.json --data-dir data/qahwa_saihat
python scripts/run_week.py --profile data/qahwa_saihat/cafe_profile.json --data-dir data/qahwa_saihat --target-week 2026-01-05
# graph pauses before delivery (HITL):
python scripts/run_week.py --profile data/qahwa_saihat/cafe_profile.json --data-dir data/qahwa_saihat --resume-thread-id <run_id> --decision approve
python scripts/export_graph.py   # writes outputs/graph.mmd
```

## Onboarding a second cafe

Supply a different `cafe_profile.json` and data directory — no application
code changes required:

```bash
python scripts/run_week.py --profile data/other_cafe/cafe_profile.json --data-dir data/other_cafe
```

## Tests

```bash
pytest tests/ -q
```

47 tests cover: period/business-date logic (cross-midnight Ramadan hours),
custom reducers, raw profile validation, all 7 parsers against the actual
supplied schemas, ingestion fault-isolation, POS double-swipe dedup (verified
against the documented ~1% rate), dead-sensor rediscovery (June 8–10), the
restricted code executor's security/timeout/policy boundaries, the analyst
self-correction repair loop, the finding critic (deterministic provenance
checks + hallucination rejection), the targeted-revision graph loop with
round cap, the finding ranker, prayer-time calculation, the content validator
(9 grounding/closed-time/stock-risk/duplicate-hook cases), and a full
end-to-end graph run (ingestion → cleaning → 5 analysts → critic → live
Tavily context → content → validation → report → HITL pause → approve →
deliver → persist).

## Architecture

See `implementation_plan_final.md` section 6 for the full mermaid diagram and
rationale. Key modules:

- `src/config/` — raw profile (unmodified supplied contract) + runtime config
  resolution + source registry + preflight.
- `src/parsers/` — one parser per source, registry-driven `Send` fan-out
  (`src/graph/ingestion_subgraph.py`), independent failure per source.
- `src/cleaning/` — POS dedup/refunds/business-date, dead-sensor detection,
  item-name repair via menu join, data-quality summary.
- `src/analysts/` + `src/tools/code_executor.py` — 5 parallel specialist
  analysts that generate and execute real Python in a restricted subprocess,
  self-correcting on failure (`src/graph/analysis_subgraph.py`).
- `src/validation/finding_critic.py` + `finding_ranker.py` — deterministic
  provenance verification, targeted revision routing (max 2 rounds), ranking
  capped at 5 findings.
- `src/context/` — Hijri calendar (Ramadan/Eid), deterministic prayer times,
  Tavily local search with graceful degradation, posting-window derivation.
- `src/content/content_agent.py` + `src/validation/content_validator.py` —
  exactly 3 bilingual ideas, independently re-validated (second layer).
- `src/reporting/` — deterministic Jinja2 HTML (Arabic RTL + English), chart
  rendering, WhatsApp-length summary, best-effort PDF.
- `src/persistence/` — SQLite checkpointer (HITL pause/resume) + separate
  long-term memory DB (cross-run history, idempotent delivery receipts).
- `src/graph/main_graph.py` — assembles all of the above into one
  `StateGraph` matching the plan's section 6.1 diagram.

## Known limitations / what's not yet built

Per the phased scope agreed for this pass (plan phases 0–5 of 8):

- **Phase 6 (scheduling)**: `APScheduler` wiring for autonomous weekly
  triggers is not yet implemented; runs are currently triggered via
  `scripts/run_week.py`.
- **Phase 7 (hardening)**: the 10-weekly-cycle harness, 5 hand-verified
  metrics notebook, and full fault-injection suite (corrupt Excel, Tavily
  outage, cost-cap-mid-revision, etc.) beyond what's covered in
  `tests/integration` are not yet built.
- **Phase 8 (generic proof/demo)**: a second-cafe fixture dataset and the
  formal clean-clone evidence log (`outputs/test_evidence/clean_clone.md`)
  are not yet produced, though the architecture itself is already
  profile/data-dir generic (see `src/config/runtime_config.py`) and every
  location/handle/timezone value is profile-derived, never hardcoded.
- The email-fact and content-idea LLM call sites fall back to deterministic
  extraction/skip when no usable API key is configured; this environment's
  `OPENAI_API_KEY` was not in a directly usable form during development, so
  live end-to-end LLM behavior was verified via `src/tools/llm_factory.py`'s
  provider-detection path and via injected fake generators in tests
  (`tests/integration/test_full_pipeline.py`) rather than a live model call
  for analysts/content. Supply a real `OPENAI_API_KEY` (or
  `ANTHROPIC_API_KEY` + `LLM_PROVIDER=anthropic`) to exercise the live path.
