# Cafe Intelligence Agent

A generic, multi-agent LangGraph system that ingests a cafe's weekly operational
data, runs five parallel specialist analysts with a critic/revision loop,
generates grounded bilingual social content, and produces a bilingual
HTML/WhatsApp report with a human approve/edit/reject gate before delivery.

Implements all 8 phases of `implementation_plan_final.md`: foundation,
ingestion/cleaning, analysts/critic, context/content, report/HITL/persistence,
cross-run memory + autonomous scheduling, a full testing/hardening pass (10
weekly cycles, 5 hand-verified metrics, hallucination/grounding tests, fault
injection), and a second-cafe generic-onboarding proof. See that document for
the full architecture contract, formulas, and prompt pack, and
`outputs/test_evidence/` for the evidence artifacts referenced below.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # or .venv/bin/activate on macOS/Linux
pip install -r requirements.lock   # or: pip install -e . (see pyproject.toml)
cp .env.example .env               # then fill in real keys
```

Required env vars (see `.env.example`):
- `OPENAI_API_KEY` — model calls for analysts/critic/content/content
  validator/email extraction/(optional) report summary compression.
  (`LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` also supported; see
  `src/tools/llm_factory.py`.)
- `ANALYST_MODEL` / `CRITIC_MODEL` / `CONTENT_MODEL` / `CONTENT_VALIDATOR_MODEL`
  / `REPORT_SUMMARY_MODEL` — per-agent model names (plan section 16 prompt
  pack); the critic and content-validator LLM passes are additive to their
  deterministic gates and degrade to deterministic-only if unset/unavailable,
  the report-summary pass stays off unless `report.use_llm_summary_compression`
  is also set to `true` in `config/app_settings.yaml`.
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

## Running from LangGraph Studio (`langgraph dev`)

```bash
pip install -e ".[dev]"     # installs langgraph-cli[inmem]
langgraph dev --no-reload   # --no-reload is REQUIRED, see below
```

Opens a Studio UI (`smith.langchain.com/studio/?baseUrl=...`) that can drive
the real graph (`src/graph/main_graph.py:graph`) directly -- start a new
thread with `{"target_week": "2026-03-16"}` (or nothing, for the latest week);
`profile_path`/`data_dir` default to `qahwa_saihat`. `resolve_config`
(`src/graph/setup_nodes.py`) is the only node whose behavior differs from a
CLI/scheduler-triggered run: it resolves the real `RuntimeCafeConfig` from
those plain fields when one isn't already in state.

**`--no-reload` is not optional.** `langgraph dev` defaults to hot-reloading
on file changes (like `uvicorn --reload`), and this pipeline writes artifact
files (generated analyst code, results, reports) under `outputs/` as core,
unavoidable behavior of every run -- the very first file an analyst writes
mid-run triggers a reload, which kills the in-flight run. This was confirmed
directly: writing one file under `outputs/artifacts/...` while `langgraph dev`
was running (no `--no-reload`) produced `WatchFiles detected changes in
'outputs\...'. Reloading...` in the server log immediately, and the same
write with `--no-reload` produced no reload event at all. Symptom without the
flag: every analyst call gets `asyncio.exceptions.CancelledError` shortly
after the 5-way fan-out starts, the run silently retries from its last
checkpoint (re-spending every LLM call in that step) instead of completing,
and Ctrl+C may not cleanly stop the server (the reload supervisor adds an
extra process layer that Windows' console signal handling doesn't always
propagate through). There is no per-path ignore/exclude option in
`langgraph_cli` today -- `--no-reload` is the only fix.

**Stale runs auto-resume across server restarts.** `langgraph dev`'s local
persistence (`.langgraph_api/`, gitignored) is explicitly dev/test-only, not
durable production state, and it resumes any pending run automatically on
the *next* server start -- including one you thought you'd already stopped.
If a run needs to be abandoned, cancel/delete it via the API first, or delete
`.langgraph_api/` before restarting, so simply starting the server again for
any reason (even just to poke at it) can't silently re-spend real API calls
on a stale run.

As defense in depth against both of the above (and against ordinary heavy
concurrent use), `get_chat_model()` wraps Gemini calls in a circuit breaker
(`src/tools/llm_factory.py::_CircuitBreakerChatModel`): once a call confirms
the provider's quota is exhausted, every subsequent call fails instantly --
no network request -- for `LLM_QUOTA_COOLDOWN_SECONDS` (default 1800s),
instead of letting an external auto-retry (or another `run_one_analyst`
sibling) re-spend a call that's guaranteed to fail the same way.

## Autonomous scheduling

```bash
python scheduler/run.py --profile data/qahwa_saihat/cafe_profile.json --data-dir data/qahwa_saihat            # weekly cron loop
python scheduler/run.py --profile data/qahwa_saihat/cafe_profile.json --data-dir data/qahwa_saihat --run-once  # fire immediately
```

Uses the cafe's own timezone, `misfire_grace_time=3600`, `coalesce=True`,
`max_instances=1`, plus a file-based `RunLock` (`src/persistence/run_lock.py`)
so an overlapping trigger is skipped and logged rather than double-running.

## Onboarding a second cafe

Supply a different `cafe_profile.json` and data directory — no application
code changes required:

```bash
python scripts/run_week.py --profile data/sundown_roasters/cafe_profile.json --data-dir data/sundown_roasters
```

`data/sundown_roasters/` is a synthetic second-cafe fixture (Jeddah, different
menu/coordinates/handles) generated by `scripts/generate_fixture_second_cafe.py`.
`tests/test_second_cafe.py` proves the report/local-search queries/prayer
times actually reflect the new profile, and
`outputs/test_evidence/second_cafe_git_diff.txt` shows the onboarding commit
touches only `data/sundown_roasters/*` — zero `src/`/`config/` changes.

## Tests

```bash
pytest tests/ -q          # ~8 minutes; spawns many real subprocesses (analyst code execution)
```

82 tests, all passing, covering everything above plus:
- **Cross-run memory**: `tests/unit/test_trends.py` (consecutive-decline
  streak detection, content repetition notes) and
  `tests/integration/test_restart_resume.py` (a *second, independent*
  compiled graph instance resumes a HITL-paused run from the on-disk SQLite
  checkpoint, and a fresh `MemoryStore` reads the prior run's findings).
- **Scheduling**: `tests/integration/test_scheduler.py` (autonomous cron
  firing with no interactive input; overlapping-trigger lock skip).
- **Five hand-verified metrics**: `tests/integration/test_ground_truth.py`
  (`scripts/verify_ground_truth.py` independently recomputes net revenue,
  valid transaction count, conversion rate, one SKU's gross profit, and known
  waste cost directly from the raw files with its own dedup logic, then
  diffs against the pipeline's own output — see
  `outputs/test_evidence/ground_truth_verification.csv`).
- **Ten weekly cycles**: `tests/test_weekly_cycles.py` runs the full graph
  across all ten plan-specified weeks (normal, Ramadan, Eid, launch, sensor
  outage, summer) and records an honest status table —
  `outputs/test_evidence/ten_weekly_cycles.md`.
- **Hallucination/grounding**: `tests/grounding/test_hallucination_injection.py`
  — false numbers, wrong periods, BOM-less item costs, rejected-finding
  citations, nonexistent context IDs, and closed-time recommendations are all
  rejected; critic rejection count is asserted non-zero (M27).
- **Fault injection**: `tests/fault_injection/test_faults.py` — corrupt
  Excel, empty reviews, Tavily outage, code syntax error, code timeout,
  missing optional source, cost/step cap abort, PDF renderer failure.
- **Second-cafe proof**: `tests/test_second_cafe.py` +
  `outputs/test_evidence/second_cafe_git_diff.txt`.

See `outputs/test_evidence/clean_clone.md` for a full from-scratch clone →
venv → install → preflight → test → run → HITL-approve → resume → graph-export
transcript, `outputs/test_evidence/production_failure_cases.md` for the three
specific "what breaks this in production" cases, and
`outputs/test_evidence/demo_script.md` for a reproducible ten-minute demo
script mapped to exact commands/tests.

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
  provenance verification (load-bearing) plus an optional constrained-LLM
  semantic pass (`prompts/critic.md`) for nuance a regex cannot judge, targeted
  revision routing (max 2 rounds), ranking capped at 5 findings.
- `src/context/` — Hijri calendar (Ramadan/Eid), deterministic prayer times,
  Tavily local search with graceful degradation, posting-window derivation.
- `src/content/content_agent.py` + `src/validation/content_validator.py` —
  exactly 3 bilingual ideas, independently re-validated (second layer:
  deterministic structural checks plus an optional constrained-LLM
  Arabic/English semantic-alignment check, `prompts/content_validator.md`).
- `src/reporting/` — deterministic Jinja2 HTML (Arabic RTL + English), chart
  rendering, WhatsApp-length summary (optional constrained-LLM compression of
  the already-approved text per `prompts/report_summary.md`, disabled by
  default per plan section 16.12's stated preference for the deterministic
  template), best-effort PDF.
- `src/persistence/` — SQLite checkpointer (HITL pause/resume) + separate
  long-term memory DB (cross-run history, idempotent delivery receipts).
- `src/graph/main_graph.py` — assembles all of the above into one
  `StateGraph` matching the plan's section 6.1 diagram.

## Known limitations

- The email-fact and content-idea/analyst-code LLM call sites fall back to
  deterministic extraction or degrade to a "failed"/"unavailable" status when
  no usable API key is configured. This environment's `OPENAI_API_KEY` was
  not in a directly usable form during development (see
  `outputs/test_evidence/clean_clone.md` for what that failure actually looks
  like end-to-end), so live analyst/content LLM behavior was verified via
  injected fake generators in tests (`tests/integration/test_full_pipeline.py`)
  rather than a real model call. Tavily live search *was* exercised live and
  works (see `src/context/context_builder.py`). Supply a real
  `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` + `LLM_PROVIDER=anthropic`) to get
  live analyst/content generation.
- `RuntimeCafeConfig` and related config objects are stored directly in
  LangGraph checkpointed state, which triggers a msgpack deprecation warning
  ("Deserializing unregistered type..."). Functionally harmless today; see
  the "Known follow-up" note in `outputs/test_evidence/clean_clone.md` for
  the recommended fix before a long-lived production deployment.
- The 10-weekly-cycle harness and second-cafe fixture run with Tavily
  disabled for reproducibility (recorded, not live, per M18A); one live
  Tavily call is proven separately in `test_full_pipeline.py`.
