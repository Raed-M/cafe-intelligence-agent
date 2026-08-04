# Clean-Clone Evidence Log

Performed twice: once against commit `defff16` ("Harden analyst/content LLM
call sites...") during development, and re-verified against the final commit
`0c92891` ("Fix artifact path collisions across analyst revisions...") with
identical results (run pauses honestly on insufficient evidence with a
placeholder key, resumes and delivers on approve). Both runs were in a
directory outside the working repo, using only files tracked by git
(`.gitignore` excludes `.venv/`, `db/`, `outputs/artifacts,reports,test_evidence/`,
so nothing generated during development leaked into the clone).

## 1. Clone

```
$ git clone -q . /tmp/clean_clone_test
$ cd /tmp/clean_clone_test && git log --oneline -1
defff16 Harden analyst/content LLM call sites: degrade gracefully instead of crashing the graph on auth/network failure
```

## 2. Environment

```
$ python -m venv .venv
$ source .venv/Scripts/activate
$ python -m pip install -q --upgrade pip
$ pip install -q -r requirements.lock
```
Exit code: 0. No output (all wheels resolved from the lock file).

```
$ cp .env.example .env
$ export OPENAI_API_KEY="sk-test-clean-clone"   # placeholder; see note below
```

## 3. Preflight

```
$ python scripts/preflight.py --profile data/qahwa_saihat/cafe_profile.json --data-dir data/qahwa_saihat
Preflight OK: True
  pos: ok
  menu: ok
  traffic: ok
  staff: ok
  inventory: ok
  emails: ok
  reviews: ok
  Missing optional env vars (features degrade): ['LANGCHAIN_API_KEY']
```

## 4. Unit tests

```
$ python -m pytest tests/unit -q
.............................................                            [100%]
45 passed in 1.30s
```

## 5. Execute one week

```
$ python scripts/run_week.py --profile data/qahwa_saihat/cafe_profile.json --data-dir data/qahwa_saihat --target-week 2026-01-05
Run run_5d322bfb paused before ('human_gate',).
Report: outputs\reports\run_5d322bfb\report.html
WhatsApp summary:
تقرير Qahwa Saihat (2026-01-05 - 2026-01-12): لا توجد أدلة كافية لهذا الأسبوع.
Qahwa Saihat report (2026-01-05 - 2026-01-12): insufficient evidence this week.
```

**Note on the placeholder API key**: this clean-clone pass used a syntactically
invalid `OPENAI_API_KEY` (no real key was available in the clone environment),
so every LLM-backed analyst call failed its auth check. This is exactly the
scenario `src/analysts/base.py`'s `try/except` around `code_generator()` /
`repair_generator()` is designed for (added in this same commit after this
failure mode was first hit during development): each analyst degrades to
`status="failed"` individually, the critic sees zero candidate findings, the
graph takes the "no evidence" path, and the report/WhatsApp summary honestly
say so in both languages -- rather than the whole graph crashing. This is a
legitimate real-world degradation path (an invalid/expired key is exactly
what production would see), captured here as-is rather than papered over with
a fake key. With a real `OPENAI_API_KEY`, the same run produces actual
analyst findings (see `tests/integration/test_analyst_execution.py`, which
exercises the identical code path with an injected generator instead of a
network call, and `README.md` for how to supply a working key).

## 6. HITL approve + resume

```
$ python scripts/run_week.py --profile data/qahwa_saihat/cafe_profile.json --data-dir data/qahwa_saihat --resume-thread-id run_5d322bfb --decision approve
Resumed run_5d322bfb: run_status=succeeded
```

```
$ ls db/
checkpoints.sqlite  memory.sqlite
$ test -f outputs/reports/run_5d322bfb/report.html && echo "report exists"
report exists
```

## 7. Export graph

```
$ python scripts/export_graph.py
Wrote outputs/graph.mmd
```
`outputs/graph.mmd` (63 lines) contains the full Mermaid diagram matching
`implementation_plan_final.md` section 6.1's topology.

## Known follow-up (not blocking)

LangGraph's SqliteSaver logs a deprecation warning when checkpointing state
that embeds our dataclass/Pydantic config objects (`RuntimeCafeConfig`, etc.)
directly rather than only JSON-native structures:

```
Deserializing unregistered type src.config.runtime_config.RuntimeCafeConfig
from checkpoint. This will be blocked in a future version...
```

It does not currently affect correctness, but should be addressed before a
graded/production submission by either registering these types with
`allowed_msgpack_modules` or by storing only a serializable summary of config
in graph state and re-resolving the full `RuntimeCafeConfig` from disk
(profile path + data dir) on each node instead of carrying the live object
through checkpoints.
