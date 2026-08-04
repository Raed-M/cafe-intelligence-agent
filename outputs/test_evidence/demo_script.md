# Ten-Minute Demo Script

Maps each required demo beat (plan section 31) to an exact command/test in
this repo, so the demo is reproducible rather than narrated.

## 1. Autonomous/scheduled trigger evidence (~1.5 min)

```
pytest tests/integration/test_scheduler.py -q -s
```
Shows an APScheduler `BackgroundScheduler` firing `run_scheduled_cycle` on a
cron trigger with no interactive input, plus the overlap-lock test proving a
second concurrent trigger is skipped and logged rather than double-running.
For a live version: `python scheduler/run.py --profile data/qahwa_saihat/cafe_profile.json --data-dir data/qahwa_saihat --run-once`.

## 2. Source and analyst parallel branches (~2 min)

```
python scripts/export_graph.py
```
Open `outputs/graph.mmd` (paste into a Mermaid renderer) and point at the
`parse_source` fan-out (7 sources, one `Send` per registry entry) and the
`run_one_analyst` fan-out (up to 5 analysts, routed by which cleaned
artifacts are actually available -- `src/graph/analysis_subgraph.py:route_analysts`).
Then: `pytest tests/integration/test_ingestion_and_cleaning.py::test_corrupt_inventory_does_not_stop_other_branches -q -s`
to show one source failing without stopping the other six.

## 3. Critic approval and revision path (~2 min)

```
pytest tests/graph/test_critic_subgraph.py -q -s
```
`test_hallucinated_finding_triggers_targeted_revision_then_rejected_after_cap`
shows: a bad finding from `margin` triggers a revision request; only `margin`
is rerun (call_log asserts `sales` is never touched); after the 2-round cap
the finding is rejected with `total_rejections > 0`. Pair with
`tests/grounding/test_hallucination_injection.py` for the specific injected
defects (wrong period, false metric, item-level cost without BOM).

## 4. Pause at HITL (~1.5 min)

```
python scripts/run_week.py --profile data/qahwa_saihat/cafe_profile.json --data-dir data/qahwa_saihat --target-week 2026-01-05
```
The run prints `Run <id> paused before ('human_gate',).` with the report
path and WhatsApp summary. Then:
```
python scripts/run_week.py --profile data/qahwa_saihat/cafe_profile.json --data-dir data/qahwa_saihat --resume-thread-id <id> --decision approve
```
completes delivery. `--decision reject` demonstrates the stop-without-delivery
path instead.

## 5. Trace one content hook to finding metric and source artifacts (~2 min)

Open the generated `outputs/reports/<run_id>/report.html`: each content idea
lists its `finding_id`; open that finding's card above it, which shows
`code_artifact`/`result_artifact` paths. Open the `result_artifact` JSON
directly to show the exact number the hook/rationale cites. This full chain
is asserted programmatically in
`tests/unit/test_content_validator.py::test_hook_with_number_matching_cited_metric_accepted`
and rejected in the `_not_matching_` counterpart.

## 6. Break one component and show graceful degradation (~1 min)

Pick one live in front of the audience:
- `pytest tests/fault_injection/test_faults.py::test_corrupt_excel_isolated -q -s`
- `pytest tests/fault_injection/test_faults.py::test_tavily_outage_degrades_gracefully -q -s`
- Or literally: `mv data/qahwa_saihat/inventory_weekly.xlsx /tmp/ && python scripts/run_week.py ...` and show the report still generates with inventory findings absent and the data-quality section naming it as failed.

## Bonus: second-cafe genericness (if time allows, ~1 min)

```
git diff --stat <baseline-commit> <second-cafe-commit>
```
(see `outputs/test_evidence/second_cafe_git_diff.txt`) -- shows only
`data/sundown_roasters/*` changed, zero `src/`/`config/` diff, then
`pytest tests/test_second_cafe.py -q -s` proving the report/queries/prayer
times actually reflect Jeddah, not Saihat.
