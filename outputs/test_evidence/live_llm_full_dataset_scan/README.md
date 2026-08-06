# Live-LLM Full Dataset Scan

Real (non-mocked) run of the analyst + critic pipeline against Claude Haiku 4.5,
across every complete week in `data/qahwa_saihat` (2026-01-05 through
2026-07-27, 29 weeks). Ingestion/cleaning ran once (deterministic, week-
independent); each week then ran the 5 live analysts (code-gen + execution +
repair) and the critic (deterministic + live semantic pass).

Saved here specifically so later sessions can inspect/reuse these results
without re-spending tokens on live model calls.

## Files

- `full_dataset_scan_results.json` — one entry per week: `week_start`,
  `week_end`, `candidate_findings` (everything the 5 analysts produced, before
  the critic), `critic_results` (approved/rejected/revision_requests/notes),
  `final_findings` (critic-approved, ranked, capped at 5). This is the
  structured source of truth — read this first.
- `full_dataset_scan.log` — human-readable run log, same content as the JSON
  but flattened to text, in the order weeks actually completed. Useful for a
  quick skim or `grep`.
- `scan_script_used.py` — the exact script that produced this data (copied
  from the scratchpad it ran in), in case it needs to be re-run or extended
  for additional weeks/sources later.
- `artifacts/scan_<run_id>*/` — the actual code + result.json the analysts
  generated and executed for every week/analyst/attempt (e.g.
  `artifacts/scan_c76afc5d_w05/code/margin/initial/1.py` and the matching
  `results/margin/initial/1/result.json`). This is the full provenance chain
  behind every finding's `result_artifact` reference — the same artifact tree
  the critic itself re-reads to verify a finding's numbers actually resolve.

## Known caveats in this data

- The `anomaly` analyst has a live bug: some of its findings render a
  malformed date like `1970-01-01 00:00:00.000000017` instead of the real
  timestamp (a nanosecond-vs-datetime unit-conversion bug in its own
  generated code, not in the pipeline). Treat anomaly findings' stated dates
  with suspicion until that's fixed; the underlying deviation/magnitude
  numbers are still real.
- `operations` findings are sparser than the other analysts across the early
  weeks -- not yet root-caused as of this scan; may just reflect genuinely
  fewer operations-relevant findings in those weeks, or may be a separate gap
  worth checking.
- This run reflects the pipeline *after* fixing: the `ANALYST_INPUTS_JSON`
  prompt ambiguity, missing column-name context, an overly narrow code-exec
  import allowlist (`typing`/`pathlib`/etc.), and a mixed-datetime-format bug
  in cleaned `pos.timestamp`. Weeks 1-3 in this file were re-run after those
  fixes landed, so every week here reflects the same, current pipeline
  behavior (no stale pre-fix data mixed in).

## How to extend later without full re-runs

`scan_script_used.py` is idempotent per week and writes incrementally --
`FIRST_MONDAY`/`LAST_MONDAY` can be narrowed to just the weeks not yet covered
and pointed at a fresh `OUT_PATH`, then merged with this file, instead of
re-running all 29 weeks.
