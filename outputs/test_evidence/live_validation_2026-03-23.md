# Live validation — target week 2026-03-23

Four end-to-end runs driven through the `langgraph dev` server (the same server
LangGraph Studio talks to), via the LangGraph SDK: create thread → stream the
graph → pause at the `human_gate` interrupt → resume with
`{"human_decision": "approve"}` → deliver → persist.

Baseline for comparison is `Successful_run_trace.json`, the last run before the
cross-domain synthesis stage existed.

Per-run artifacts live in `outputs/test_evidence/live_run_2026-03-23_*/`:
`run_trace.json` (every node's output), `final_state.json`,
`state_at_human_gate.json`, `llm_telemetry.jsonl` (one record per provider
request: node, latency, tokens, error).

---

## 1. Headline comparison

| run | LLM calls | failed | tokens | wall | delivered | rejections | cross-domain approved | PDF |
|---|---|---|---|---|---|---|---|---|
| baseline (pre-change) | 78 | **52 (67%)** | 106,121 | — | 3 | 12 | n/a | no |
| run 1 (as implemented) | 44 | 0 | 130,095 | 347s | 5 | 7 | 0 | no |
| run 2 (+abs grounding) | 36 | 0 | 95,013 | 260s | 5 | 3 | 1 | no |
| run 3 (+pdf, +V60) | 56 | 0 | 154,374 | 396s | 5 | 7 | **2** | yes |
| run 4 (+prose, +whatsapp) | 35 | 0 | 101,522 | 253s | 4 | 3 | 0 | yes |

**The single largest change is the failure column.** The baseline lost 52 of 78
calls to `429 RESOURCE_EXHAUSTED`; across all four live runs, 171 calls failed
zero times. That is not a quota-supply difference — it is the removal of the
hot-reload run-killing and stale-run auto-resume behaviour (`--no-reload`, and
clearing `.langgraph_api/` before each run), which previously re-issued every
in-flight call after a worker restart.

Token totals are *not* directly comparable to the baseline: the baseline's
106,121 covers only its 26 *successful* calls, and its 52 failures did real work
before failing. Comparing like with like, a healthy run now costs roughly
95k–154k tokens and 250–400 seconds.

**Run-to-run variance is large** (35 to 56 calls for identical input). The driver
is how many revision rounds fire, which depends on the quality of that run's
analyst output. Treat any single run's numbers as a sample, not a measurement.

---

## 2. The cross-domain stage

**It works.** Verbatim from run 3, both findings approved by the critic:

> **Operational Efficiency and Revenue Contraction** — Revenue declined by
> 23.54% while labour costs increased by 10.90%. This divergence suggests that
> the business is maintaining or increasing staffing levels despite a
> significant drop in sales volume… *(origins: operations + sales)*

> **Revenue Decline and Margin Resilience** — Sales revenue fell by 23.54%
> while the median margin rate remained high at 71.11%. This combination
> suggests that the revenue drop is not being driven by aggressive discounting
> or margin-eroding promotions, but rather by a reduction in transaction
> volume. *(origins: margin + sales)*

These are findings **no single analyst could produce**, each number is grounded
in the artifact of the analyst whose executed code computed it, and they cost
one LLM call and ~3k tokens per run.

The catch-22 is gone. Under the old design a finding citing a neighbouring
domain's number was rejected *for citing it*, and burned its full revision
budget doing so — 6 of the baseline's 12 rejections. Across the four live runs,
zero rejections were of that kind.

### Reliability: 2 of 3 eligible runs produced an approved finding

| run | drafts | outcome |
|---|---|---|
| run 2 | 1 | approved |
| run 3 | 2 | both approved |
| run 4 | 1 | dropped — cited only `sales__rev_pct` + `sales__bask_pct` |

Run 4's failure is the model ignoring rule 1 (two different analysts). It is
**safe** — the deterministic guard dropped it before the critic, so it cost one
call and nothing else, with no revision loop — but it means the feature is not
yet reliable per-run. `prompts/cross_domain.md` rule 1 has since been rewritten
to state the prefix check explicitly and to point at `co_movements` as a
guaranteed-valid source of pairs. **That prompt change is not yet live-validated.**

---

## 3. Defects found and fixed

Every one of these was invisible to offline testing and surfaced only against
real data.

### 3.1 Magnitudes were treated as ungrounded numbers
Run 1's cross-domain finding was **rejected**, caused by my own `__abs`
improvement: the claim read "fell by 23.54%" while the evidence value was
`-23.54`, and the grounding rule matched literally.

Fixed in `_grounded_values`: a value's magnitude is now grounded too. |v| is
provably from the data; whether the *direction word* matches the sign is a
semantic question the LLM critic reviews. Fabrication is still caught in both
signs (test asserts this).

### 3.2 "V60 Filter" — digits inside a product name
The cafe sells a **V60 Filter** (the Hario V60 pour-over). The number regex
extracted `60` from the product name and demanded evidence for it. **No rewrite
could ever satisfy this** — the product is simply called that — so it burned
both revision rounds on 3 findings in run 1 and 2 in run 2.

Fixed with a lookbehind so digits bound into a word are not read as quantities.
Menu items with digits are entirely normal (V60, 1850 blend, No.2 roast).

### 3.3 PDF silently disabled under `langgraph dev`
`pdf_path` was `None` with: *"Blocking call to os.getcwd"*. The dev server
installs **blockbuster**, which raises on Playwright's sync API. Every Studio
run was silently HTML-only.

The patch is **process-wide** — I verified with a minimal graph inside the dev
server that a worker thread does *not* escape it. Rendering now runs in a
subprocess (`src/reporting/pdf_render.py`); path building avoids `os.getcwd`
too. Verified inside the dev server at zero API cost, then live: **90,917 bytes**.

### 3.4 Double negatives in delivered prose
Run 2's actual WhatsApp message read *"Net revenue decreased by **-23.54%**"*.
The `__abs` placeholder form now exists for all analysts, not just cross-domain.
Run 4's message reads *"decreased by **23.54%** … driven by a **23.64%**
reduction"*. The model still types no digits.

### 3.5 The WhatsApp summary went silent
Run 3 shipped a summary with **no findings at all** — header, content-idea
count, link. The earlier "never truncate mid-sentence" fix stopped at the first
line that didn't fit, and run 3's top-ranked finding was a long cross-domain
claim.

Now each finding degrades (full claim → first sentence → title alone) and an
unfittable finding is skipped rather than everything behind it. Run 4: 490
characters, two findings named, top one fully described, nothing clipped.

---

## 4. Critic performance

**The critic is doing real work, and its judgment is mostly sound.** Genuine
catches from these runs:

- *"The claim states that labour hours increased by -4.31%. Mathematically, a
  negative percentage increase is a decrease."* — correct, and exactly the
  semantic layer's job.
- *"The claim states waste cost 'exceeded' the baseline mean, but a value of
  0 SAR is significantly lower than the mean of 998.56"* — a real directional
  error.
- *"The list of star items contains duplicate entries ('CORTADO' and
  'Cortado', 'SPANISH LATTE' and 'سبانيش لاتيه')"* — a genuine data-quality
  problem (unnormalised menu names) the analyst should have handled.
- Duplicate-finding dedup fired correctly in every run.

**Where it wastes budget.** The dominant remaining rejection reason is
`metric 'X' missing period bounds` — analysts omitting `period_start`/
`period_end` on metrics. In runs 3 and 4 this rejected operations and customer
findings *after both revision rounds*, meaning the analyst was told twice and
still didn't comply. This is now the largest single source of wasted calls, and
it is a prompt/schema problem rather than a critic problem: the requirement is
stated but not enforced at the point the analyst writes the JSON.

**One structural asymmetry remains.** A finding rejected on a deterministic rule
gets two revision rounds; if the analyst can't fix it, that's 2 wasted calls per
finding. Cross-domain findings are correctly exempt (non-revisable). Nothing
detects "the analyst produced the identical defect twice" and stops early.

---

## 5. Point-in-time correctness — confirmed leak

You asked earlier whether future data is hidden from the model when running on
an old week. It is **not**, and run 3 shows it concretely. The anomaly analyst
produced:

> Daily revenue on **2026-05-30** was 12005.10 SAR, which is a significant
> deviation from the baseline mean of 6161.19 SAR (z-score: 3.83).

The analysis period is **2026-03-23 → 2026-03-30**. It found an anomaly **61
days in the future** and z-scored it against a correctly-past baseline.

The critic caught it via the period rule and it never reached the report, so no
incorrect output was delivered — but that is a validation-layer backstop, not
prevention, and it cost two revision rounds. Only email parsing is date-filtered
today; the other sources are handed to analysts whole.

**Recommendation:** truncate cleaned artifacts to `date <= analysis_period.end`
before the analyst fan-out. The trailing baselines are all in the past, so this
costs no legitimate capability. I have not made this change — it is a
data-pipeline change beyond the scope of this validation, and it's your call.

---

## 6. Test coverage added

The offline verification previously lived in a scratchpad script. It is now in
the repo:

- `tests/unit/test_cross_domain.py` (17) — pooling, divergent ranking, tautology
  suppression, the cost gate, placeholder grounding, multi-artifact provenance,
  additive-only failure, **critic accepts a multi-analyst finding**, fabrication
  still rejected, never enters the revision loop.
- `tests/unit/test_claim_placeholders.py` (9) — substitution contract, `__abs`,
  NaN handling, unresolved keys.
- `tests/unit/test_whatsapp_summary.py` (8) — the silence bug, graceful
  degradation, budget never exceeded.
- `tests/unit/test_finding_critic.py` (+3) — the V60 case, magnitude grounding,
  fabrication in both signs.

`tests/fault_injection/test_faults.py` was updated: its PDF fault could no
longer be injected in-process once rendering moved to a subprocess.

---

## 7. Assessment

The cross-domain feature is **working and worth keeping**: it produces findings
no single analyst can, at one call and ~3k tokens, with full provenance, and
the critic accepts them. Its reliability is the open item — 2 of 3 eligible runs
— with a safe failure mode and an unvalidated prompt fix in place.

Ranked by remaining value:

1. **Point-in-time truncation** (§5) — a real correctness issue with live proof,
   deterministic to fix.
2. **Period bounds on analyst metrics** (§4) — now the top source of wasted
   revision rounds; enforce at schema level rather than by instruction.
3. **Cross-domain reliability** (§2) — validate the strengthened rule 1 on the
   next run before further tuning.
4. **Menu-name normalisation** — the critic is right that 'CORTADO'/'Cortado'/
   'كورتادو' are one item; fixing it in cleaning would improve margin analysis.
