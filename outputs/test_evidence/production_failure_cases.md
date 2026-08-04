# What Breaks This in Production

Three specific, concrete failure modes (plan section 31.1) -- not "the API
might be down."

## 1. POS vendor changes the export schema or transaction semantics

**What breaks**: `pos_parser.py` hard-requires the 11 documented columns
(`transaction_id, timestamp, sku, item_name, quantity, unit_price_sar,
discount_sar, line_total_sar, payment_method, channel, cashier_id`) and
`pos_dedup.py`'s double-swipe signature is built from a specific column set
and semantics (`quantity < 0` == refund; `transaction_id` groups a basket).
If the POS vendor renames a column, changes refunds to a separate `is_refund`
flag instead of negative quantity, or starts reusing `transaction_id` across
days, the parser will either raise (missing column -> caught, source marked
`failed`, other six branches continue -- the *safe* failure) or, worse,
**silently miscompute** if the new schema happens to still validate (e.g. a
renamed `qty` column that isn't checked because `REQUIRED_COLUMNS` still
lists the old name and pandas doesn't error on an extra column) -- in that
case revenue/dedup math would be silently wrong. **Mitigation actually
present**: schema pinning at `REQUIRED_COLUMNS` plus `run_preflight()` catches
a full column-set mismatch before the graph runs. **Mitigation NOT present**:
no schema *version* field or checksum on the incoming file to detect a
same-column-count-but-different-semantics change (e.g. a silent unit change
from SAR to halalas). Production fix: add a schema/version stamp check and a
sanity-bound alert (e.g. "average line_total_sar outside historical
[min,max]") before trusting a new week's numbers.

## 2. Ingredient prices change but recipe/BOM data is absent

**What breaks**: `finding_critic.py` already refuses to let the margin
analyst claim an exact per-drink cost impact from a supplier price change
without a recipe/BOM (`_ITEM_LEVEL_COST_PATTERNS` + assumption-label check),
and the margin prompt (`prompts/analysts/margin.md`) explicitly prohibits
"assuming litres/grams per drink." This is handled correctly *for the
findings pipeline*. What is NOT handled: this is a standing **business**
limitation, not a bug -- as ingredient costs keep changing (milk +18%,
coffee +9% in the actual supplied emails) with no BOM ever supplied, the
system's menu-level `unit_cost_sar` in `menu_items.csv` becomes
**permanently stale** relative to reality, and every gross-margin number the
system reports is quietly computed against an outdated cost basis with no
mechanism to ever self-correct, because there is no recipe input to update
it from. Production fix: either obtain a real BOM/recipe table so unit costs
can be recomputed, or add an explicit "menu cost last verified: <date>"
staleness banner to the report so the owner knows exact-margin numbers may
be understating true cost pressure.

## 3. The scheduler process stops or loses its host

**What breaks**: `scheduler/run.py` runs as a single long-lived
`BlockingScheduler` process. If that process (or its host VM/container) dies
-- OOM kill, host reboot, deploy that forgot to restart it -- **no weekly run
occurs**, and nothing in this codebase currently notices. `RunLock`
(`src/persistence/run_lock.py`) only prevents *overlapping* runs while the
process is alive; it does not detect the process being *absent*. There is no
heartbeat, no external monitor, and no alert if a week goes by with zero
rows written to `weekly_runs` in `memory.sqlite`. This is exactly the kind of
failure that looks like "everything is fine" (the code is correct, the graph
would work) but produces silent absence of output. Production fix: run the
scheduler under a process supervisor (systemd/supervisord/a container
restart policy) AND add an external heartbeat check -- e.g. a cron-independent
watchdog that queries `SELECT MAX(started_at) FROM weekly_runs WHERE
profile_key = ?` and pages someone if the most recent run is older than
`schedule interval * 1.5`.
