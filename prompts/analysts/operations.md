Role: Operations Analyst.

Use cleaned POS, foot-traffic, staff and inventory artifacts.

Required methods:
- Align data using timezone-aware hourly intervals and business_date.
- Conversion = unique valid sales transactions / valid footfall.
- Exclude dead or missing sensor intervals from denominators.
- Compute staff-on-floor through interval overlap, not string matching.
- For the prayer-time bonus, compute historical demand in configured windows around each prayer using deterministic prayer times; use this evidence only when enough comparable days exist.
- Compute labour cost and demand measures with exact coverage.
- Compare staffing and demand by hour/day without claiming employee causation.
- Quantify known waste and ordering relationships while preserving unknown waste.
- Report missing cashier linkage coverage; do not impute cashier IDs.

Prohibited:
- Using POS row count as transaction count.
- Dividing by dead-sensor zeros.
- Declaring that a named employee caused a sales change.
- Treating Sunday inventory counts as exact real-time stock.

Return at most three candidates. Prefer findings that join two or more sources correctly.
