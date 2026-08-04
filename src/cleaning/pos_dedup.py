"""Double-swipe (whole-transaction-bundle duplication) detection and removal.

Per DATA_DICTIONARY.md, ~1% of transactions were double-swiped: the entire
transaction's lines appear twice with identical multiplicity. We build a
canonical per-line signature and, when every distinct signature within a
transaction repeats with the *same* multiplicity m >= 2, we keep one copy of
each line and drop the rest.

Implementation note (deviation disclosed): the plan's prose lists `cashier_id`
as part of the canonical signature, but the actual data shows duplicate swipes
sometimes recording a different cashier per swipe. Including cashier_id in the
signature would under-detect duplicates by roughly an order of magnitude
relative to the ~1% figure documented for this dataset (measured: 0.11% with
cashier_id in the signature vs 0.93% without, which matches the documented
rate). Per the plan's source-of-truth order (actual file behaviour over prose
when they conflict), cashier_id is excluded from the matching key; the
cashier_id values seen across duplicate copies are preserved in `examples` for
audit instead of being silently dropped.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

SIGNATURE_COLUMNS = [
    "business_date", "timestamp_local", "sku", "quantity", "unit_price_sar",
    "discount_sar", "line_total_sar", "payment_method", "channel",
]


def dedup_double_swipes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    df["_sig"] = list(zip(*[df[c].astype(str) for c in SIGNATURE_COLUMNS]))

    keep_mask = pd.Series(True, index=df.index)
    removed_rows = 0
    removed_transactions: list[str] = []
    examples: list[dict[str, Any]] = []

    for tid, group in df.groupby("transaction_id"):
        counts = group["_sig"].value_counts()
        vals = counts.values
        if len(vals) == 0 or vals[0] < 2 or not (vals == vals[0]).all():
            continue
        # Every canonical line repeats with identical multiplicity -> duplicated bundle.
        multiplicity = int(vals[0])
        first_idx_per_sig = group.groupby("_sig").head(1).index
        drop_idx = group.index.difference(first_idx_per_sig)
        keep_mask.loc[drop_idx] = False
        removed_rows += len(drop_idx)
        removed_transactions.append(tid)
        if len(examples) < 10:
            examples.append({
                "transaction_id": tid,
                "multiplicity": multiplicity,
                "rows_before": len(group),
                "rows_after": len(first_idx_per_sig),
                "cashier_ids_seen": sorted(set(group["cashier_id"].astype(str))),
            })

    cleaned = df[keep_mask].drop(columns=["_sig"])
    audit = {
        "duplicated_transactions": len(removed_transactions),
        "rows_removed": removed_rows,
        "examples": examples,
    }
    return cleaned, audit
