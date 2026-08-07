"""Placeholder substitution in analyst claims.

The model names which fact to state; our code supplies the digit. These cover
the substitution contract itself, including the magnitude form added after a
live run put "decreased by -23.54%" into a real WhatsApp message.
"""
import math

from src.analysts.base import _substitute_claim_placeholders, with_abs_variants

METRICS = {
    "delta_pct": {"value": -23.54, "unit": "%"},
    "revenue": {"value": 42910.2, "unit": "SAR"},
    "count": {"value": 6.0, "unit": None},
    "ratio": {"value": 0.5, "unit": None},
    "label": {"value": "Cortado", "unit": None},
    "broken": {"value": float("nan"), "unit": "%"},
}


def _subs():
    return with_abs_variants(METRICS)


def test_signed_value_substituted_verbatim():
    text, bad = _substitute_claim_placeholders("revenue changed by <<delta_pct>>%", _subs())
    assert bad == [] and text == "revenue changed by -23.54%"


def test_abs_variant_avoids_double_negative():
    text, bad = _substitute_claim_placeholders("revenue fell <<delta_pct__abs>>%", _subs())
    assert bad == [] and text == "revenue fell 23.54%"


def test_abs_variant_leaves_positive_values_unchanged():
    text, bad = _substitute_claim_placeholders("revenue was <<revenue__abs>> SAR", _subs())
    assert bad == [] and text == "revenue was 42910.20 SAR"


def test_whole_floats_render_without_decimals():
    text, bad = _substitute_claim_placeholders("<<count>> items", _subs())
    assert bad == [] and text == "6 items"


def test_unknown_key_is_reported_unresolved():
    text, bad = _substitute_claim_placeholders("<<nope>> and <<nope__abs>>", _subs())
    assert sorted(bad) == ["nope", "nope__abs"]
    assert "<<nope>>" in text, "unresolved placeholders are left in place for the caller to drop"


def test_nan_is_unresolved_rather_than_printed():
    """Substituting the string "nan" would just be a different flavour of the
    restatement problem this mechanism exists to prevent."""
    _, bad = _substitute_claim_placeholders("waste was <<broken>>%", _subs())
    assert bad == ["broken"]


def test_non_numeric_metrics_get_no_abs_variant():
    subs = _subs()
    assert "label__abs" not in subs
    assert subs["label"]["value"] == "Cortado"


def test_abs_variant_preserves_unit_and_original_entry():
    subs = _subs()
    assert subs["delta_pct"]["value"] == -23.54
    assert subs["delta_pct__abs"]["value"] == 23.54
    assert subs["delta_pct__abs"]["unit"] == "%"


def test_nan_abs_variant_still_unresolved():
    subs = _subs()
    assert math.isnan(subs["broken__abs"]["value"])
    _, bad = _substitute_claim_placeholders("<<broken__abs>>", subs)
    assert bad == ["broken__abs"]
