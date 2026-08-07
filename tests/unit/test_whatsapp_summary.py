"""WhatsApp summary fitting.

The message has a hard character budget and is often the only thing the owner
actually reads, so the two failure modes that matter are opposite: cutting a
sentence in half, and going silent rather than cutting anything.
"""
from src.reporting.whatsapp_summary import build_whatsapp_summary, first_sentence

PERIOD = {"start": "2026-03-23T00:00:00+03:00", "end": "2026-03-30T00:00:00+03:00"}
LINK = "outputs/reports/r1/report.html"

LONG_CLAIM = (
    "Sales revenue fell by 23.54% while the median margin rate remained high at 71.11%. "
    "This combination suggests that the revenue drop is not being driven by aggressive "
    "discounting or margin-eroding promotions, but rather by a reduction in transaction "
    "volume, which implies a demand-side rather than a pricing-side problem and points at "
    "retention rather than discounting as the correct response this week."
)


def _f(fid, title, claim):
    return {"finding_id": fid, "title": title, "claim": claim}


def _build(findings, ideas=(), max_chars=500):
    return build_whatsapp_summary(
        cafe_name="Qahwa Saihat", run_status="succeeded", analysis_period=PERIOD,
        final_findings=list(findings), content_ideas=list(ideas),
        report_reference=LINK, max_chars=max_chars,
    )


def test_long_top_finding_still_reports_something():
    """Regression, found live on 2026-03-23: the top-ranked finding was a long
    cross-domain claim, the fitter stopped at the first line that would not
    fit, and the message shipped with no findings at all."""
    text, n = _build([_f("F1", "Revenue Decline and Margin Resilience", LONG_CLAIM)])
    assert n <= 500
    assert "Revenue Decline and Margin Resilience" in text
    assert LINK in text


def test_oversized_claim_degrades_to_a_complete_sentence():
    text, n = _build([_f("F1", "Revenue Decline", LONG_CLAIM)])
    assert n <= 500
    # the opening sentence survives whole; the trailing analysis is dropped
    assert "Sales revenue fell by 23.54% while the median margin rate remained high at 71.11%." in text
    assert "retention rather than discounting" not in text


def test_no_line_is_cut_mid_sentence():
    text, _ = _build([_f("F1", "A", LONG_CLAIM), _f("F2", "B", LONG_CLAIM)])
    for line in text.splitlines():
        assert not line.endswith("..."), line


def test_short_findings_all_fit():
    findings = [_f(f"F{i}", f"Title {i}", f"Short claim {i}.") for i in range(1, 4)]
    text, n = _build(findings)
    assert n <= 500
    for i in range(1, 4):
        assert f"Title {i}" in text


def test_a_short_finding_behind_a_long_one_is_not_lost():
    """A finding that cannot fit must not stop the ones after it."""
    text, n = _build([
        _f("F1", "T" * 300, "X" * 300),          # cannot fit in any form
        _f("F2", "Short One", "Brief claim."),
    ])
    assert n <= 500
    assert "Short One" in text


def test_budget_is_never_exceeded():
    findings = [_f(f"F{i}", f"Title {i}", LONG_CLAIM) for i in range(3)]
    for limit in (200, 300, 500, 900):
        _, n = _build(findings, ideas=[{"id": "c1"}], max_chars=limit)
        assert n <= limit, limit


def test_no_findings_says_so_in_both_languages():
    text, n = _build([])
    assert n <= 500
    assert "insufficient evidence this week" in text
    assert "لا توجد أدلة كافية" in text


def test_first_sentence_handles_no_terminator():
    assert first_sentence("no full stop here") == "no full stop here"
    assert first_sentence("One. Two.") == "One."
