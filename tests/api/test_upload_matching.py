from pathlib import PurePosixPath

from api.services.uploads import _match_source


def test_basename_matches_the_only_wildcard_source() -> None:
    sources = [{"name": "emails", "path": "supplier_emails/*.txt"}]

    match = _match_source(PurePosixPath("2026-07-15_13.txt"), sources)

    assert match == ("emails", PurePosixPath("supplier_emails/2026-07-15_13.txt"))


def test_basename_stays_rejected_when_wildcard_source_is_ambiguous() -> None:
    sources = [
        {"name": "emails", "path": "supplier_emails/*.txt"},
        {"name": "notes", "path": "notes/*.txt"},
    ]

    assert _match_source(PurePosixPath("item.txt"), sources) is None
