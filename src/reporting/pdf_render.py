"""Standalone PDF renderer, invoked as a subprocess by report_generator.

Why a subprocess rather than a plain function call
--------------------------------------------------
`langgraph dev` (and LangGraph Platform) install blockbuster, which patches
low-level blocking calls -- os.getcwd among them -- and raises BlockingError
when they are used. Playwright's sync API calls os.getcwd during startup, so
rendering in-process fails under the dev server with:

    PDF generation unavailable/failed: BlockingError: Blocking call to os.getcwd

...silently downgrading every Studio run to HTML-only (observed live, run
2026-03-23). The patch is process-wide, so moving the work to a worker thread
does not avoid it -- verified directly with a minimal graph running inside the
dev server. A separate interpreter has no blockbuster patches at all, which
makes this the one approach that works identically under `langgraph dev`, the
CLI, the scheduler and the test suite, without asking every caller to remember
`--allow-blocking`.

Playwright already spawns a browser process, so the extra interpreter is
marginal on top of what this path costs anyway.

Usage: python -m src.reporting.pdf_render <html_path> <pdf_path>
"""
from __future__ import annotations

import sys
from pathlib import Path


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    # weasyprint needs native GTK/Pango/GObject libraries that pip cannot
    # install on Windows (confirmed: pip install succeeds, rendering fails
    # with "cannot load library 'libgobject-2.0-0'"). Playwright's headless
    # Chromium is self-contained (installs into a user-level cache via
    # `playwright install chromium`, no system PATH/registry changes) and
    # renders the already-written HTML file directly, so relative asset paths
    # (the menu-engineering chart PNG) resolve exactly as they would in a real
    # browser -- no base_url workaround needed.
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(html_path.resolve().as_uri())
            page.pdf(path=str(pdf_path), format="A4", print_background=True)
        finally:
            browser.close()


if __name__ == "__main__":
    render_pdf(Path(sys.argv[1]), Path(sys.argv[2]))
