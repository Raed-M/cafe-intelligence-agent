"""Deterministic report rendering (Module 7). HTML is the mandatory baseline;
PDF is a best-effort additional artifact -- its failure degrades the run
status but never invalidates the HTML/WhatsApp outputs already produced.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.persistence.memory_store import MemoryStore
from src.persistence.trends import all_metric_streaks
from src.reporting.charts import render_menu_engineering_chart
from src.reporting.whatsapp_summary import build_whatsapp_summary
from src.schemas.reports import ReportOutput
from src.state import CafeIntelligenceState

TEMPLATE_DIR = Path("src/reporting/templates")


PDF_TIMEOUT_SECONDS = 120

# Derived from __file__ rather than Path.cwd(): os.getcwd is one of the calls
# blockbuster blocks under `langgraph dev`, and this module is imported inside
# that server. src/reporting/report_generator.py -> project root is two up.
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _abs(p: Path) -> Path:
    """Absolute path without touching os.getcwd (blocked under `langgraph dev`
    -- Path.absolute() calls it for relative inputs, and report paths here are
    relative, e.g. outputs/reports/<run_id>/report.html)."""
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _render_pdf(html_path: Path, pdf_path: Path) -> tuple[str | None, str | None]:
    """Renders the report PDF, returning (path, warning).

    Runs in a subprocess -- see src/reporting/pdf_render.py for why that is
    required rather than merely tidy (blockbuster under `langgraph dev` raises
    on Playwright's sync API, and the patch is process-wide so a worker thread
    does not escape it).

    PDF is best-effort by design: any failure returns a warning string and
    leaves the HTML and WhatsApp artifacts untouched."""
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "src.reporting.pdf_render",
             str(_abs(html_path)), str(_abs(pdf_path))],
            capture_output=True, text=True, timeout=PDF_TIMEOUT_SECONDS,
            cwd=str(_PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return None, f"PDF generation unavailable/failed: timed out after {PDF_TIMEOUT_SECONDS}s"
    except Exception as e:  # noqa: BLE001
        return None, f"PDF generation unavailable/failed: {type(e).__name__}: {e}"

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return None, f"PDF generation unavailable/failed: {detail[-1] if detail else 'unknown error'}"
    if not _abs(pdf_path).exists():
        return None, "PDF generation unavailable/failed: renderer reported success but wrote no file"
    return str(pdf_path), None


def generate_report(state: CafeIntelligenceState) -> dict[str, Any]:
    config = state["config"]
    run_id = state["run_id"]
    out_dir = Path("outputs", "reports", run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_findings = state.get("final_findings", [])
    content_ideas = state.get("content_ideas", [])
    data_quality = state.get("data_quality", {})
    context_bundle = state.get("context_bundle", {"evidence": [], "posting_windows": [], "search_status": "unavailable", "warnings": []})
    run_status = state.get("run_status") or ("succeeded" if final_findings else "partial")

    chart_path = render_menu_engineering_chart(final_findings, out_dir)

    whatsapp_max = config.app_settings.report.whatsapp_max_chars
    report_ref = f"outputs/reports/{run_id}/report.html"
    whatsapp_text, whatsapp_len = build_whatsapp_summary(
        config.raw_profile.cafe_name, run_status, state["analysis_period"],
        final_findings, content_ideas, report_ref, whatsapp_max,
        use_llm_compression=config.app_settings.report.use_llm_summary_compression,
        model_name=config.app_settings.models.report_summary,
    )
    whatsapp_parts = whatsapp_text.split("\n\n", 1)
    whatsapp_ar = whatsapp_parts[0]
    whatsapp_en = whatsapp_parts[1] if len(whatsapp_parts) > 1 else ""

    critic_results = state.get("critic_results", {})

    trend_statements = []
    try:
        current_metrics = {
            ev["result_key"]: ev["value"] for f in final_findings for ev in f["evidence"]
            if isinstance(ev["value"], (int, float))
        }
        if current_metrics:
            store = MemoryStore(config.memory_db)
            trend_statements = all_metric_streaks(
                store, config.profile_key, current_metrics, state["analysis_period"]["end"]
            )
            store.close()
    except Exception:  # noqa: BLE001
        trend_statements = []

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=select_autoescape(["html"]))
    template = env.get_template("report.html.j2")
    html = template.render(
        cafe_name=config.raw_profile.cafe_name,
        run_id=run_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        analysis_period=state["analysis_period"],
        recommendation_period=state["recommendation_period"],
        run_status=run_status,
        whatsapp_summary_ar=whatsapp_ar,
        whatsapp_summary_en=whatsapp_en,
        data_quality=data_quality,
        final_findings=final_findings,
        menu_engineering_chart=(Path(chart_path).name if chart_path else None),
        content_ideas=content_ideas,
        context_bundle=context_bundle,
        step_count=state.get("step_count", 0),
        cost_usd=state.get("cost_usd", 0.0),
        critic_rejections=critic_results.get("total_rejections", 0),
        source_statuses={r["source_name"]: r["status"] for r in state.get("source_results", [])},
        trend_statements=trend_statements,
    )
    html_path = out_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")

    whatsapp_path = out_dir / "whatsapp_summary.txt"
    whatsapp_path.write_text(whatsapp_text, encoding="utf-8")

    pdf_path, pdf_warning = _render_pdf(html_path, out_dir / "report.pdf")

    report = ReportOutput(
        html_path=str(html_path),
        pdf_path=str(pdf_path) if pdf_path else None,
        pdf_warning=pdf_warning,
        whatsapp_summary=whatsapp_text,
        whatsapp_path=str(whatsapp_path),
        whatsapp_char_count=whatsapp_len,
        generated_at=datetime.now(timezone.utc).isoformat(),
        context={"chart_path": chart_path},
    )

    final_status = run_status
    if not final_findings and not content_ideas:
        final_status = "partial" if data_quality.get("sources_successful") else "failed"

    return {"report": dict(report), "run_status": final_status, "step_count": 1}
