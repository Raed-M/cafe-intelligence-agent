"""One-off live-LLM scan across every complete week in data/qahwa_saihat,
to see what real findings the analyst/critic pipeline surfaces end to end.

Run from the project root:
    cd "cafe-intelligence-agent"
    ../.venv/Scripts/python.exe <this file>

Ingestion+cleaning is deterministic and week-independent, so it runs once and
is reused for every week's analyst run (only the analysts/critic make live
LLM calls, per week).
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Users\User\Desktop\Week 6\Cafe Intelligence Agent\cafe-intelligence-agent")
assert (PROJECT_ROOT / "pyproject.toml").exists(), f"not the project root: {PROJECT_ROOT}"

sys.path.insert(0, str(PROJECT_ROOT))

LOG_PATH = Path(__file__).resolve().parent / "full_dataset_scan.log"
_log_file = open(LOG_PATH, "a", encoding="utf-8")


def log(msg: str) -> None:
    print(msg)
    _log_file.write(msg + "\n")
    _log_file.flush()
    os.fsync(_log_file.fileno())


from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from src.cleaning.cleaner import clean_and_materialise  # noqa: E402
from src.config.runtime_config import resolve_runtime_config  # noqa: E402
from src.graph.analysis_subgraph import build_analysis_subgraph  # noqa: E402
from src.graph.critic_subgraph import build_critic_subgraph  # noqa: E402
from src.graph.ingestion_subgraph import build_ingestion_subgraph  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "qahwa_saihat"
OUT_PATH = Path(__file__).resolve().parent / "full_dataset_scan_results.json"

FIRST_MONDAY = date(2026, 1, 5)
LAST_MONDAY = date(2026, 7, 20)  # last week fully inside the 2026-01-05..2026-07-26 data range


def build_config(target_week: date):
    return resolve_runtime_config(
        profile_path=DATA_DIR / "cafe_profile.json",
        data_dir=DATA_DIR,
        app_settings_path=PROJECT_ROOT / "config" / "app_settings.yaml",
        source_registry_path=PROJECT_ROOT / "config" / "source_registry.yaml",
        target_week=target_week,
        artifact_root=PROJECT_ROOT / "outputs" / "artifacts",
    )


def main() -> None:
    log("=== full_dataset_scan starting ===")
    base_run_id = "scan_" + uuid.uuid4().hex[:8]
    base_config = build_config(FIRST_MONDAY)

    log("Ingesting + cleaning once (deterministic, week-independent)...")
    ingestion_graph = build_ingestion_subgraph()
    ingest_out = ingestion_graph.invoke({"run_id": base_run_id, "config": base_config})
    clean_out = clean_and_materialise({
        "run_id": base_run_id, "config": base_config, "source_results": ingest_out["source_results"],
    })
    cleaned_artifacts = clean_out["cleaned_artifacts"]
    log(f"Sources: {[(r['source_name'], r['status']) for r in ingest_out['source_results']]}")

    analysis_graph = build_analysis_subgraph()
    critic_graph = build_critic_subgraph()

    weeks = []
    d = FIRST_MONDAY
    while d <= LAST_MONDAY:
        weeks.append(d)
        d += timedelta(days=7)
    log(f"Total weeks to scan: {len(weeks)}")

    all_results = []
    for i, monday in enumerate(weeks):
        config = build_config(monday)
        run_id = f"{base_run_id}_w{i:02d}"
        state = {
            "run_id": run_id,
            "config": config,
            "cleaned_artifacts": cleaned_artifacts,
            "analysis_period": config.analysis_period,
            "previous_period": config.previous_period,
            "trailing_baseline_periods": config.trailing_baseline_periods,
            "candidate_findings": [],
        }
        log(f"\n=== Week {i + 1}/{len(weeks)}: {config.analysis_period['start'][:10]} .. {config.analysis_period['end'][:10]} ===")
        try:
            analysis_out = analysis_graph.invoke(state)
        except Exception as e:  # noqa: BLE001
            log(f"  analysis_subgraph raised: {type(e).__name__}: {e}")
            all_results.append({"week_start": config.analysis_period["start"], "error": str(e)})
            continue
        state.update(analysis_out)
        n_candidates = len(state.get("candidate_findings", []))
        log(f"  candidate findings: {n_candidates}")
        for f in state.get("candidate_findings", []):
            log(f"    * [{f['analyst_name']}] {f['title']}: {f['claim']}")

        try:
            critic_out = critic_graph.invoke(state)
        except Exception as e:  # noqa: BLE001
            log(f"  critic_subgraph raised: {type(e).__name__}: {e}")
            all_results.append({
                "week_start": config.analysis_period["start"],
                "candidate_findings": state.get("candidate_findings", []),
                "error": str(e),
            })
            continue
        state.update(critic_out)
        final = state.get("final_findings", [])
        log(f"  final (critic-approved, ranked) findings: {len(final)}")
        for f in final:
            log(f"    -> [{f['analyst_name']}] {f['title']}: {f['claim']}")

        all_results.append({
            "week_start": config.analysis_period["start"],
            "week_end": config.analysis_period["end"],
            "candidate_findings": state.get("candidate_findings", []),
            "critic_results": state.get("critic_results", {}),
            "final_findings": final,
        })
        OUT_PATH.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")

    log(f"\nDone. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
