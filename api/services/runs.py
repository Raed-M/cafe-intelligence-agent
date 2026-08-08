from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from api.artifacts import ArtifactRepository
from api.config import ApiSettings
from api.database import ApiDatabase


logger = logging.getLogger(__name__)


class RunService:
    def __init__(self, settings: ApiSettings, database: ApiDatabase, artifacts: ArtifactRepository):
        self.settings = settings
        self.database = database
        self.artifacts = artifacts
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="waddehha-run")

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def start(self, cafe: dict, target_week: date | None, user_id: str) -> dict:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        run = self.database.create_live_run(
            run_id, cafe["id"], target_week.isoformat() if target_week else None, user_id
        )
        if not self.settings.runs_enabled:
            self.database.update_live_run(run_id, status="failed", stage="disabled", error_code="runs_disabled")
            return self.database.get_live_run(run_id) or run
        if cafe.get("_profile_path") is None or cafe.get("_data_dir") is None:
            self.database.update_live_run(
                run_id, status="failed", stage="preflight", error_code="data_not_connected"
            )
            return self.database.get_live_run(run_id) or run
        self._executor.submit(self._execute, run_id, cafe, target_week)
        return run

    def _execute(self, run_id: str, cafe: dict, target_week: date | None) -> None:
        self.database.update_live_run(run_id, status="running", stage="preflight")
        saver = None
        try:
            from src.config.runtime_config import resolve_runtime_config
            from src.graph.main_graph import build_main_graph
            from src.persistence.checkpointer import build_checkpointer

            config = resolve_runtime_config(
                profile_path=cafe["_profile_path"],
                data_dir=cafe["_data_dir"],
                app_settings_path=self.settings.project_root / "config" / "app_settings.yaml",
                source_registry_path=self.settings.project_root / "config" / "source_registry.yaml",
                target_week=target_week,
                artifact_root=self.settings.project_root / "outputs" / "artifacts",
                checkpoint_db=self.settings.checkpoint_db,
                memory_db=self.settings.project_root / "db" / "memory.sqlite",
            )
            saver = build_checkpointer(config.checkpoint_db)
            graph = build_main_graph(checkpointer=saver)
            thread_config = {"configurable": {"thread_id": run_id}, "recursion_limit": 200}
            initial_state = {
                "run_id": run_id,
                "thread_id": run_id,
                "config": config,
                "analysis_period": config.analysis_period,
                "previous_period": config.previous_period,
                "trailing_baseline_periods": config.trailing_baseline_periods,
                "recommendation_period": config.recommendation_period,
                "critic_round": 0,
                "content_repair_attempts": 0,
            }
            output = graph.invoke(initial_state, config=thread_config)
            snapshot = graph.get_state(thread_config)
            if snapshot.next:
                self.database.update_live_run(run_id, status="waiting_review", stage="manager_review")
            else:
                status = str(output.get("run_status") or "partial")
                self.database.update_live_run(run_id, status=status, stage="completed")
        except Exception:
            logger.exception("Background run %s failed", run_id)
            self.database.update_live_run(
                run_id, status="failed", stage="failed", error_code="backend_run_failed"
            )
        finally:
            if saver is not None:
                saver.conn.close()

    def resume(self, run_id: str, decision: str) -> None:
        self.database.update_live_run(run_id, status="running", stage="owner_decision")
        self._executor.submit(self._resume, run_id, decision)

    def _resume(self, run_id: str, decision: str) -> None:
        saver = None
        try:
            from src.graph.main_graph import build_main_graph
            from src.persistence.checkpointer import build_checkpointer

            saver = build_checkpointer(self.settings.checkpoint_db)
            graph = build_main_graph(checkpointer=saver)
            thread_config = {"configurable": {"thread_id": run_id}, "recursion_limit": 200}
            graph.update_state(thread_config, {"human_decision": decision})
            output = graph.invoke(None, config=thread_config)
            snapshot = graph.get_state(thread_config)
            if snapshot.next:
                self.database.update_live_run(run_id, status="waiting_review", stage="manager_review")
            else:
                status = str(output.get("run_status") or "partial")
                self.database.update_live_run(run_id, status=status, stage="completed")
        except Exception:
            logger.exception("Resume for run %s failed", run_id)
            self.database.update_live_run(
                run_id, status="failed", stage="failed", error_code="backend_resume_failed"
            )
        finally:
            if saver is not None:
                saver.conn.close()

    def can_resume(self, run_id: str) -> bool:
        return self.artifacts.checkpoint_values(run_id) is not None
