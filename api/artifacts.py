from __future__ import annotations

import base64
import hashlib
import json
import re
import sqlite3
from contextlib import closing
from datetime import date, datetime
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from api.database import ApiDatabase


SCHEMA_VERSION = "1.0"
_PATH_PATTERN = re.compile(
    r"(?:file://[^<>\"'\r\n]+|\\\\[^<>\"'\r\n]+|[A-Za-z]:[\\/][^<>\"'\r\n]+|"
    r"/(?:home|users|tmp|var|opt|workspace|mnt)/[^<>\"'\r\n]+|(?:outputs|data|db)[\\/][^<>\"'\r\n]+)",
    re.IGNORECASE,
)
_SOURCE_FILES = {
    "pos": "pos_transactions.csv",
    "menu": "menu_items.csv",
    "traffic": "foot_traffic.csv",
    "staff": "staff_shifts.csv",
    "inventory": "inventory_weekly.xlsx",
    "emails": "supplier_emails",
    "reviews": "customer_reviews.json",
}
_IDENTITY_FIELDS = {
    "pos": ("transaction_id", "sku", "timestamp"),
    "menu": ("sku",),
    "traffic": ("date", "hour"),
    "staff": ("employee_id", "date", "shift_start"),
    "inventory": ("week_starting", "sku"),
    "emails": ("email_id", "message_id", "received_at", "subject"),
    "reviews": ("review_id",),
}


def strip_path_text(value: str) -> str:
    return _PATH_PATTERN.sub("[Link attached]", value)


class _ReportHTMLSanitizer(HTMLParser):
    _allowed_tags = {
        "html", "head", "title", "body", "main", "section", "article", "header", "footer",
        "h1", "h2", "h3", "h4", "p", "div", "span", "table", "thead", "tbody", "tfoot",
        "tr", "th", "td", "caption", "ul", "ol", "li", "strong", "em", "b", "i", "small",
        "br", "hr", "code", "pre", "blockquote",
    }
    _void_tags = {"br", "hr"}
    _blocked_tags = {"script", "style", "iframe", "object", "embed", "svg", "math", "link", "form"}
    _allowed_attributes = {"class", "dir", "lang", "colspan", "rowspan", "scope"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.output: list[str] = []
        self._blocked_depth = 0

    def handle_decl(self, decl: str) -> None:
        if self._blocked_depth == 0 and decl.lower() == "doctype html":
            self.output.append("<!doctype html>")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self._blocked_depth:
            if tag not in self._void_tags:
                self._blocked_depth += 1
            return
        if tag in self._blocked_tags:
            self._blocked_depth = 1
            return
        if tag not in self._allowed_tags:
            return
        safe_attrs = []
        for name, value in attrs:
            name = name.lower()
            if name in self._allowed_attributes and value is not None:
                safe_attrs.append(f' {name}="{escape(strip_path_text(value), quote=True)}"')
        self.output.append(f"<{tag}{''.join(safe_attrs)}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._blocked_depth:
            return
        self.handle_starttag(tag, attrs)
        if tag.lower() in self._allowed_tags and tag.lower() not in self._void_tags:
            self.output.append(f"</{tag.lower()}>")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._blocked_depth:
            self._blocked_depth -= 1
            return
        if tag in self._allowed_tags and tag not in self._void_tags:
            self.output.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if self._blocked_depth == 0:
            self.output.append(escape(strip_path_text(data)))


def sanitize_report_html(value: str) -> str:
    sanitizer = _ReportHTMLSanitizer()
    sanitizer.feed(value)
    sanitizer.close()
    return "".join(sanitizer.output)


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, str):
        return strip_path_text(value)
    if isinstance(value, dict):
        return {
            str(key): _safe_value(item)
            for key, item in value.items()
            if "path" not in str(key).lower()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    return str(value)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "cafe"


def profile_id(profile: dict[str, Any]) -> str:
    coordinates = profile.get("coordinates", {})
    material = "|".join(
        str(value)
        for value in (
            profile.get("cafe_name") or profile.get("name"),
            profile.get("city"),
            coordinates.get("lat", ""),
            coordinates.get("lng", ""),
        )
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:8]
    return f"{_slug(str(profile.get('cafe_name') or profile.get('name') or 'cafe'))}-{digest}"


class ArtifactRepository:
    def __init__(
        self, root: Path, database: ApiDatabase, include_test_evidence: bool = True,
        checkpoint_db: Path | None = None,
    ):
        self.root = Path(root).resolve()
        self.database = database
        self.include_test_evidence = include_test_evidence
        # Must be the same store RunService writes -- see ApiSettings.checkpoint_db.
        self.checkpoint_db = Path(checkpoint_db).resolve() if checkpoint_db else self.root / "db" / "checkpoints.sqlite"

    def list_cafes(self) -> list[dict[str, Any]]:
        cafes: list[dict[str, Any]] = []
        data_root = self.root / "data"
        for profile_path in sorted(data_root.glob("*/cafe_profile.json")):
            try:
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            cafe_id = profile_id(profile)
            cafes.append(
                {
                    "id": cafe_id,
                    "branch_id": cafe_id,
                    "name": profile["cafe_name"],
                    "city": profile["city"],
                    "region": profile["region"],
                    "country": profile["country"],
                    "timezone": profile["timezone"],
                    "currency": profile["currency"],
                    "seats": profile["seats"],
                    "data_status": "available",
                    "_data_dir": profile_path.parent,
                    "_profile_path": profile_path,
                }
            )
        for cafe in self.database.list_api_cafes():
            cafes.append(
                {
                    "id": cafe["id"],
                    "branch_id": cafe["id"],
                    **{key: cafe[key] for key in ("name", "city", "region", "country", "timezone", "currency", "seats")},
                    "data_status": "not_connected",
                    "_data_dir": None,
                    "_profile_path": None,
                }
            )
        return cafes

    def public_cafe(self, cafe: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in cafe.items() if not key.startswith("_")}

    def get_cafe(self, cafe_id: str) -> dict[str, Any] | None:
        return next((cafe for cafe in self.list_cafes() if cafe["id"] == cafe_id), None)

    def _memory_paths(self) -> list[Path]:
        paths: list[Path] = []
        primary = self.root / "db" / "memory.sqlite"
        if primary.exists():
            paths.append(primary)
        if self.include_test_evidence:
            paths.extend(sorted((self.root / "outputs" / "test_evidence").glob("*memory*.sqlite")))
        return paths

    def _memory_rows(self) -> Iterable[tuple[Path, sqlite3.Row]]:
        for path in self._memory_paths():
            conn: sqlite3.Connection | None = None
            try:
                conn = sqlite3.connect(path)
                conn.row_factory = sqlite3.Row
                tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "weekly_runs" not in tables:
                    conn.close()
                    continue
                rows = conn.execute("SELECT * FROM weekly_runs ORDER BY started_at DESC").fetchall()
            except sqlite3.Error:
                continue
            finally:
                if conn is not None:
                    conn.close()
            yield from ((path, row) for row in rows)

    def _checkpoint_path(self) -> Path:
        return self.checkpoint_db

    def _checkpoint_thread_ids(self) -> list[str]:
        path = self._checkpoint_path()
        if not path.exists():
            return []
        try:
            with closing(sqlite3.connect(path)) as conn:
                return [
                    row[0]
                    for row in conn.execute(
                        "SELECT thread_id FROM checkpoints GROUP BY thread_id ORDER BY MAX(rowid) DESC"
                    )
                ]
        except sqlite3.Error:
            return []

    def _checkpoint_tuple(self, run_id: str):
        path = self._checkpoint_path()
        if not path.exists():
            return None
        from src.persistence.checkpointer import build_checkpointer

        saver = build_checkpointer(path)
        try:
            return saver.get_tuple({"configurable": {"thread_id": run_id}})
        finally:
            saver.conn.close()

    def checkpoint_values(self, run_id: str) -> dict[str, Any] | None:
        item = self._checkpoint_tuple(run_id)
        return item.checkpoint.get("channel_values", {}) if item else None

    @staticmethod
    def _stage_from_channels(channels: list[str]) -> str:
        for channel in channels:
            if channel.startswith("branch:to:"):
                return channel.removeprefix("branch:to:")
        if "report" in channels:
            return "manager_review"
        if "final_findings" in channels:
            return "findings"
        if "cleaned_artifacts" in channels:
            return "cleaning"
        return "running"

    def checkpoint_events(self, run_id: str) -> list[dict[str, Any]]:
        path = self._checkpoint_path()
        if not path.exists():
            return []
        from src.persistence.checkpointer import build_checkpointer

        saver = build_checkpointer(path)
        try:
            checkpoints = list(saver.list({"configurable": {"thread_id": run_id}}))
        finally:
            saver.conn.close()
        events: list[dict[str, Any]] = []
        for item in reversed(checkpoints):
            checkpoint = item.checkpoint
            channels = list(checkpoint.get("updated_channels") or [])
            values = checkpoint.get("channel_values", {})
            stage = self._stage_from_channels(channels)
            run_status = str(values.get("run_status") or "running")
            status = "waiting_review" if stage == "human_gate" else run_status
            event_id = f"{run_id}:{checkpoint.get('id')}"
            events.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "event_id": event_id,
                    "run_id": run_id,
                    "type": "run.status",
                    "at": checkpoint.get("ts"),
                    "stage": stage,
                    "status": status,
                    "message": f"Run advanced to {stage.replace('_', ' ')}.",
                }
            )
        return events

    def _apply_report_state(self, run: dict[str, Any]) -> dict[str, Any]:
        decision = self.database.latest_decision(run["id"])
        review = self.database.latest_review(run["id"])
        if decision:
            run["report_state"] = {
                "approve": "approved",
                "reject": "rejected",
                "edit": "analyzing",
            }[decision["decision"]]
        elif review and review["decision"] == "submit":
            run["report_state"] = "owner_review"
        elif run["status"] in {"queued", "running"}:
            run["report_state"] = "analyzing"
        elif run["status"] == "succeeded":
            run["report_state"] = "delivered"
        elif run.get("has_report"):
            run["report_state"] = "manager_review"
        else:
            run["report_state"] = "analyzing"
        run.pop("has_report", None)
        return run

    def list_runs(self, cafe_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        runs: dict[str, dict[str, Any]] = {}
        for _, row in self._memory_rows():
            if cafe_id and row["profile_key"] != cafe_id:
                continue
            runs[row["run_id"]] = {
                "id": row["run_id"],
                "cafe_id": row["profile_key"],
                "status": row["status"],
                "stage": "completed",
                "created_at": row["started_at"],
                "updated_at": row["completed_at"] or row["started_at"],
                "analysis_period": {"start": row["analysis_start"], "end": row["analysis_end"]},
                "findings_count": row["findings_count"],
                "error_count": row["critic_rejections"],
                "has_report": bool(row["report_html_path"]),
            }
        for run_id in self._checkpoint_thread_ids():
            item = self._checkpoint_tuple(run_id)
            if item is None:
                continue
            values = item.checkpoint.get("channel_values", {})
            config = values.get("config")
            resolved_cafe_id = getattr(config, "profile_key", None)
            if not resolved_cafe_id or (cafe_id and resolved_cafe_id != cafe_id):
                continue
            events = self.checkpoint_events(run_id)
            latest_stage = events[-1]["stage"] if events else "running"
            raw_status = str(values.get("run_status") or "running")
            status = "waiting_review" if latest_stage == "human_gate" else raw_status
            errors = values.get("errors") or []
            runs[run_id] = {
                "id": run_id,
                "cafe_id": resolved_cafe_id,
                "status": status,
                "stage": "manager_review" if status == "waiting_review" else latest_stage,
                "created_at": events[0]["at"] if events else item.checkpoint.get("ts"),
                "updated_at": item.checkpoint.get("ts"),
                "analysis_period": _safe_value(values.get("analysis_period")),
                "findings_count": len(values.get("final_findings") or []),
                "error_count": len(errors),
                "has_report": bool(values.get("report")),
            }
        for row in self.database.list_live_runs(cafe_id):
            existing = runs.get(row["id"], {})
            runs[row["id"]] = {
                **existing,
                "id": row["id"],
                "cafe_id": row["cafe_id"],
                "status": row["status"],
                "stage": row["stage"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "analysis_period": existing.get("analysis_period"),
                "findings_count": existing.get("findings_count", 0),
                "error_count": existing.get("error_count", int(bool(row.get("error_code")))),
                "has_report": existing.get("has_report", row["status"] == "waiting_review"),
            }
        result = [self._apply_report_state(run) for run in runs.values()]
        result.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return result[:limit]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return next((run for run in self.list_runs(limit=500) if run["id"] == run_id), None)

    def _memory_row_for_run(self, run_id: str) -> tuple[Path, sqlite3.Row] | None:
        return next(((path, row) for path, row in self._memory_rows() if row["run_id"] == run_id), None)

    @staticmethod
    def _public_evidence(run_id: str, finding_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
        artifact_material = str(evidence.get("result_path") or f"{run_id}:{finding_id}:{evidence.get('result_key')}")
        artifact_id = f"artifact-{hashlib.sha256(artifact_material.encode()).hexdigest()[:16]}"
        evidence_id = f"evidence-{hashlib.sha256((finding_id + artifact_material).encode()).hexdigest()[:16]}"
        return {
            "id": evidence_id,
            "metric_name": _safe_value(evidence.get("metric_name") or evidence.get("result_key")),
            "value": _safe_value(evidence.get("value")),
            "unit": _safe_value(evidence.get("unit")),
            "numerator": _safe_value(evidence.get("numerator")),
            "denominator": _safe_value(evidence.get("denominator")),
            "period_start": _safe_value(evidence.get("period_start")),
            "period_end": _safe_value(evidence.get("period_end")),
            "comparison_period_start": _safe_value(evidence.get("comparison_period_start")),
            "comparison_period_end": _safe_value(evidence.get("comparison_period_end")),
            "source_names": _safe_value(list(evidence.get("source_names") or [])),
            "artifact_id": artifact_id,
        }

    def findings(self, run_id: str) -> list[dict[str, Any]]:
        memory = self._memory_row_for_run(run_id)
        if memory:
            path, _ = memory
            with closing(sqlite3.connect(path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM findings_history WHERE run_id=? AND approved=1 ORDER BY id", (run_id,)
                ).fetchall()
            return [
                {
                    "id": row["finding_id"],
                    "analyst": _safe_value(row["analyst_name"]),
                    "title": strip_path_text(row["title"]),
                    "claim": strip_path_text(row["claim"]),
                    "type": None,
                    "confidence": row["confidence"],
                    "approved": bool(row["approved"]),
                    "evidence": [
                        self._public_evidence(run_id, row["finding_id"], evidence)
                        for evidence in json.loads(row["metrics_json"] or "[]")
                    ],
                }
                for row in rows
            ]
        values = self.checkpoint_values(run_id) or {}
        findings: list[dict[str, Any]] = []
        for finding in values.get("final_findings") or []:
            finding_id = finding.get("finding_id") or f"finding-{len(findings) + 1}"
            findings.append(
                {
                    "id": finding_id,
                    "analyst": _safe_value(finding.get("analyst_name")),
                    "title": strip_path_text(str(finding.get("title") or "")),
                    "claim": strip_path_text(str(finding.get("claim") or "")),
                    "type": finding.get("finding_type"),
                    "confidence": finding.get("confidence"),
                    "approved": True,
                    "evidence": [
                        self._public_evidence(run_id, finding_id, evidence)
                        for evidence in finding.get("evidence") or []
                    ],
                }
            )
        return findings

    def report(self, run_id: str) -> dict[str, Any] | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        stored_path: str | None = None
        whatsapp = ""
        generated_at = run.get("updated_at")
        memory = self._memory_row_for_run(run_id)
        if memory:
            _, row = memory
            stored_path = row["report_html_path"]
            whatsapp = row["whatsapp_summary"] or ""
        else:
            report = (self.checkpoint_values(run_id) or {}).get("report") or {}
            stored_path = report.get("html_path")
            whatsapp = report.get("whatsapp_summary") or ""
            generated_at = report.get("generated_at") or generated_at
        html = self._read_report(stored_path) if stored_path else None
        if html is None:
            return None
        return {
            "run_id": run_id,
            "state": run["report_state"],
            "format": "html",
            "html": html,
            "whatsapp_summary": strip_path_text(whatsapp),
            "generated_at": generated_at,
        }

    def _read_report(self, stored_path: str) -> str | None:
        candidate = Path(stored_path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve()
            resolved.relative_to((self.root / "outputs" / "reports").resolve())
            html = resolved.read_text(encoding="utf-8")
        except (OSError, ValueError):
            return None
        return sanitize_report_html(html)

    def _checkpoint_for_cafe(self, cafe_id: str) -> tuple[str, dict[str, Any]] | None:
        for run_id in self._checkpoint_thread_ids():
            values = self.checkpoint_values(run_id) or {}
            if getattr(values.get("config"), "profile_key", None) == cafe_id:
                return run_id, values
        return None

    def sources(self, cafe_id: str) -> list[dict[str, Any]]:
        checkpoint = self._checkpoint_for_cafe(cafe_id)
        if checkpoint:
            run_id, values = checkpoint
            items = []
            for result in values.get("source_results") or []:
                items.append(
                    {
                        "id": result.get("source_name"),
                        "name": result.get("source_name"),
                        "status": result.get("status"),
                        "raw_rows": result.get("raw_row_count"),
                        "accepted_rows": result.get("accepted_row_count"),
                        "rejected_rows": result.get("rejected_row_count"),
                        "schema_version": result.get("schema_version") or SCHEMA_VERSION,
                        "last_run_id": run_id,
                    }
                )
            return items
        # No checkpoint: fall back to the artifacts of the most recent recorded
        # run for this cafe (see _latest_run_dir_for_cafe) so row counts reflect
        # data that genuinely exists, rather than reporting every source as
        # unknown for runs this API did not itself launch.
        fallback = self._latest_run_dir_for_cafe(cafe_id)
        cafe = self.get_cafe(cafe_id)
        data_dir = cafe.get("_data_dir") if cafe else None
        return [
            {
                "id": source,
                "name": source,
                "status": "available" if data_dir and (data_dir / relative).exists() else "missing",
                **self._fallback_row_counts(fallback, source),
                "schema_version": SCHEMA_VERSION,
            }
            for source, relative in _SOURCE_FILES.items()
        ]

    @staticmethod
    def _parquet_rows(path: Path) -> int | None:
        try:
            return pq.ParquetFile(path).metadata.num_rows
        except Exception:  # noqa: BLE001 -- unreadable/absent artifact is just "unknown"
            return None

    def _fallback_row_counts(
        self, fallback: tuple[str, Path] | None, source: str
    ) -> dict[str, Any]:
        """Row counts read straight from a run's parquet artifacts, for sources
        whose numbers are not available from a checkpoint. Reads only parquet
        footer metadata, so it does not load the data."""
        if fallback is None:
            return {"raw_rows": None, "accepted_rows": None, "rejected_rows": None, "last_run_id": None}
        run_id, run_dir = fallback
        raw = self._parquet_rows(run_dir / "parsed" / f"{source}.parquet")
        accepted = self._parquet_rows(run_dir / "cleaned" / f"{source}.parquet")
        return {
            "raw_rows": raw,
            "accepted_rows": accepted,
            "rejected_rows": (raw - accepted) if raw is not None and accepted is not None else None,
            "last_run_id": run_id if raw is not None else None,
        }

    def _latest_run_dir_for_cafe(self, cafe_id: str) -> tuple[str, Path] | None:
        """Most recent recorded run for this cafe that still has artifacts on disk.

        Runs produced by the CLI, the scheduler or LangGraph Studio leave their
        artifacts under outputs/artifacts/<run_id>/ and a row in the memory DB,
        but no entry in the API's checkpoint store. Without this, the Data
        Explorer and lineage view report "no processed data" for a cafe the
        cafe list simultaneously advertises as `available`.

        Scoped by profile_key so one cafe can never be shown another's rows,
        and both parsed and cleaned come from the same run so lineage compares
        like with like. _memory_rows() is already ordered newest-first.
        """
        artifacts_root = self.root / "outputs" / "artifacts"
        for _, row in self._memory_rows():
            keys = row.keys()
            if "profile_key" not in keys or "run_id" not in keys:
                continue
            if row["profile_key"] != cafe_id:
                continue
            run_dir = artifacts_root / str(row["run_id"])
            if (run_dir / "parsed").is_dir():
                return str(row["run_id"]), run_dir
        return None

    def _artifact_paths(self, cafe_id: str, source: str) -> tuple[str, Path, Path | None] | None:
        checkpoint = self._checkpoint_for_cafe(cafe_id)
        if checkpoint is not None:
            run_id, values = checkpoint
            result = next(
                (item for item in values.get("source_results") or [] if item.get("source_name") == source), None
            )
            parsed = (result or {}).get("artifact", {}).get("path")
            cleaned = (values.get("cleaned_artifacts") or {}).get(source, {}).get("path")
            if parsed:
                parsed_path = self._resolve_artifact(parsed)
                cleaned_path = self._resolve_artifact(cleaned) if cleaned else None
                if parsed_path is not None:
                    return run_id, parsed_path, cleaned_path

        fallback = self._latest_run_dir_for_cafe(cafe_id)
        if fallback is None:
            return None
        run_id, run_dir = fallback
        parsed_path = self._resolve_artifact(str(run_dir / "parsed" / f"{source}.parquet"))
        if parsed_path is None:
            return None
        cleaned_path = self._resolve_artifact(str(run_dir / "cleaned" / f"{source}.parquet"))
        return run_id, parsed_path, cleaned_path

    def _resolve_artifact(self, stored_path: str | None) -> Path | None:
        if not stored_path:
            return None
        candidate = Path(stored_path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve()
            resolved.relative_to((self.root / "outputs" / "artifacts").resolve())
        except ValueError:
            return None
        return resolved if resolved.is_file() else None

    @staticmethod
    def _identity(source: str, row: dict[str, Any]) -> tuple[Any, ...]:
        fields = [field for field in _IDENTITY_FIELDS.get(source, ()) if field in row]
        if not fields:
            fields = sorted(row)[:3]
        return tuple(_safe_value(row.get(field)) for field in fields)

    @classmethod
    def _record_id(cls, source: str, row: dict[str, Any]) -> str:
        identity = json.dumps(cls._identity(source, row), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(f"{source}:{identity}".encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _encode_cursor(offset: int) -> str:
        return base64.urlsafe_b64encode(str(offset).encode()).decode().rstrip("=")

    @staticmethod
    def _decode_cursor(cursor: str | None) -> int:
        if not cursor:
            return 0
        padded = cursor + "=" * (-len(cursor) % 4)
        try:
            return max(0, int(base64.urlsafe_b64decode(padded).decode()))
        except (ValueError, UnicodeDecodeError):
            raise ValueError("Invalid cursor") from None

    def data_page(
        self, cafe_id: str, source: str, *, limit: int, cursor: str | None
    ) -> tuple[list[dict[str, Any]], str | None] | None:
        paths = self._artifact_paths(cafe_id, source)
        if paths is None:
            return None
        _, parsed_path, _ = paths
        offset = self._decode_cursor(cursor)
        table = pq.read_table(parsed_path)
        rows = table.slice(offset, limit).to_pylist()
        items = [
            {"record_id": self._record_id(source, row), "data": _safe_value(row)} for row in rows
        ]
        next_offset = offset + len(rows)
        next_cursor = self._encode_cursor(next_offset) if next_offset < table.num_rows else None
        return items, next_cursor

    def _find_row(self, path: Path, source: str, record_id: str) -> dict[str, Any] | None:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                if self._record_id(source, row) == record_id:
                    return row
        return None

    def _find_identity(self, path: Path, source: str, identity: tuple[Any, ...]) -> dict[str, Any] | None:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                if self._identity(source, row) == identity:
                    return row
        return None

    def lineage(self, cafe_id: str, source: str, record_id: str) -> dict[str, Any] | None:
        paths = self._artifact_paths(cafe_id, source)
        if paths is None:
            return None
        run_id, parsed_path, cleaned_path = paths
        raw = self._find_row(parsed_path, source, record_id)
        if raw is None:
            return None
        cleaned = self._find_identity(cleaned_path, source, self._identity(source, raw)) if cleaned_path else None
        changes: list[dict[str, Any]] = []
        if cleaned is None:
            changes.append({"field": "_record", "before": "present", "after": None, "reason": "dropped by cleaning"})
        else:
            for field in sorted(set(raw) | set(cleaned)):
                before, after = _safe_value(raw.get(field)), _safe_value(cleaned.get(field))
                if before != after:
                    changes.append(
                        {
                            "field": field,
                            "before": before,
                            "after": after,
                            "reason": "derived during cleaning" if field not in raw else "normalized during cleaning",
                        }
                    )
        return {
            "cafe_id": cafe_id,
            "source": source,
            "record_id": record_id,
            "run_id": run_id,
            "raw": _safe_value(raw),
            "cleaned": _safe_value(cleaned),
            "changes": changes,
        }
