from __future__ import annotations

import base64
import binascii
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_BATCH_BYTES = 80 * 1024 * 1024


def _safe_relative_path(value: str) -> PurePosixPath | None:
    normalized = value.replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts or any(part in {"", "."} for part in path.parts):
        return None
    return path


def _registered_sources(project_root: Path) -> list[dict[str, Any]]:
    registry = yaml.safe_load((project_root / "config" / "source_registry.yaml").read_text(encoding="utf-8")) or {}
    return [item for item in registry.get("sources", []) if isinstance(item, dict) and item.get("path")]


def _match_source(path: PurePosixPath, sources: list[dict[str, Any]]) -> tuple[str, PurePosixPath] | None:
    wildcard_candidates: list[tuple[str, PurePosixPath]] = []
    for source in sources:
        registered = PurePosixPath(str(source["path"]).replace("\\", "/"))
        has_wildcard = any(any(token in part for token in ("*", "?", "[")) for part in registered.parts)
        if not has_wildcard:
            if path.name.lower() == registered.name.lower():
                return str(source["name"]), registered
            continue
        parent = registered.parent.name.lower()
        if registered.suffix.lower() != path.suffix.lower():
            continue
        candidate = (str(source["name"]), registered.parent / path.name)
        if parent in {part.lower() for part in path.parts[:-1]}:
            return candidate
        wildcard_candidates.append(candidate)

    # A normal browser file picker does not expose the file's original parent
    # directory. Accept a basename-only file when its extension maps to exactly
    # one registered wildcard source; keep ambiguous extensions rejected.
    if len(path.parts) == 1 and len(wildcard_candidates) == 1:
        return wildcard_candidates[0]
    return None


def process_browser_files(
    *, project_root: Path, cafe_id: str, data_dir: Path, files: list[dict[str, Any]]
) -> dict[str, Any]:
    sources = _registered_sources(project_root)
    batch_id = f"upload-{uuid.uuid4().hex[:12]}"
    safe_cafe = re.sub(r"[^A-Za-z0-9_.-]", "-", cafe_id)
    batch_root = project_root / "outputs" / "uploads" / safe_cafe / batch_id
    staged_root = batch_root / "staged"
    backup_root = batch_root / "backup"
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    decoded_files: list[tuple[dict[str, Any], str, PurePosixPath, bytes]] = []
    destinations: set[str] = set()
    total_bytes = 0

    for item in files:
        relative = _safe_relative_path(str(item.get("relative_path") or item.get("name") or ""))
        if relative is None:
            rejected.append({"name": item.get("name"), "status": "rejected", "reason": "Unsafe relative path."})
            continue
        match = _match_source(relative, sources)
        if match is None:
            rejected.append({"name": item.get("name"), "relative_path": str(relative), "status": "rejected", "reason": "Filename or type is not registered as a cafe source."})
            continue
        source_name, destination_relative = match
        destination_key = destination_relative.as_posix().lower()
        if destination_key in destinations:
            rejected.append({"name": item.get("name"), "relative_path": str(relative), "status": "rejected", "reason": "Another selected file maps to the same registered source."})
            continue
        try:
            content = base64.b64decode(str(item.get("content_base64") or ""), validate=True)
        except (binascii.Error, ValueError):
            rejected.append({"name": item.get("name"), "relative_path": str(relative), "status": "rejected", "reason": "File content could not be decoded."})
            continue
        if len(content) != int(item.get("size") or 0):
            rejected.append({"name": item.get("name"), "relative_path": str(relative), "status": "rejected", "reason": "File size changed during selection."})
            continue
        if len(content) > MAX_FILE_BYTES or total_bytes + len(content) > MAX_BATCH_BYTES:
            rejected.append({"name": item.get("name"), "relative_path": str(relative), "status": "rejected", "reason": "Upload size limit exceeded."})
            continue
        destinations.add(destination_key)
        total_bytes += len(content)
        decoded_files.append((item, source_name, destination_relative, content))

    if not decoded_files:
        return {"batch_id": batch_id, "accepted": [], "rejected": rejected, "processed_at": datetime.now(timezone.utc).isoformat()}

    staged_root.mkdir(parents=True, exist_ok=False)
    for item, source_name, destination_relative, content in decoded_files:
        staged_path = staged_root.joinpath(*destination_relative.parts)
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        staged_path.write_bytes(content)

        destination = data_dir.joinpath(*destination_relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            backup = backup_root.joinpath(*destination_relative.parts)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup)
        incoming = destination.with_name(f".{destination.name}.{batch_id}.incoming")
        shutil.copy2(staged_path, incoming)
        os.replace(incoming, destination)
        accepted.append(
            {
                "name": item["name"],
                "relative_path": str(item["relative_path"]),
                "source": source_name,
                "type": item.get("media_type") or destination.suffix.lower().lstrip(".") or "unknown",
                "size": len(content),
                "last_modified": item.get("last_modified"),
                "status": "accepted",
                "replaced_existing": (backup_root.joinpath(*destination_relative.parts)).exists(),
            }
        )

    result = {
        "batch_id": batch_id,
        "accepted": accepted,
        "rejected": rejected,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    (batch_root / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
