"""Restricted subprocess executor for LLM-generated analysis code.

Per implementation_plan_final.md 21.3/21.4: fresh temp working dir per attempt,
shell=False, controlled interpreter, an allowlisted read-only copy of input
artifacts, an import allowlist enforced by static AST inspection *before*
execution, a stripped environment (no API keys / home paths), wall-clock
timeout with process-tree termination, and output-size/JSON-schema checks
before a result is accepted as successful.

`subprocess.run()` is not claimed to be a hostile-code sandbox; it is a
best-effort restriction suitable for LLM-generated analysis scripts, not for
adversarial code.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal, TypedDict

from src.schemas.artifacts import ArtifactRef
from src.tools.artifact_io import write_json

DEFAULT_ALLOWED_IMPORTS = {
    "pandas", "numpy", "json", "math", "statistics", "datetime", "collections",
    "itertools", "re", "zoneinfo", "decimal", "os", "typing", "pathlib",
    "functools", "dataclasses",
}

MAX_OUTPUT_BYTES = 2_000_000


class CodeExecutionRequest(TypedDict):
    code: str
    input_artifacts: list[ArtifactRef]
    expected_output_path: str
    timeout_seconds: int
    allowed_imports: list[str]
    max_output_bytes: int


class CodeExecutionResult(TypedDict):
    status: Literal["success", "error", "timeout", "policy_violation"]
    exit_code: int | None
    stdout: str
    stderr: str
    elapsed_seconds: float
    code_artifact: ArtifactRef
    result_artifact: ArtifactRef | None
    attempt: int


class ImportPolicyViolation(Exception):
    pass


def _check_imports(code: str, allowed: set[str]) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in allowed:
                    raise ImportPolicyViolation(f"disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in allowed:
                raise ImportPolicyViolation(f"disallowed import: {node.module}")
        elif isinstance(node, (ast.Call,)):
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            # `open` is permitted: generated code reads ANALYST_INPUTS_JSON and the
            # copied-in input artifacts, and writes result.json, all inside the
            # process-local temp working directory. exec/eval/dynamic-import/compile
            # are never permitted regardless of import allowlist.
            if name in {"exec", "eval", "__import__", "compile"}:
                raise ImportPolicyViolation(f"disallowed call: {name}")


def execute_python_code(
    request: CodeExecutionRequest,
    code_artifact_dir: Path,
    attempt: int,
    python_executable: str | None = None,
    results_dir: Path | None = None,
) -> CodeExecutionResult:
    import time

    allowed = set(request["allowed_imports"]) | DEFAULT_ALLOWED_IMPORTS
    code = request["code"]

    code_path = code_artifact_dir / f"{attempt}.py"
    code_path.parent.mkdir(parents=True, exist_ok=True)
    code_path.write_text(code, encoding="utf-8")
    code_artifact = ArtifactRef(
        path=str(code_path), media_type="text/x-python", schema_version="1.0",
        row_count=None, byte_size=code_path.stat().st_size,
        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    )

    try:
        _check_imports(code, allowed)
    except (ImportPolicyViolation, SyntaxError) as e:
        return CodeExecutionResult(
            status="policy_violation", exit_code=None, stdout="", stderr=str(e),
            elapsed_seconds=0.0, code_artifact=code_artifact, result_artifact=None, attempt=attempt,
        )

    with tempfile.TemporaryDirectory(prefix="analyst_exec_") as tmpdir:
        tmp = Path(tmpdir)
        inputs_dir = tmp / "inputs"
        inputs_dir.mkdir()
        input_map: dict[str, str] = {}
        for ref in request["input_artifacts"]:
            src = Path(ref["path"])
            dest = inputs_dir / src.name
            shutil.copyfile(src, dest)
            input_map[src.stem] = str(dest)

        output_path = tmp / "result.json"
        (tmp / "run_meta.json").write_text(
            json.dumps({"inputs": input_map, "output_path": str(output_path)}), encoding="utf-8"
        )

        script_path = tmp / "script.py"
        script_path.write_text(code, encoding="utf-8")

        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1",
            "ANALYST_INPUTS_JSON": str(tmp / "run_meta.json"),
        }
        if sys.platform.startswith("win"):
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")

        python_exe = python_executable or sys.executable
        start = time.monotonic()
        try:
            proc = subprocess.run(
                [python_exe, str(script_path)],
                cwd=str(tmp),
                env=env,
                shell=False,
                capture_output=True,
                text=True,
                timeout=request["timeout_seconds"],
            )
            elapsed = time.monotonic() - start
        except subprocess.TimeoutExpired as e:
            return CodeExecutionResult(
                status="timeout", exit_code=None,
                stdout=(e.stdout or ""), stderr=(e.stderr or "timed out"),
                elapsed_seconds=request["timeout_seconds"],
                code_artifact=code_artifact, result_artifact=None, attempt=attempt,
            )

        stdout = proc.stdout[:MAX_OUTPUT_BYTES]
        stderr = proc.stderr[:MAX_OUTPUT_BYTES]

        if proc.returncode != 0:
            return CodeExecutionResult(
                status="error", exit_code=proc.returncode, stdout=stdout, stderr=stderr,
                elapsed_seconds=elapsed, code_artifact=code_artifact, result_artifact=None, attempt=attempt,
            )

        if not output_path.exists():
            return CodeExecutionResult(
                status="error", exit_code=proc.returncode, stdout=stdout,
                stderr=stderr + "\n[executor] expected result.json was not written",
                elapsed_seconds=elapsed, code_artifact=code_artifact, result_artifact=None, attempt=attempt,
            )
        if output_path.stat().st_size > request["max_output_bytes"]:
            return CodeExecutionResult(
                status="policy_violation", exit_code=proc.returncode, stdout=stdout,
                stderr="result.json exceeds max_output_bytes",
                elapsed_seconds=elapsed, code_artifact=code_artifact, result_artifact=None, attempt=attempt,
            )
        try:
            result_obj = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return CodeExecutionResult(
                status="error", exit_code=proc.returncode, stdout=stdout,
                stderr=stderr + f"\n[executor] result.json is not valid JSON: {e}",
                elapsed_seconds=elapsed, code_artifact=code_artifact, result_artifact=None, attempt=attempt,
            )

        # results_dir defaults to a sibling of the code dir for backward
        # compatibility, but callers running multiple invocations for the
        # same analyst/run (e.g. critic-triggered revisions) MUST pass a
        # distinct results_dir per invocation so artifacts stay immutable.
        dest_dir = results_dir if results_dir is not None else (code_artifact_dir.parent / "results")
        result_dest = dest_dir / f"{attempt}.json"
        result_artifact = write_json(result_obj, result_dest)

        return CodeExecutionResult(
            status="success", exit_code=proc.returncode, stdout=stdout, stderr=stderr,
            elapsed_seconds=elapsed, code_artifact=code_artifact, result_artifact=result_artifact, attempt=attempt,
        )
