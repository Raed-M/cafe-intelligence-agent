"""Generic specialist-analyst runner.

Each analyst (sales/margin/operations/customer/anomaly) is a thin config over
this shared engine: build the prompt with schema/period context, ask the
configured LLM to write Python analysis code, execute it in the restricted
subprocess, and on failure feed stderr back through the code-repair prompt up
to `analyst_code_attempts` times. Findings are only ever built from a
successful, schema-valid structured result -- never from prose.
"""
from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.config.runtime_config import RuntimeCafeConfig
from src.schemas.artifacts import ArtifactRef
from src.schemas.findings import AnalystFinding, MetricEvidence
from src.tools.artifact_io import artifact_columns, artifact_dir
from src.tools.code_executor import CodeExecutionRequest, execute_python_code

CommonPromptPath = Path("prompts/analysts/common.md")
CodeRepairPromptPath = Path("prompts/code_repair.md")


def _build_correlation_hints(
    cleaned_artifacts: dict[str, ArtifactRef], analysis_period: dict[str, str], previous_period: dict[str, str]
) -> dict[str, Any]:
    """Deterministic, code-only cross-source hints (no LLM) -- see
    src/analysis/correlation_hints.py. Additive scaffolding only: the analyst
    must still independently compute and cite any number it uses from these
    in its own executed code before a finding can cite it."""
    try:
        from src.analysis.correlation_hints import calendar_overlaps, procurement_cost_scenarios, weekly_signal_deltas

        hints: dict[str, Any] = {
            "signal_deltas": weekly_signal_deltas(cleaned_artifacts, analysis_period, previous_period),
            "calendar_overlaps": calendar_overlaps(analysis_period),
        }
        if "emails" in cleaned_artifacts:
            from src.tools.artifact_io import read_dataframe

            hints["procurement_cost_scenarios"] = procurement_cost_scenarios(read_dataframe(cleaned_artifacts["emails"]))
        return hints
    except Exception:  # noqa: BLE001
        return {}


@dataclass
class AnalystSpec:
    name: str
    prompt_path: Path
    required_artifacts: list[str]
    optional_artifacts: list[str] = field(default_factory=list)


@dataclass
class AnalystRunResult:
    analyst_name: str
    findings: list[AnalystFinding]
    status: str
    attempts: int
    notes: list[str]


CodeGenerator = Callable[[str, dict[str, Any]], str]
"""A function (system_prompt, context) -> generated python source. The default
implementation calls the configured OpenAI model; tests inject a fake generator
to exercise the repair loop deterministically without network access."""


def _default_llm_code_generator(model_name: str):
    def _generate(system_prompt: str, context: dict[str, Any]) -> str:
        from src.tools.llm_factory import extract_text, get_chat_model

        llm = get_chat_model(model_name, temperature=0)
        user_prompt = (
            "Context (artifact paths, schemas, periods):\n"
            f"{json.dumps(context, indent=2, default=str)}\n\n"
            "Write one complete Python program. os.environ['ANALYST_INPUTS_JSON'] is a "
            "FILE PATH, not JSON content -- open and json.load() that file to get "
            "{'inputs': {artifact_name: path}, 'output_path': path}, e.g.:\n"
            "    with open(os.environ['ANALYST_INPUTS_JSON']) as f:\n"
            "        run_meta = json.load(f)\n"
            "    inputs = run_meta['inputs']; output_path = run_meta['output_path']\n"
            "Read only the listed parquet/json input artifacts with pandas/json, using the "
            "exact column names given in each input artifact's `columns` list above -- never "
            "guess or invent a column name. Write your final result as JSON to output_path. "
            "Do not print secrets. Return ONLY the code, no markdown fences, no prose."
        )
        resp = llm.invoke([("system", system_prompt), ("user", user_prompt)])
        return _strip_code_fences(extract_text(resp))

    return _generate


def _strip_code_fences(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


_CLAIM_PLACEHOLDER_RE = re.compile(r"<<([a-zA-Z0-9_]+)>>")


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return str(int(value)) if value == int(value) else f"{value:.2f}"
    return str(value)


def _substitute_claim_placeholders(claim: str, metrics: dict[str, Any]) -> tuple[str, list[str]]:
    """Replaces <<metric_key>> tokens in a claim with the literal value from this
    finding's own metrics dict, so a restated number can never drift from the
    evidence it cites -- the model names which fact to state, our code supplies
    the digit. Returns (substituted_claim, unresolved_keys); an unresolved key
    means the finding must be dropped (see _build_findings). A NaN/inf value
    (e.g. a 0/0 division that slipped through generated code) is treated as
    unresolved too -- substituting the literal string "nan" into a claim would
    just be a different flavour of the same restatement problem this exists to
    prevent, so drop the finding instead."""
    unresolved: list[str] = []

    def _sub(m: re.Match) -> str:
        key = m.group(1)
        metric = metrics.get(key)
        value = metric.get("value") if isinstance(metric, dict) else None
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            unresolved.append(key)
            return m.group(0)
        return _format_metric_value(value)

    return _CLAIM_PLACEHOLDER_RE.sub(_sub, claim), unresolved


def _repair_code_generator(model_name: str):
    def _repair(previous_code: str, stderr: str, context: dict[str, Any]) -> str:
        from src.tools.llm_factory import extract_text, get_chat_model

        system_prompt = CodeRepairPromptPath.read_text(encoding="utf-8")
        llm = get_chat_model(model_name, temperature=0)
        user_prompt = (
            f"Previous code:\n```python\n{previous_code}\n```\n\n"
            f"stderr:\n{stderr}\n\n"
            f"Context:\n{json.dumps(context, indent=2, default=str)}\n\n"
            "Return ONLY the complete replacement program, no markdown fences, no prose."
        )
        resp = llm.invoke([("system", system_prompt), ("user", user_prompt)])
        return _strip_code_fences(extract_text(resp))

    return _repair


def run_analyst(
    spec: AnalystSpec,
    run_id: str,
    config: RuntimeCafeConfig,
    cleaned_artifacts: dict[str, ArtifactRef],
    analysis_period: dict[str, str],
    previous_period: dict[str, str],
    trailing_baseline_periods: list[dict[str, str]],
    code_generator: CodeGenerator | None = None,
    repair_generator: Callable[[str, str, dict[str, Any]], str] | None = None,
    revision_feedback: list[str] | None = None,
    invocation_tag: str = "initial",
) -> AnalystRunResult:
    limits = config.app_settings.limits
    max_attempts = limits.analyst_code_attempts
    model_name = config.app_settings.models.analyst

    available = {k: v for k, v in cleaned_artifacts.items() if k in spec.required_artifacts + spec.optional_artifacts}
    missing_required = [a for a in spec.required_artifacts if a not in available]
    if missing_required:
        return AnalystRunResult(
            analyst_name=spec.name, findings=[], status="unavailable", attempts=0,
            notes=[f"missing required artifacts: {missing_required}"],
        )
    empty_required = [a for a in spec.required_artifacts if (available[a].get("row_count") or 0) == 0]
    if empty_required:
        return AnalystRunResult(
            analyst_name=spec.name, findings=[], status="unavailable", attempts=0,
            notes=[f"required artifact(s) empty, no fabricated findings: {empty_required}"],
        )

    common_prompt = CommonPromptPath.read_text(encoding="utf-8").replace(
        "MAX_CANDIDATE_FINDINGS", str(limits.max_candidate_findings_per_analyst)
    )
    role_prompt = spec.prompt_path.read_text(encoding="utf-8")
    system_prompt = f"{common_prompt}\n\n---\n\n{role_prompt}"

    correlation_hints = _build_correlation_hints(cleaned_artifacts, analysis_period, previous_period)

    context = {
        "analysis_period": analysis_period,
        "previous_period": previous_period,
        "trailing_baseline_periods": trailing_baseline_periods,
        "input_artifacts": {
            k: {"path": v["path"], "row_count": v["row_count"], "columns": artifact_columns(v)}
            for k, v in available.items()
        },
        "correlation_hints": correlation_hints,
        "required_output_schema": {
            "status": "success | insufficient_data",
            "findings": [{
                "title": "str",
                "claim": "str (use <<metric_key>> placeholders for evidence-backed numbers, "
                         "substituted from this finding's own metrics after you return -- see "
                         "common.md rule 13; never type the literal digit yourself)",
                "finding_type": "str",
                "metrics": {"<result_key>": {"value": "number|string|null", "unit": "str|null",
                                              "numerator": "number|null", "denominator": "number|null",
                                              "period_start": "iso", "period_end": "iso"}},
                "source_names": ["str"], "sample_size": "int|null", "coverage_notes": ["str"],
                "assumptions": ["str"], "confidence": "0..1",
            }],
        },
    }
    if revision_feedback:
        context["critic_revision_feedback"] = revision_feedback

    gen = code_generator or _default_llm_code_generator(model_name)
    repair = repair_generator or _repair_code_generator(model_name)

    # Namespaced by invocation_tag so a revision rerun never overwrites the
    # artifact files of a prior invocation for the same analyst/run_id --
    # artifacts must stay immutable even across critic-triggered reruns.
    code_dir = artifact_dir(config.artifact_root, run_id, "code", spec.name, invocation_tag)
    results_dir = artifact_dir(config.artifact_root, run_id, "results", spec.name, invocation_tag)
    input_refs = list(available.values())

    notes: list[str] = []
    try:
        code = gen(system_prompt, context)
    except Exception as e:  # noqa: BLE001
        # A model/LLM-layer failure (auth error, network error, etc.) must not
        # crash the whole graph -- one analyst degrades to "failed", exactly
        # like a code-execution failure would.
        return AnalystRunResult(
            analyst_name=spec.name, findings=[], status="failed", attempts=0,
            notes=[f"code generation call failed: {type(e).__name__}: {e}"],
        )

    last_stderr = ""
    exec_result = None
    for attempt in range(1, max_attempts + 1):
        request = CodeExecutionRequest(
            code=code, input_artifacts=input_refs, expected_output_path="result.json",
            timeout_seconds=limits.code_timeout_seconds, allowed_imports=[],
            max_output_bytes=2_000_000,
        )
        exec_result = execute_python_code(request, code_dir, attempt=attempt, results_dir=results_dir)
        if exec_result["status"] == "success":
            break
        last_stderr = exec_result["stderr"] or f"execution status: {exec_result['status']}"
        notes.append(f"attempt {attempt} failed ({exec_result['status']}): {last_stderr[:200]}")
        if attempt < max_attempts:
            try:
                code = repair(code, last_stderr, context)
            except Exception as e:  # noqa: BLE001
                notes.append(f"repair call failed: {type(e).__name__}: {e}")
                break

    if exec_result is None or exec_result["status"] != "success":
        return AnalystRunResult(
            analyst_name=spec.name, findings=[], status="failed",
            attempts=exec_result["attempt"] if exec_result else 0, notes=notes,
        )

    result_obj = json.loads(Path(exec_result["result_artifact"]["path"]).read_text(encoding="utf-8"))
    if result_obj.get("status") == "insufficient_data":
        notes.append(result_obj.get("explanation", "insufficient_data"))
        return AnalystRunResult(analyst_name=spec.name, findings=[], status="insufficient_data", attempts=exec_result["attempt"], notes=notes)

    try:
        findings, build_notes = _build_findings(
            spec, result_obj, exec_result["code_artifact"], exec_result["result_artifact"], limits.max_candidate_findings_per_analyst
        )
    except Exception as e:  # noqa: BLE001
        # A malformed-but-schema-passing result (e.g. an unexpected value type
        # in metrics) must degrade this one analyst to "failed", not crash the
        # whole graph run -- same principle as the code-gen/repair try/excepts above.
        return AnalystRunResult(
            analyst_name=spec.name, findings=[], status="failed", attempts=exec_result["attempt"],
            notes=notes + [f"finding construction failed: {type(e).__name__}: {e}"],
        )
    notes.extend(build_notes)
    return AnalystRunResult(analyst_name=spec.name, findings=findings, status="success", attempts=exec_result["attempt"], notes=notes)


def _build_findings(
    spec: AnalystSpec, result_obj: dict[str, Any], code_artifact: ArtifactRef,
    result_artifact: ArtifactRef, max_findings: int,
) -> tuple[list[AnalystFinding], list[str]]:
    findings: list[AnalystFinding] = []
    notes: list[str] = []
    for raw in result_obj.get("findings", [])[:max_findings]:
        metrics_raw = raw.get("metrics", {})
        claim_text, unresolved = _substitute_claim_placeholders(raw.get("claim", ""), metrics_raw)
        coverage_notes: list[str] = []
        for note in raw.get("coverage_notes", []):
            substituted, bad = _substitute_claim_placeholders(note, metrics_raw)
            coverage_notes.append(substituted)
            unresolved.extend(bad)
        assumptions: list[str] = []
        for assumption in raw.get("assumptions", []):
            substituted, bad = _substitute_claim_placeholders(assumption, metrics_raw)
            assumptions.append(substituted)
            unresolved.extend(bad)
        if unresolved:
            notes.append(
                f"finding {raw.get('title', '')!r} dropped: claim/coverage_notes/assumptions reference "
                f"placeholder(s) {unresolved} not present in this finding's own metrics"
            )
            continue
        evidence: list[MetricEvidence] = []
        for key, m in metrics_raw.items():
            evidence.append(MetricEvidence(
                metric_name=key, value=m.get("value"), unit=m.get("unit"),
                numerator=m.get("numerator"), denominator=m.get("denominator"),
                period_start=m.get("period_start", ""), period_end=m.get("period_end", ""),
                comparison_period_start=m.get("comparison_period_start"),
                comparison_period_end=m.get("comparison_period_end"),
                source_names=raw.get("source_names", []),
                result_path=result_artifact["path"], result_key=key,
            ))
        findings.append(AnalystFinding(
            finding_id=f"F-{spec.name}-{uuid.uuid4().hex[:8]}",
            analyst_name=spec.name, title=raw.get("title", ""), claim=claim_text,
            finding_type=raw.get("finding_type", "trend"), evidence=evidence,
            source_names=raw.get("source_names", []), code_artifact=code_artifact,
            result_artifact=result_artifact, sample_size=raw.get("sample_size"),
            coverage_notes=coverage_notes, assumptions=assumptions,
            confidence=float(raw.get("confidence", 0.5)),
            business_impact_score=float(raw.get("business_impact_score", 0.5)),
            actionability_score=float(raw.get("actionability_score", 0.5)),
            revision_count=0,
            execution_metadata={"generated_at": datetime.now(timezone.utc).isoformat()},
        ))
    return findings, notes
