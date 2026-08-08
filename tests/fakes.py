"""Shared fake chat model for tests that exercise the real graph without
spending API calls.

Extracted from tests/integration/test_full_pipeline.py so the API test suite
can reuse it: those tests need a genuinely checkpointed pipeline run (paused at
the human gate) to exercise the manager-review / owner-decision flow, and that
has to come from running the real graph rather than from ambient local state.

The single fake inspects the system prompt to decide what to return, so one
object serves analyst code-generation, email extraction, content ideas, and the
structured-output call styles alike.
"""
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeStructuredChatModel:
    """Return value of _FakeChatModel.with_structured_output(schema): mirrors
    the real ChatModel.with_structured_output(...).invoke(...) contract (a
    dict shaped like `schema`), reusing the same prompt-sniffing fake content
    as .invoke() so both call styles produce consistent fake data."""

    def __init__(self, fake_model: "_FakeChatModel", schema: type):
        self._fake_model = fake_model
        self._schema_name = getattr(schema, "__name__", "")

    def invoke(self, messages):
        if self._schema_name == "SemanticReviewResult":
            return {"decision": "approve", "explanation": "", "required_fix": ""}
        if self._schema_name == "AlignmentCheckResult":
            return {"aligned": True, "explanation": ""}
        # EmailExtractionOutput / ContentIdeasOutput: both wrap a bare list under "items".
        raw = self._fake_model.invoke(messages).content
        return {"items": json.loads(raw)}


class _FakeChatModel:
    def invoke(self, messages):
        system_prompt = messages[0][1] if messages and isinstance(messages[0], tuple) else ""
        user_prompt = messages[1][1] if len(messages) > 1 else ""

        if "content strategist" in system_prompt.lower():
            return _FakeMessage(self._fake_content_ideas(user_prompt))
        if "extract structured business facts" in system_prompt.lower():
            return _FakeMessage(self._fake_email_extraction())
        # Default: analyst code-generation / repair prompt.
        return _FakeMessage(self._fake_analyst_code(user_prompt))

    def with_structured_output(self, schema, **kwargs):
        return _FakeStructuredChatModel(self, schema)

    @staticmethod
    def _fake_analyst_code(user_prompt: str) -> str:
        ctx = json.loads(user_prompt.split("Context (artifact paths, schemas, periods):\n", 1)[-1].split("\n\nWrite one complete")[0]) \
            if "Context (artifact paths, schemas, periods):" in user_prompt else {}
        inputs = ctx.get("input_artifacts", {})
        first_artifact = next(iter(inputs), None)
        period = ctx.get("analysis_period", {})
        period_start = period.get("start", "")
        period_end = period.get("end", "")
        return f'''
import json, os
import pandas as pd
meta = json.load(open(os.environ["ANALYST_INPUTS_JSON"]))
inputs = meta["inputs"]
findings = []
if {first_artifact!r} and {first_artifact!r} in inputs:
    df = pd.read_parquet(inputs[{first_artifact!r}])
    findings.append({{
        "title": "Row count summary",
        "claim": f"Observed {{len(df)}} rows in {first_artifact}.",
        "finding_type": "descriptive",
        "metrics": {{"row_count": {{"value": int(len(df)), "unit": "rows", "numerator": None,
                                    "denominator": None, "period_start": {period_start!r}, "period_end": {period_end!r}}}}},
        "source_names": [{first_artifact!r}] if {first_artifact!r} else [],
        "sample_size": int(len(df)),
        "coverage_notes": [],
        "assumptions": [],
        "confidence": 0.6,
    }})
result = {{"status": "success", "findings": findings}}
json.dump(result, open(meta["output_path"], "w"))
'''

    @staticmethod
    def _fake_email_extraction() -> str:
        return json.dumps([{
            "email_file": "fake.txt", "sender": "fake@example.com", "date": "2026-01-01",
            "subject": "fake", "category": "noise", "entity_or_ingredient": None, "old_price": None,
            "new_price": None, "currency": None, "unit": None, "effective_date": None,
            "event_start": None, "event_end": None, "location": None, "facts": ["fake"],
            "confidence": 0.5, "evidence_text": "fake",
        }])

    @staticmethod
    def _fake_content_ideas(user_prompt: str) -> str:
        ctx_start = user_prompt.find("Context:\n") + len("Context:\n")
        ctx_end = user_prompt.find("\n\nProduce exactly 3")
        ctx = json.loads(user_prompt[ctx_start:ctx_end]) if ctx_start >= 0 and ctx_end > 0 else {}
        findings = ctx.get("approved_findings", [])
        finding_id = findings[0]["finding_id"] if findings else ""
        metric_keys = findings[0]["metric_keys"] if findings else []
        local_ids = [e["context_id"] for e in ctx.get("local_context", [])][:1]
        cal_ids = [e["context_id"] for e in ctx.get("calendar_context", [])][:1]
        windows = ctx.get("posting_windows", [])
        menu = ctx.get("menu_reference", [])
        sku = menu[0]["sku"] if menu else "ICE-001"
        sku_ar = menu[0]["item_ar"] if menu else "منتج"
        sku_en = menu[0]["item_en"] if menu else "Product"

        ideas = []
        for i in range(3):
            window = windows[i % len(windows)] if windows else {"window_id": "", "busy_metric_keys": [], "post_date": "", "start_time_local": "10:00"}
            ideas.append({
                "idea_id": f"idea-{i}", "hook_ar": f"فكرة رقم {i}", "hook_en": f"Idea number {i} for you",
                "format": ["reel", "carousel", "trend_audio"][i % 3], "product_sku": sku,
                "product_name_ar": sku_ar, "product_name_en": sku_en, "finding_id": finding_id,
                "cited_metric_keys": metric_keys, "local_context_ids": local_ids, "calendar_context_ids": cal_ids,
                "posting_window_id": window["window_id"], "timing_metric_keys": window.get("busy_metric_keys", []),
                "rationale_ar": "سياق محلي وتقويمي", "rationale_en": f"Grounded rationale variant {i}",
                "post_date": window.get("post_date", ""), "post_time_local": window.get("start_time_local", "10:00"),
                "timing_reason": "observed busy period", "inventory_suitability": "unknown",
            })
        return json.dumps(ideas)


class _FakeToolCallingModel(BaseChatModel):
    """Minimal tool-calling chat model for create_react_agent.

    Lets the chat endpoint be exercised end-to-end with no network: it calls
    each scripted tool once, then answers with fixed text. A real BaseChatModel
    subclass because create_react_agent coerces its model to a Runnable and
    rejects a plain object.

    Needed because the prompt-sniffing _FakeChatModel above has no bind_tools
    and cannot drive a ReAct loop, so chat tests previously fell through to a
    real provider call (observed: a live Anthropic request 404ing on "gpt-4o").
    """

    tool_calls_script: list = []
    answer: str = "Fake grounded answer."

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002 -- tool schemas unused
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        called = {getattr(m, "name", None) for m in messages if isinstance(m, ToolMessage)}
        pending = [c for c in self.tool_calls_script if c["name"] not in called]
        if pending:
            call = pending[0]
            message = AIMessage(content="", tool_calls=[{
                "name": call["name"],
                "args": call.get("args", {}),
                "id": f"call_{call['name']}",
            }])
        else:
            message = AIMessage(content=self.answer)
        return ChatResult(generations=[ChatGeneration(message=message)])
