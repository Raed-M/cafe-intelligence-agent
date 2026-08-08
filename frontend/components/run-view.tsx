"use client";

import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, ArrowLeft, ArrowRight, Braces, Clock3, Radio, ShieldCheck, TriangleAlert } from "lucide-react";
import Link from "next/link";
import { api } from "@/lib/api";
import { displayError, formatDate } from "@/lib/format";
import { invalidateRunScoped, livePolling, useRunStatusSync } from "@/lib/run-state";
import type { AgentState, RunEvent } from "@/lib/types";
import { useWorkspace } from "@/components/providers";
import { HumanGate } from "@/components/human-gate";
import { ReportLocationButton } from "@/components/report-location";
import { Pearl, Skeleton, StatePanel, StatusBadge } from "@/components/ui";

const expectedAgents = [
  ["sales", "المبيعات والمنتجات", "Sales & product mix"],
  ["margin", "الهامش والتكلفة", "Margin & cost"],
  ["operations", "العمليات", "Operations"],
  ["customer", "صوت العميل", "Voice of customer"],
  ["anomaly", "رصد الشذوذ", "Anomaly detection"],
  ["critic", "الناقد", "Critic"],
] as const;

function normalizeAgents(agents?: Record<string, AgentState> | AgentState[]) {
  if (Array.isArray(agents)) return agents;
  if (agents && typeof agents === "object") return Object.entries(agents).map(([id, value]) => ({ id, ...value }));
  return [];
}

export function RunView({ runId }: { runId: string }) {
  const { locale, viewMode, cafeId } = useWorkspace();
  const Arrow = locale === "ar" ? ArrowLeft : ArrowRight;
  const queryClient = useQueryClient();
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [streamState, setStreamState] = useState<"connecting" | "open" | "closed" | "error">("connecting");
  const run = useQuery({ queryKey: ["run", runId], queryFn: () => api.run(runId), refetchInterval: (query) => livePolling(query.state.data?.status) });
  // Findings arrive mid-run, long after this query first resolves. Without its
  // own poll it kept serving the empty list it was seeded with at run creation.
  const findings = useQuery({ queryKey: ["findings", runId], queryFn: () => api.findings(runId), enabled: Boolean(run.data), retry: 1, refetchInterval: () => livePolling(run.data?.status) });
  useRunStatusSync(queryClient, runId, run.data?.status, cafeId);

  useEffect(() => {
    const stream = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`, { withCredentials: true });
    const onStatus = (event: MessageEvent) => {
      try {
        const next = JSON.parse(event.data) as RunEvent;
        setEvents((current) => [...current.slice(-39), next]);
        setStreamState("open");
        invalidateRunScoped(queryClient, runId, cafeId);
      } catch { setStreamState("error"); }
    };
    const onEnd = () => { setStreamState("closed"); stream.close(); invalidateRunScoped(queryClient, runId, cafeId); };
    stream.addEventListener("run.status", onStatus as EventListener);
    stream.addEventListener("end", onEnd);
    stream.onopen = () => setStreamState("open");
    stream.onerror = () => { if (stream.readyState === EventSource.CLOSED) setStreamState("closed"); else setStreamState("error"); };
    return () => stream.close();
  }, [runId, queryClient, cafeId]);

  const reportedAgents = normalizeAgents(run.data?.agents);
  const agentRows = useMemo(() => expectedAgents.map(([id, ar, en]) => ({ id, label: locale === "ar" ? ar : en, state: reportedAgents.find((agent) => agent.id === id || agent.name?.toLowerCase().includes(id)) })), [locale, reportedAgents]);
  const lastEvent = events.at(-1);

  return <div className="page-stack">
    <section className="page-heading"><div><Link className="back-link" href="/dashboard"><Arrow />{locale === "ar" ? "العودة للقصة" : "Back to story"}</Link><span className="eyebrow">{locale === "ar" ? "مراقبة تنفيذ حقيقية" : "Real execution monitor"}</span><h1>{locale === "ar" ? "تشغيل الوكلاء" : "Agent run"}</h1><code>{runId}</code></div>{run.data && <div className="heading-actions"><StatusBadge status={run.data.status} label={run.data.status} /><Pearl status={run.data.status} stage={run.data.stage} /></div>}</section>
    {run.isLoading ? <Skeleton lines={6} /> : run.isError ? <StatePanel kind="error" title={locale === "ar" ? "تعذر فتح التشغيل" : "Could not open run"} body={displayError(run.error, "API error")} /> : run.data && <>
      <section className="run-hero panel"><div className="run-hero-status"><div className="section-label"><Activity />{locale === "ar" ? "المرحلة الحالية" : "Current stage"}</div><strong>{lastEvent?.stage || run.data.stage || "—"}</strong><p>{lastEvent?.message || (locale === "ar" ? "لم ترسل قناة الأحداث رسالة مرحلية بعد." : "The event stream has not sent a stage message yet.")}</p></div><dl className="run-facts"><div><dt>{locale === "ar" ? "بدأ" : "Started"}</dt><dd>{formatDate(run.data.created_at, locale)}</dd></div><div><dt>{locale === "ar" ? "آخر تحديث" : "Updated"}</dt><dd>{formatDate(run.data.updated_at, locale)}</dd></div><div><dt>SSE</dt><dd><span className={`stream-dot stream-${streamState}`} />{streamState}</dd></div><div><dt>{locale === "ar" ? "الأخطاء" : "Errors"}</dt><dd>{run.data.error_count ?? "—"}</dd></div></dl></section>

      {streamState === "error" && <StatePanel kind="error" title={locale === "ar" ? "انقطع بث التقدم" : "Progress stream disconnected"} body={locale === "ar" ? "تظل الحالة محفوظة ويستمر تحديثها دوريًا من REST. أعد تحميل الصفحة لإعادة فتح SSE." : "Saved state still refreshes over REST. Reload to reopen SSE."} />}
      {(run.data.error_count ?? 0) > 0 && <div className="degraded-banner"><TriangleAlert /><div><strong>{locale === "ar" ? "التشغيل يحتوي على أخطاء" : "This run contains errors"}</strong><span>{locale === "ar" ? "قد تكتمل مراحل أخرى، لكن التقرير والنتائج يجب التعامل معها كحالة متأثرة." : "Other stages may complete, but reports and findings should be treated as degraded."}</span></div></div>}

      <section><div className="section-head"><div><span className="eyebrow">{locale === "ar" ? "تفرّع متوازٍ ثم تحقق" : "Parallel fan-out, then verification"}</span><h2>{locale === "ar" ? "حالات الوكلاء" : "Agent states"}</h2></div><small>{locale === "ar" ? "غير مُبلغ ≠ مكتمل" : "Unreported ≠ complete"}</small></div><div className="agent-grid">{agentRows.map(({ id, label, state }) => <article className={`agent-card agent-${state?.status || "unreported"}`} key={id}><div className="agent-symbol" aria-hidden="true"><span className="agent-eye eye-a" /><span className="agent-eye eye-b" /><span className="agent-expression" /></div><div><span className="technical-name">{id}</span><h3>{label}</h3></div><StatusBadge status={state?.status} label={state?.status || (locale === "ar" ? "غير مُبلغ" : "Unreported")} />{state?.message && <p>{state.message}</p>}{viewMode === "technical" && <dl><div><dt>{locale === "ar" ? "المحاولات" : "Attempts"}</dt><dd>{state?.attempts ?? "—"}</dd></div><div><dt>{locale === "ar" ? "الزمن" : "Runtime"}</dt><dd>{state?.runtime_ms ? `${state.runtime_ms} ms` : "—"}</dd></div></dl>}{state?.error && <div className="agent-error"><TriangleAlert />{state.error}</div>}</article>)}</div></section>

      {viewMode === "technical" && <section className="two-column"><article className="panel"><div className="panel-head"><div><span className="eyebrow">SSE</span><h2>{locale === "ar" ? "سجل الأحداث" : "Event ledger"}</h2></div><Radio /></div>{events.length ? <ol className="event-list">{events.slice().reverse().map((event, index) => <li key={event.event_id || `${event.at}-${index}`}><Clock3 /><div><strong>{event.stage || event.type}</strong><span>{event.message || event.status}</span><time>{formatDate(event.at, locale)}</time></div></li>)}</ol> : <StatePanel title={locale === "ar" ? "لا توجد إطارات SSE بعد" : "No SSE frames received yet"} />}</article><article className="panel"><div className="panel-head"><div><span className="eyebrow">REST</span><h2>{locale === "ar" ? "عقد التشغيل" : "Run contract"}</h2></div><Braces /></div><pre className="json-block">{JSON.stringify(run.data, null, 2)}</pre></article></section>}

      <section><div className="section-head"><div><span className="eyebrow"><ShieldCheck /> Critic</span><h2>{locale === "ar" ? "النتائج بعد التحقق" : "Post-Critic findings"}</h2></div></div>{findings.isLoading ? <Skeleton /> : findings.isError ? <StatePanel kind="error" title={locale === "ar" ? "تعذر تحميل النتائج" : "Could not load findings"} body={displayError(findings.error, "API error")} /> : findings.data?.length ? <div className="finding-list">{findings.data.map((finding) => <Link href={`/findings/${runId}/${finding.id}`} className="finding-row" key={finding.id}><div><span>{finding.analyst}</span><h3>{finding.title}</h3><p>{finding.claim}</p></div><div><span>{finding.evidence.length} {locale === "ar" ? "أدلة" : "evidence"}</span><Arrow /></div></Link>)}</div> : <StatePanel title={locale === "ar" ? "لا توجد نتائج كافية" : "No sufficient findings"} body={locale === "ar" ? "يمكن أن يكون هذا ناتجًا حقيقيًا لتشغيل متأثر أو أدلة غير كافية." : "This can be a real outcome of a degraded run or insufficient evidence."} />}</section>

      {/* The decision sits immediately under the findings it is about, so the
          human gate is reachable from the evidence rather than only from the
          report screen. */}
      <section className="panel human-gate-panel" id="human-gate">
        <div className="panel-head">
          <div><span className="eyebrow">HITL</span><h2>{locale === "ar" ? "بوابة القرار البشري" : "Human decision gate"}</h2></div>
          {run.data.status === "waiting_review" && <StatusBadge status="waiting_review" label={locale === "ar" ? "بانتظار قرارك" : "Awaiting your decision"} />}
        </div>
        <p>{locale === "ar" ? "راجع النتائج أعلاه ثم اقبل أو ارفض. لا ينتقل التقرير للتسليم دون قرار مخوّل، وكل قرار يُسجَّل مع تعليقه." : "Review the findings above, then accept or reject. Nothing moves to delivery without an authorized decision, and every decision is recorded with its comment."}</p>
        <HumanGate runId={runId} />
        <Link className="text-link" href={`/reports/${runId}`}>{locale === "ar" ? "افتح التقرير الكامل قبل القرار" : "Open the full report before deciding"}<Arrow /></Link>
        <ReportLocationButton runId={runId} status={run.data.status} />
      </section>
    </>}
  </div>;
}
