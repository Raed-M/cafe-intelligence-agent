"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, ArrowRight, Database, FileCheck2, Play, RefreshCw, ShieldCheck, TriangleAlert } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { displayError, formatDate, formatValue } from "@/lib/format";
import type { Evidence, Finding } from "@/lib/types";
import { useWorkspace } from "@/components/providers";
import { Pearl, Skeleton, StatePanel, StatusBadge } from "@/components/ui";
import { WorkflowMonitor } from "@/components/workflow-monitor";

const kpiDefinitions = [
  { ar: "صافي الإيراد", en: "Net revenue", terms: ["net revenue", "revenue", "صافي الإيراد", "المبيعات"] },
  { ar: "الربح الإجمالي", en: "Gross profit", terms: ["gross profit", "margin", "الربح الإجمالي", "الهامش"] },
  { ar: "التحويل", en: "Conversion", terms: ["conversion", "التحويل"] },
  { ar: "تكلفة الهدر", en: "Waste cost", terms: ["waste", "الهدر"] },
  { ar: "تكلفة العمل", en: "Labour cost", terms: ["labour", "labor", "staffing", "تكلفة العمل"] },
];

function matchingEvidence(findings: Finding[], terms: string[]) {
  return findings.flatMap((finding) => finding.evidence || []).find((evidence) => terms.some((term) => evidence.metric_name?.toLowerCase().includes(term)));
}

function KpiCard({ label, evidence, locale }: { label: string; evidence?: Evidence; locale: "ar" | "en" }) {
  return <article className="kpi-card"><span>{label}</span><strong>{formatValue(evidence?.value, locale, evidence?.unit)}</strong>{evidence ? <small>{evidence.metric_name}</small> : <small>{locale === "ar" ? "لم تُرجع الواجهة الخلفية هذا المؤشر" : "This metric was not returned by the backend"}</small>}</article>;
}

export function DashboardView() {
  const { locale, cafeId, cafe, viewMode, user } = useWorkspace();
  const router = useRouter();
  const queryClient = useQueryClient();
  const runs = useQuery({ queryKey: ["runs", cafeId], queryFn: () => api.runs(cafeId, 10), enabled: Boolean(cafeId), refetchInterval: 15_000 });
  const latest = runs.data?.[0];
  const findings = useQuery({ queryKey: ["findings", latest?.id], queryFn: () => api.findings(latest!.id), enabled: Boolean(latest?.id), retry: 1 });
  const report = useQuery({ queryKey: ["report", latest?.id], queryFn: () => api.report(latest!.id), enabled: Boolean(latest?.id), retry: false });
  const sources = useQuery({ queryKey: ["sources", cafeId], queryFn: () => api.sources(cafeId), enabled: Boolean(cafeId), retry: 1 });
  const canRun = user?.role === "owner" || user?.role === "manager";
  const start = useMutation({
    mutationFn: () => api.startRun(cafeId),
    onSuccess(run) { queryClient.invalidateQueries({ queryKey: ["runs", cafeId] }); router.push(`/runs/${run.id}`); },
  });
  const Arrow = locale === "ar" ? ArrowLeft : ArrowRight;
  const sourceFailures = sources.data?.filter((source) => ["failed", "error", "degraded", "invalid"].includes(source.status.toLowerCase())) ?? [];
  const story = report.data?.whatsapp_summary || findings.data?.[0]?.claim;

  return <div className="page-stack">
    <section className="page-heading" id="weekly-story">
      <div><span className="eyebrow">{locale === "ar" ? "وضوح أسبوعي · قرار موثّق" : "Weekly clarity · traceable decision"}</span><h1>{locale === "ar" ? "القصة الأسبوعية" : "Weekly story"}</h1><p>{cafe?.name || "—"}{latest?.analysis_period ? ` · ${typeof latest.analysis_period === "string" ? latest.analysis_period : `${latest.analysis_period.start ?? ""} — ${latest.analysis_period.end ?? ""}`}` : ""}</p></div>
      <div className="heading-actions"><StatusBadge status={latest?.status} /><button className="primary-button" type="button" disabled={!canRun || !cafeId || start.isPending} onClick={() => start.mutate()}><Play />{start.isPending ? (locale === "ar" ? "بدء التشغيل…" : "Starting…") : (locale === "ar" ? "تشغيل الأسبوع الآن" : "Run this week")}</button></div>
    </section>
    {start.isError && <StatePanel kind="error" title={locale === "ar" ? "لم يبدأ التشغيل" : "Run did not start"} body={displayError(start.error, locale === "ar" ? "تحقق من اتصال API وصلاحيات الحساب." : "Check the API connection and account permissions.")} />}

    <section className="story-grid" aria-label={locale === "ar" ? "ملخص الأسبوع" : "Week summary"}>
      <article className="story-card"><div className="story-copy"><div className="section-label"><ShieldCheck />{locale === "ar" ? "الخلاصة المتحققة" : "Verified narrative"}</div>{runs.isLoading || findings.isLoading ? <Skeleton lines={4} /> : story ? <blockquote>{story}</blockquote> : <StatePanel title={locale === "ar" ? "لا توجد قصة أسبوعية بعد" : "No weekly story yet"} body={locale === "ar" ? "ابدأ تشغيلًا حقيقيًا. لن نعرض استنتاجًا قبل أن يعيده الخادم مع الأدلة." : "Start a real run. No conclusion appears until the server returns it with evidence."} />}{latest && <div className="story-meta"><span>{locale === "ar" ? "آخر تحديث" : "Updated"}: {formatDate(latest.updated_at, locale)}</span><span>{latest.findings_count ?? findings.data?.length ?? 0} {locale === "ar" ? "نتيجة" : "findings"}</span></div>}</div><Pearl status={latest?.status} stage={latest?.stage} /></article>
      <aside className="integrity-card"><span className="section-label"><Database />{locale === "ar" ? "سلامة المصادر" : "Source integrity"}</span>{sources.isLoading ? <Skeleton lines={3} /> : sources.isError ? <StatePanel kind="error" title={locale === "ar" ? "تعذّر فحص المصادر" : "Could not inspect sources"} body={displayError(sources.error, "API error")} /> : sources.data?.length ? <><strong>{sources.data.length - sourceFailures.length}/{sources.data.length}</strong><p>{locale === "ar" ? "مصادر لم تُبلغ عن فشل" : "sources without a reported failure"}</p>{sourceFailures.length > 0 && <div className="degraded-note"><TriangleAlert />{locale === "ar" ? `${sourceFailures.length} مصدر متأثر؛ النتائج قد تكون جزئية.` : `${sourceFailures.length} source(s) degraded; results may be partial.`}</div>}<Link className="text-link" href="/data">{locale === "ar" ? "افتح سجل البيانات" : "Open data ledger"}<Arrow /></Link></> : <StatePanel title={locale === "ar" ? "لم يسجل الخادم أي مصدر" : "No sources registered"} />}</aside>
    </section>

    <section><div className="section-head"><div><span className="eyebrow">{locale === "ar" ? "مؤشرات من الأدلة فقط" : "Evidence-only indicators"}</span><h2>{locale === "ar" ? "صورة الأداء" : "Performance readout"}</h2></div><small>{locale === "ar" ? "تعرض الشرطة الطويلة قيمة غير متاحة، وليست صفرًا." : "An em dash means unavailable, not zero."}</small></div><div className="kpi-strip">{kpiDefinitions.map((definition) => <KpiCard key={definition.en} label={locale === "ar" ? definition.ar : definition.en} evidence={matchingEvidence(findings.data ?? [], definition.terms)} locale={locale} />)}</div></section>

    <section className={`two-column dashboard-control-grid ${viewMode === "technical" ? "technical-dashboard-grid" : ""}`}>
      <article className="panel" id="agents"><div className="panel-head"><div><span className="eyebrow">{locale === "ar" ? "خط التحليل" : "Analysis pipeline"}</span><h2>{locale === "ar" ? "التشغيل الحالي" : "Current run"}</h2></div>{latest && <Link className="secondary-button" href={`/runs/${latest.id}`}>{locale === "ar" ? "افتح التشغيل" : "Open run"}<Arrow /></Link>}</div>{runs.isLoading ? <Skeleton lines={5} /> : latest ? <div className="run-summary"><div className="run-id"><code>{latest.id}</code><StatusBadge status={latest.status} label={latest.status} /></div><div className="stage-track"><span className="stage-track-fill" style={{ transform: `scaleX(${latest.status === "completed" || latest.status === "saved" ? 1 : 0.48})` }} /></div><dl><div><dt>{locale === "ar" ? "المرحلة" : "Stage"}</dt><dd>{latest.stage || "—"}</dd></div><div><dt>{locale === "ar" ? "حالة التقرير" : "Report state"}</dt><dd>{latest.report_state || "—"}</dd></div><div><dt>{locale === "ar" ? "الأخطاء" : "Errors"}</dt><dd>{latest.error_count ?? "—"}</dd></div></dl>{viewMode === "technical" && <WorkflowMonitor run={latest} locale={locale} />}</div> : <StatePanel title={locale === "ar" ? "لا يوجد تشغيل لهذا المقهى" : "No run for this café"} body={canRun ? (locale === "ar" ? "استخدم زر تشغيل الأسبوع لإنشاء أول تشغيل." : "Use Run this week to create the first run.") : (locale === "ar" ? "صلاحية حسابك للقراءة فقط." : "Your account is read-only.")} />}</article>

      <article className="panel" id="reports"><div className="panel-head"><div><span className="eyebrow">{locale === "ar" ? "قرار بشري إلزامي" : "Required human decision"}</span><h2>{locale === "ar" ? "التقرير والموافقة" : "Report & approval"}</h2></div>{latest && report.data && <Link className="secondary-button" href={`/reports/${latest.id}`}>{locale === "ar" ? "راجع التقرير" : "Review report"}<Arrow /></Link>}</div>{report.isLoading ? <Skeleton lines={4} /> : report.data ? <div className="report-preview"><div><FileCheck2 /><StatusBadge status={report.data.state} label={report.data.state} /></div><p>{report.data.whatsapp_summary || (locale === "ar" ? "التقرير موجود دون ملخص واتساب." : "Report available without a WhatsApp summary.")}</p><small>{locale === "ar" ? "لا يتم التسليم قبل اكتمال المراجعة والموافقة." : "Delivery is blocked until review and approval are complete."}</small></div> : <StatePanel title={locale === "ar" ? "لا يوجد تقرير قابل للمراجعة" : "No report ready for review"} body={report.isError ? displayError(report.error, "API error") : undefined} />}</article>
    </section>

    <section><div className="section-head"><div><span className="eyebrow">{locale === "ar" ? "كل مطالبة لها مسار" : "Every claim has a trail"}</span><h2>{locale === "ar" ? "النتائج والأدلة" : "Findings & evidence"}</h2></div></div>{findings.isLoading ? <Skeleton lines={4} /> : findings.isError ? <StatePanel kind="error" title={locale === "ar" ? "تعذّر تحميل النتائج" : "Could not load findings"} body={displayError(findings.error, "API error")} /> : findings.data?.length ? <div className="finding-grid">{findings.data.slice(0, 6).map((finding) => <Link className="finding-card" key={finding.id} href={`/findings/${latest!.id}/${finding.id}`}><div><span className="analyst-name">{finding.analyst}</span>{finding.approved !== undefined && <span className={`critic-mark ${finding.approved ? "approved" : "rejected"}`}>{finding.approved ? (locale === "ar" ? "تحقق" : "Verified") : (locale === "ar" ? "غير معتمد" : "Not approved")}</span>}</div><h3>{finding.title}</h3><p>{finding.claim}</p><footer><span>{finding.evidence?.length ?? 0} {locale === "ar" ? "أدلة" : "evidence items"}</span><Arrow /></footer></Link>)}</div> : <StatePanel title={locale === "ar" ? "لا توجد نتائج متحققة" : "No verified findings"} body={locale === "ar" ? "قد يكون التشغيل لم يبدأ، أو انتهى دون أدلة كافية. هذه ليست نتيجة نجاح فارغة." : "The run may not have started, or completed without sufficient evidence. This is not presented as a successful result."} action={latest ? <Link className="text-link" href={`/runs/${latest.id}`}><RefreshCw />{locale === "ar" ? "افحص أخطاء التشغيل" : "Inspect run errors"}</Link> : undefined} />}</section>
  </div>;
}
