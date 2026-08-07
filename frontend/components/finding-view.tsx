"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, ArrowRight, Calculator, Database, FileSearch, ShieldCheck } from "lucide-react";
import Link from "next/link";
import { api } from "@/lib/api";
import { displayError, formatValue } from "@/lib/format";
import { useWorkspace } from "@/components/providers";
import { Skeleton, StatePanel, StatusBadge } from "@/components/ui";

export function FindingView({ runId, findingId }: { runId: string; findingId: string }) {
  const { locale, viewMode } = useWorkspace();
  const Arrow = locale === "ar" ? ArrowLeft : ArrowRight;
  const findings = useQuery({ queryKey: ["findings", runId], queryFn: () => api.findings(runId) });
  const finding = findings.data?.find((candidate) => candidate.id === findingId);

  return <div className="page-stack">
    <section className="page-heading"><div><Link className="back-link" href={`/runs/${runId}`}><Arrow />{locale === "ar" ? "العودة للتشغيل" : "Back to run"}</Link><span className="eyebrow">{locale === "ar" ? "مطالبة ← حساب ← مصدر" : "Claim → calculation → source"}</span><h1>{locale === "ar" ? "مسار الدليل" : "Evidence trail"}</h1><code>{findingId}</code></div>{finding && <StatusBadge status={finding.approved ? "approved" : "rejected"} label={finding.approved ? (locale === "ar" ? "تحقق منها الناقد" : "Critic verified") : (locale === "ar" ? "غير معتمدة" : "Not approved")} />}</section>
    {findings.isLoading ? <Skeleton lines={7} /> : findings.isError ? <StatePanel kind="error" title={locale === "ar" ? "تعذر فتح النتيجة" : "Could not open finding"} body={displayError(findings.error, "API error")} /> : !finding ? <StatePanel title={locale === "ar" ? "النتيجة غير موجودة في هذا التشغيل" : "Finding is not present in this run"} body={locale === "ar" ? "ربما تغيرت النتائج بعد إعادة التشغيل أو لا يملك حسابك صلاحية رؤيتها." : "The finding may have changed after a rerun, or your account may not be authorized to view it."} /> : <>
      <article className="claim-card"><div className="section-label"><ShieldCheck />{finding.analyst}</div><h2>{finding.title}</h2><blockquote>{finding.claim}</blockquote><dl><div><dt>{locale === "ar" ? "النوع" : "Type"}</dt><dd>{finding.type || "—"}</dd></div><div><dt>{locale === "ar" ? "الثقة" : "Confidence"}</dt><dd>{finding.confidence === undefined ? "—" : `${formatValue(finding.confidence <= 1 ? finding.confidence * 100 : finding.confidence, locale)}%`}</dd></div><div><dt>{locale === "ar" ? "قرار الناقد" : "Critic decision"}</dt><dd>{finding.approved === undefined ? (locale === "ar" ? "لم يُبلغ" : "Unreported") : finding.approved ? (locale === "ar" ? "معتمد" : "Approved") : (locale === "ar" ? "مرفوض" : "Rejected")}</dd></div></dl></article>
      <section><div className="section-head"><div><span className="eyebrow">{locale === "ar" ? "قيم أعادها الخادم" : "Server-returned values"}</span><h2>{locale === "ar" ? "الحسابات المساندة" : "Supporting calculations"}</h2></div></div>{finding.evidence.length ? <div className="evidence-grid">{finding.evidence.map((evidence) => <article className="evidence-card" key={evidence.id}><div className="evidence-value"><Calculator /><strong>{formatValue(evidence.value, locale, evidence.unit)}</strong></div><h3>{evidence.metric_name}</h3><dl><div><dt>{locale === "ar" ? "الفترة" : "Period"}</dt><dd>{evidence.period_start || "—"} → {evidence.period_end || "—"}</dd></div><div><dt>{locale === "ar" ? "معرّف الأثر" : "Artifact ID"}</dt><dd><code>{evidence.artifact_id || "—"}</code></dd></div></dl><div className="source-chips">{evidence.source_names?.length ? evidence.source_names.map((source) => <Link key={source} href={`/data?source=${encodeURIComponent(source)}${evidence.artifact_id ? `&record=${encodeURIComponent(evidence.artifact_id)}` : ""}`}><Database />{source}</Link>) : <span>{locale === "ar" ? "لم يحدد الخادم مصدرًا" : "No source named by server"}</span>}</div></article>)}</div> : <StatePanel title={locale === "ar" ? "المطالبة بلا دليل مرفق" : "Claim has no attached evidence"} body={locale === "ar" ? "لا تعامل هذه المطالبة كحقيقة إلى أن يعيد الخادم سجلات الحساب والمصدر." : "Do not treat this claim as verified until the server returns calculation and source records."} />}</section>
      <div className="lineage-callout"><FileSearch /><div><strong>{locale === "ar" ? "البيانات الخام لا تتغير" : "Raw records stay immutable"}</strong><p>{locale === "ar" ? "روابط المصدر تفتح المستكشف. إذا أعاد API معرّف سجل صالحًا فسيعرض مقارنة الخام والمنظف وأسباب الإصلاح." : "Source links open the explorer. When the API returns a valid record ID, it shows raw versus cleaned values and repair reasons."}</p></div><Link className="secondary-button" href="/data">{locale === "ar" ? "افتح المستكشف" : "Open explorer"}<Arrow /></Link></div>
      {viewMode === "technical" && <section className="panel"><div className="panel-head"><div><span className="eyebrow">JSON</span><h2>{locale === "ar" ? "عقد النتيجة" : "Finding contract"}</h2></div></div><pre className="json-block">{JSON.stringify(finding, null, 2)}</pre></section>}
    </>}
  </div>;
}
