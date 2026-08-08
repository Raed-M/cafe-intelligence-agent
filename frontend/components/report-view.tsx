"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, ArrowRight, FileText, LockKeyhole, MessageSquareText } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { api } from "@/lib/api";
import { displayError, formatDate } from "@/lib/format";
import { enhanceReportHtml } from "@/lib/report-format";
import { livePolling } from "@/lib/run-state";
import { useWorkspace } from "@/components/providers";
import { HumanGate } from "@/components/human-gate";
import { ReportLocationButton } from "@/components/report-location";
import { Skeleton, StatePanel, StatusBadge } from "@/components/ui";

export function ReportView({ runId }: { runId: string }) {
  const { locale } = useWorkspace();
  const Arrow = locale === "ar" ? ArrowLeft : ArrowRight;
  const [tab, setTab] = useState<"full" | "whatsapp">("full");
  const run = useQuery({ queryKey: ["run", runId], queryFn: () => api.run(runId), retry: false, refetchInterval: (query) => livePolling(query.state.data?.status) });
  // An owner decision resumes the graph, which rewrites the report state and
  // can regenerate the HTML, so this cannot be a fetch-once query either.
  const report = useQuery({ queryKey: ["report", runId], queryFn: () => api.report(runId), retry: false, refetchInterval: () => livePolling(run.data?.status) });

  return <div className="page-stack report-page">
    <section className="page-heading"><div><Link className="back-link" href={`/runs/${runId}`}><Arrow />{locale === "ar" ? "العودة للتشغيل" : "Back to run"}</Link><span className="eyebrow">{locale === "ar" ? "مراجعة المدير ← قرار المالك" : "Manager review → owner decision"}</span><h1>{locale === "ar" ? "التقرير ثنائي اللغة" : "Bilingual report"}</h1><code>{runId}</code></div>{report.data && <div className="heading-actions"><StatusBadge status={report.data.state} label={report.data.state} /><span className="generated-at">{formatDate(report.data.generated_at, locale)}</span></div>}</section>
    {report.isLoading ? <Skeleton lines={8} /> : report.isError ? <StatePanel kind="error" title={locale === "ar" ? "التقرير غير متاح" : "Report unavailable"} body={displayError(report.error, locale === "ar" ? "قد لا يكون التشغيل وصل إلى مرحلة التقرير بعد." : "The run may not have reached report generation yet.")} /> : report.data && <>
      <div className="report-toolbar"><div className="segmented" role="tablist" aria-label={locale === "ar" ? "صيغة التقرير" : "Report format"}><button role="tab" aria-selected={tab === "full"} className={tab === "full" ? "selected" : ""} onClick={() => setTab("full")}><FileText />{locale === "ar" ? "التقرير الكامل" : "Full report"}</button><button role="tab" aria-selected={tab === "whatsapp"} className={tab === "whatsapp" ? "selected" : ""} onClick={() => setTab("whatsapp")}><MessageSquareText />WhatsApp</button></div><div className="computed-lock"><LockKeyhole />{locale === "ar" ? "الأرقام المحسوبة للقراءة فقط" : "Computed figures are read-only"}</div></div>
      <section className="report-layout">
        <article className="report-canvas">{tab === "full" ? report.data.html ? <iframe title={locale === "ar" ? "معاينة التقرير" : "Readable report preview"} sandbox="" srcDoc={enhanceReportHtml(report.data.html)} /> : <StatePanel title={locale === "ar" ? "لم يعد الخادم محتوى HTML" : "The server returned no HTML"} /> : report.data.whatsapp_summary ? <div className="whatsapp-preview"><span className="wa-label">WhatsApp · {locale === "ar" ? "معاينة فقط" : "preview only"}</span><p>{report.data.whatsapp_summary}</p><small>{locale === "ar" ? "الإرسال الفعلي يحتاج تكامل WhatsApp Business؛ لا يوجد إرسال من هذه الشاشة." : "Real delivery requires a WhatsApp Business integration; this screen does not send it."}</small></div> : <StatePanel title={locale === "ar" ? "لا يوجد ملخص واتساب" : "No WhatsApp summary"} />}</article>
        <aside className="approval-panel"><span className="eyebrow">HITL</span><h2>{locale === "ar" ? "بوابة القرار البشري" : "Human decision gate"}</h2><p>{locale === "ar" ? "التقرير لا ينتقل للتسليم إلا عبر الأدوار المخولة. كل قرار وتعليق يرسل مباشرة إلى API." : "The report cannot move to delivery without an authorized role. Every decision and comment is sent directly to the API."}</p>
          <HumanGate runId={runId} />
          <ReportLocationButton runId={runId} status={run.data?.status} />
        </aside>
      </section>
    </>}
  </div>;
}
