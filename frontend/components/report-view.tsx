"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, ArrowRight, Check, FileText, LockKeyhole, MessageSquareText, PencilLine, Send, X } from "lucide-react";
import Link from "next/link";
import { FormEvent, useState } from "react";
import { api } from "@/lib/api";
import { displayError, formatDate } from "@/lib/format";
import { enhanceReportHtml } from "@/lib/report-format";
import { useWorkspace } from "@/components/providers";
import { Skeleton, StatePanel, StatusBadge } from "@/components/ui";

type ReviewAction = "submit" | "request_changes" | "approve" | "edit" | "reject";

export function ReportView({ runId }: { runId: string }) {
  const { locale, user } = useWorkspace();
  const Arrow = locale === "ar" ? ArrowLeft : ArrowRight;
  const queryClient = useQueryClient();
  const [tab, setTab] = useState<"full" | "whatsapp">("full");
  const [comment, setComment] = useState("");
  const [action, setAction] = useState<ReviewAction | null>(null);
  const report = useQuery({ queryKey: ["report", runId], queryFn: () => api.report(runId), retry: false });
  const run = useQuery({ queryKey: ["run", runId], queryFn: () => api.run(runId), retry: false });
  const canManagerReview = user?.role === "manager" && run.data?.status === "waiting_review" && report.data?.state === "manager_review";
  const canOwnerDecide = user?.role === "owner" && run.data?.status === "waiting_review" && report.data?.state === "owner_review";
  const decision = useMutation({
    mutationFn: ({ type, note }: { type: ReviewAction; note: string }) => type === "submit" || type === "request_changes" ? api.managerReview(runId, type, note) : api.ownerDecision(runId, type, note),
    onSuccess() { setAction(null); setComment(""); queryClient.invalidateQueries({ queryKey: ["report", runId] }); queryClient.invalidateQueries({ queryKey: ["run", runId] }); },
  });

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!action) return;
    if (["request_changes", "edit", "reject"].includes(action) && !comment.trim()) return;
    decision.mutate({ type: action, note: comment.trim() });
  };

  return <div className="page-stack report-page">
    <section className="page-heading"><div><Link className="back-link" href={`/runs/${runId}`}><Arrow />{locale === "ar" ? "العودة للتشغيل" : "Back to run"}</Link><span className="eyebrow">{locale === "ar" ? "مراجعة المدير ← قرار المالك" : "Manager review → owner decision"}</span><h1>{locale === "ar" ? "التقرير ثنائي اللغة" : "Bilingual report"}</h1><code>{runId}</code></div>{report.data && <div className="heading-actions"><StatusBadge status={report.data.state} label={report.data.state} /><span className="generated-at">{formatDate(report.data.generated_at, locale)}</span></div>}</section>
    {report.isLoading ? <Skeleton lines={8} /> : report.isError ? <StatePanel kind="error" title={locale === "ar" ? "التقرير غير متاح" : "Report unavailable"} body={displayError(report.error, locale === "ar" ? "قد لا يكون التشغيل وصل إلى مرحلة التقرير بعد." : "The run may not have reached report generation yet.")} /> : report.data && <>
      <div className="report-toolbar"><div className="segmented" role="tablist" aria-label={locale === "ar" ? "صيغة التقرير" : "Report format"}><button role="tab" aria-selected={tab === "full"} className={tab === "full" ? "selected" : ""} onClick={() => setTab("full")}><FileText />{locale === "ar" ? "التقرير الكامل" : "Full report"}</button><button role="tab" aria-selected={tab === "whatsapp"} className={tab === "whatsapp" ? "selected" : ""} onClick={() => setTab("whatsapp")}><MessageSquareText />WhatsApp</button></div><div className="computed-lock"><LockKeyhole />{locale === "ar" ? "الأرقام المحسوبة للقراءة فقط" : "Computed figures are read-only"}</div></div>
      <section className="report-layout">
        <article className="report-canvas">{tab === "full" ? report.data.html ? <iframe title={locale === "ar" ? "معاينة التقرير" : "Readable report preview"} sandbox="" srcDoc={enhanceReportHtml(report.data.html)} /> : <StatePanel title={locale === "ar" ? "لم يعد الخادم محتوى HTML" : "The server returned no HTML"} /> : report.data.whatsapp_summary ? <div className="whatsapp-preview"><span className="wa-label">WhatsApp · {locale === "ar" ? "معاينة فقط" : "preview only"}</span><p>{report.data.whatsapp_summary}</p><small>{locale === "ar" ? "الإرسال الفعلي يحتاج تكامل WhatsApp Business؛ لا يوجد إرسال من هذه الشاشة." : "Real delivery requires a WhatsApp Business integration; this screen does not send it."}</small></div> : <StatePanel title={locale === "ar" ? "لا يوجد ملخص واتساب" : "No WhatsApp summary"} />}</article>
        <aside className="approval-panel"><span className="eyebrow">HITL</span><h2>{locale === "ar" ? "بوابة القرار البشري" : "Human decision gate"}</h2><p>{locale === "ar" ? "التقرير لا ينتقل للتسليم إلا عبر الأدوار المخولة. كل قرار وتعليق يرسل مباشرة إلى API." : "The report cannot move to delivery without an authorized role. Every decision and comment is sent directly to the API."}</p><ol className="approval-steps"><li className={report.data.state?.includes("manager") ? "current" : ""}><span>1</span><div><strong>{locale === "ar" ? "مراجعة المدير" : "Manager review"}</strong><small>{locale === "ar" ? "تعليق ثم إرسال للمالك أو طلب تعديل" : "Comment, submit, or request changes"}</small></div></li><li className={report.data.state?.includes("owner") ? "current" : ""}><span>2</span><div><strong>{locale === "ar" ? "قرار المالك" : "Owner decision"}</strong><small>{locale === "ar" ? "موافقة، طلب تحرير، أو رفض" : "Approve, request edit, or reject"}</small></div></li><li className={report.data.state === "approved" || report.data.state === "delivered" ? "current" : ""}><span>3</span><div><strong>{locale === "ar" ? "معتمد" : "Approved"}</strong><small>{locale === "ar" ? "التسليم عملية منفصلة" : "Delivery is a separate operation"}</small></div></li></ol>
          <div className="decision-buttons">{canManagerReview && <><button type="button" className="secondary-button" onClick={() => setAction("request_changes")}><PencilLine />{locale === "ar" ? "اطلب تعديلًا" : "Request changes"}</button><button type="button" className="secondary-button" onClick={() => setAction("submit")}><Send />{locale === "ar" ? "أرسل للمالك" : "Submit to owner"}</button></>}{canOwnerDecide && <><button type="button" className="secondary-button danger-outline" onClick={() => setAction("reject")}><X />{locale === "ar" ? "رفض" : "Reject"}</button><button type="button" className="secondary-button" onClick={() => setAction("edit")}><PencilLine />{locale === "ar" ? "اطلب تحريرًا" : "Request edit"}</button><button type="button" className="primary-button" onClick={() => setAction("approve")}><Check />{locale === "ar" ? "موافقة نهائية" : "Final approval"}</button></>}</div>{!canManagerReview && !canOwnerDecide && <StatePanel title={locale === "ar" ? "لا يوجد قرار متاح لهذا الحساب الآن" : "No decision is available to this account now"} body={locale === "ar" ? "المراجعة للمدير فقط في مرحلة manager_review، والقرار النهائي للمالك فقط في owner_review." : "Manager review is manager-only in manager_review; final decision is owner-only in owner_review."} />}</aside>
      </section>
      {decision.isError && <StatePanel kind="error" title={locale === "ar" ? "لم يُحفظ القرار" : "Decision was not saved"} body={displayError(decision.error, "API error")} />}
    </>}
    {action && <div className="modal-layer" role="dialog" aria-modal="true" aria-labelledby="decision-title"><button className="modal-scrim" type="button" onClick={() => setAction(null)} aria-label={locale === "ar" ? "إلغاء" : "Cancel"} /><form className="decision-dialog" onSubmit={submit}><span className="eyebrow">{action}</span><h2 id="decision-title">{locale === "ar" ? "سجّل قرارك" : "Record your decision"}</h2><label htmlFor="review-comment">{locale === "ar" ? "التعليق" : "Comment"}{["request_changes", "edit", "reject"].includes(action) && " *"}</label><textarea id="review-comment" rows={5} value={comment} onChange={(event) => setComment(event.target.value)} placeholder={locale === "ar" ? "اكتب سببًا محددًا يساعد المراجع التالي…" : "Give the next reviewer a specific reason…"} required={["request_changes", "edit", "reject"].includes(action)} /><div className="dialog-actions"><button className="secondary-button" type="button" onClick={() => setAction(null)}>{locale === "ar" ? "إلغاء" : "Cancel"}</button><button className="primary-button" type="submit" disabled={decision.isPending}>{decision.isPending ? (locale === "ar" ? "جارٍ الحفظ…" : "Saving…") : (locale === "ar" ? "تأكيد القرار" : "Confirm decision")}</button></div></form></div>}
  </div>;
}
