"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, PencilLine, Send, X } from "lucide-react";
import { FormEvent, useState } from "react";
import { api } from "@/lib/api";
import { displayError } from "@/lib/format";
import { invalidateRunScoped, livePolling } from "@/lib/run-state";
import { useWorkspace } from "@/components/providers";
import { StatePanel } from "@/components/ui";

export type ReviewAction = "submit" | "request_changes" | "approve" | "edit" | "reject";

const NEEDS_COMMENT: ReviewAction[] = ["request_changes", "edit", "reject"];

/** The approve/reject controls for the HITL gate.
 *
 *  Shared by the run page (shown directly under the post-critic findings, so
 *  the decision sits with the evidence it is about) and the report page. Both
 *  read the same `["run", runId]` / `["report", runId]` cache entries, so
 *  mounting it twice costs no extra requests and the two screens can never
 *  disagree about whose turn it is. */
export function HumanGate({ runId, showSteps = true }: { runId: string; showSteps?: boolean }) {
  const { locale, user, cafeId } = useWorkspace();
  const queryClient = useQueryClient();
  const [comment, setComment] = useState("");
  const [action, setAction] = useState<ReviewAction | null>(null);

  const run = useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.run(runId),
    retry: false,
    refetchInterval: (query) => livePolling(query.state.data?.status),
  });
  const report = useQuery({
    queryKey: ["report", runId],
    queryFn: () => api.report(runId),
    retry: false,
    refetchInterval: () => livePolling(run.data?.status),
  });

  const decision = useMutation({
    mutationFn: ({ type, note }: { type: ReviewAction; note: string }) =>
      type === "submit" || type === "request_changes"
        ? api.managerReview(runId, type, note)
        : api.ownerDecision(runId, type, note),
    onSuccess() {
      setAction(null);
      setComment("");
      invalidateRunScoped(queryClient, runId, cafeId);
    },
  });

  const capabilities = useQuery({ queryKey: ["capabilities"], queryFn: api.capabilities, retry: false, staleTime: 5 * 60_000 });

  const state = report.data?.state;
  const awaitingHuman = run.data?.status === "waiting_review";
  // The manager step is a manager's by default: owner + manager is two-person
  // review. An owner is only offered it when the deployment explicitly opted
  // out, which is the single-operator escape hatch.
  const ownerMayReview = user?.role === "owner" && capabilities.data?.owner_self_review === true;
  const canManagerReview = awaitingHuman && state === "manager_review" && (user?.role === "manager" || ownerMayReview);
  const canOwnerDecide = awaitingHuman && state === "owner_review" && user?.role === "owner";
  const actingAsManager = canManagerReview && user?.role === "owner";
  const blockedOwner = awaitingHuman && state === "manager_review" && user?.role === "owner" && !ownerMayReview;

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!action) return;
    if (NEEDS_COMMENT.includes(action) && !comment.trim()) return;
    decision.mutate({ type: action, note: comment.trim() });
  };

  const idleReason = !awaitingHuman
    ? (locale === "ar" ? "لم يتوقف هذا التشغيل عند بوابة القرار البشري." : "This run is not paused at the human decision gate.")
    : blockedOwner
      ? (locale === "ar" ? "هذه الخطوة للمدير: المراجعة على مرحلتين تتطلب شخصين. اعتمد حساب مدير من صفحة طلبات الوصول، أو شغّل WADDEHHA_ALLOW_OWNER_SELF_REVIEW=1 لتشغيل بمشغّل واحد." : "This step belongs to a manager: two-stage review is intentionally two people. Approve a manager account from Access requests, or set WADDEHHA_ALLOW_OWNER_SELF_REVIEW=1 for a single-operator setup.")
      : state === "manager_review"
        ? (locale === "ar" ? "المراجعة متاحة للمدير في مرحلة manager_review." : "Manager review is manager-only during manager_review.")
        : (locale === "ar" ? "القرار النهائي متاح للمالك فقط في مرحلة owner_review." : "The final decision is owner-only during owner_review.");

  return <>
    {showSteps && <ol className="approval-steps">
      <li className={state?.includes("manager") ? "current" : ""}><span>1</span><div><strong>{locale === "ar" ? "مراجعة المدير" : "Manager review"}</strong><small>{locale === "ar" ? "تعليق ثم إرسال للمالك أو طلب تعديل" : "Comment, submit, or request changes"}</small></div></li>
      <li className={state?.includes("owner") ? "current" : ""}><span>2</span><div><strong>{locale === "ar" ? "قرار المالك" : "Owner decision"}</strong><small>{locale === "ar" ? "موافقة، طلب تحرير، أو رفض" : "Approve, request edit, or reject"}</small></div></li>
      <li className={state === "approved" || state === "delivered" ? "current" : ""}><span>3</span><div><strong>{locale === "ar" ? "معتمد" : "Approved"}</strong><small>{locale === "ar" ? "التسليم عملية منفصلة" : "Delivery is a separate operation"}</small></div></li>
    </ol>}

    {actingAsManager && <p className="gate-note">{locale === "ar" ? "لا يوجد حساب مدير، لذلك تسجّل هذه الخطوة باسمك كمالك." : "No manager account is staffed, so this step is recorded against your owner account."}</p>}

    <div className="decision-buttons">
      {canManagerReview && <>
        <button type="button" className="secondary-button" onClick={() => setAction("request_changes")}><PencilLine />{locale === "ar" ? "اطلب تعديلًا" : "Request changes"}</button>
        <button type="button" className="primary-button" onClick={() => setAction("submit")}><Send />{locale === "ar" ? "قبول وإرسال للمالك" : "Accept & submit to owner"}</button>
      </>}
      {canOwnerDecide && <>
        <button type="button" className="secondary-button danger-outline" onClick={() => setAction("reject")}><X />{locale === "ar" ? "رفض" : "Reject"}</button>
        <button type="button" className="secondary-button" onClick={() => setAction("edit")}><PencilLine />{locale === "ar" ? "اطلب تحريرًا" : "Request edit"}</button>
        <button type="button" className="primary-button" onClick={() => setAction("approve")}><Check />{locale === "ar" ? "موافقة نهائية" : "Final approval"}</button>
      </>}
    </div>

    {!canManagerReview && !canOwnerDecide && <StatePanel title={locale === "ar" ? "لا يوجد قرار متاح لهذا الحساب الآن" : "No decision is available to this account now"} body={idleReason} />}
    {decision.isError && <StatePanel kind="error" title={locale === "ar" ? "لم يُحفظ القرار" : "Decision was not saved"} body={displayError(decision.error, "API error")} />}

    {action && <div className="modal-layer" role="dialog" aria-modal="true" aria-labelledby="decision-title">
      <button className="modal-scrim" type="button" onClick={() => setAction(null)} aria-label={locale === "ar" ? "إلغاء" : "Cancel"} />
      <form className="decision-dialog" onSubmit={submit}>
        <span className="eyebrow">{action}</span>
        <h2 id="decision-title">{locale === "ar" ? "سجّل قرارك" : "Record your decision"}</h2>
        <label htmlFor="review-comment">{locale === "ar" ? "التعليق" : "Comment"}{NEEDS_COMMENT.includes(action) && " *"}</label>
        <textarea id="review-comment" rows={5} value={comment} onChange={(event) => setComment(event.target.value)} placeholder={locale === "ar" ? "اكتب سببًا محددًا يساعد المراجع التالي…" : "Give the next reviewer a specific reason…"} required={NEEDS_COMMENT.includes(action)} />
        <div className="dialog-actions">
          <button className="secondary-button" type="button" onClick={() => setAction(null)}>{locale === "ar" ? "إلغاء" : "Cancel"}</button>
          <button className="primary-button" type="submit" disabled={decision.isPending}>{decision.isPending ? (locale === "ar" ? "جارٍ الحفظ…" : "Saving…") : (locale === "ar" ? "تأكيد القرار" : "Confirm decision")}</button>
        </div>
      </form>
    </div>}
  </>;
}
