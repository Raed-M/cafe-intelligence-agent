"use client";

import { AlertTriangle, CheckCircle2, Database, LoaderCircle, Radio, Sparkles } from "lucide-react";
import { useState, type ReactNode } from "react";
import { useWorkspace } from "@/components/providers";

const liveStates = new Set(["running", "ingesting", "analyzing", "verifying", "queued"]);
const degradedStates = new Set(["failed", "error", "degraded", "partial", "rejected"]);
const cachedStates = new Set(["saved", "completed", "complete", "approved", "delivered", "paused", "manager_review", "owner_review"]);

export function statusKind(status?: string) {
  const normalized = status?.toLowerCase() ?? "unknown";
  if (liveStates.has(normalized)) return "live";
  if (degradedStates.has(normalized)) return "degraded";
  if (cachedStates.has(normalized)) return "cached";
  return "planned";
}

export function StatusBadge({ status, label }: { status?: string; label?: string }) {
  const { locale } = useWorkspace();
  const kind = statusKind(status);
  const names = locale === "ar"
    ? { live: "مباشر", cached: "محفوظ", planned: "غير متاح", degraded: "متأثر" }
    : { live: "Live", cached: "Cached", planned: "Unavailable", degraded: "Degraded" };
  const Icon = kind === "live" ? Radio : kind === "cached" ? CheckCircle2 : kind === "degraded" ? AlertTriangle : Database;
  return <span className={`status status-${kind}`}><Icon aria-hidden="true" />{label ?? names[kind]}</span>;
}

export function Pearl({ status, stage }: { status?: string; stage?: string }) {
  const { locale } = useWorkspace();
  const [open, setOpen] = useState(false);
  const kind = statusKind(status);
  return (
    <div className="pearl-wrap">
      <button className={`pearl pearl-${kind}`} type="button" onClick={() => setOpen((value) => !value)} aria-expanded={open} aria-label={locale === "ar" ? "تفاصيل حالة اللؤلؤة" : "Pearl status details"}>
        <span className="pearl-core"><Sparkles aria-hidden="true" /></span>
        <span className="pearl-orbit pearl-orbit-a" aria-hidden="true" />
        <span className="pearl-orbit pearl-orbit-b" aria-hidden="true" />
      </button>
      {open && <div className="pearl-note" role="status"><StatusBadge status={status} /><strong>{stage || status || (locale === "ar" ? "لا يوجد تشغيل حالي" : "No current run")}</strong><span>{locale === "ar" ? "تعكس اللؤلؤة حالة خط التحليل الفعلية." : "The pearl reflects the real pipeline state."}</span></div>}
    </div>
  );
}

export function StatePanel({ kind = "empty", title, body, action }: { kind?: "empty" | "loading" | "error"; title: string; body?: string; action?: ReactNode }) {
  const Icon = kind === "loading" ? LoaderCircle : kind === "error" ? AlertTriangle : Database;
  return <div className={`state-panel state-${kind}`} role={kind === "error" ? "alert" : "status"}><Icon className={kind === "loading" ? "spin" : ""} aria-hidden="true" /><div><strong>{title}</strong>{body && <p>{body}</p>}{action}</div></div>;
}

export function Skeleton({ lines = 3 }: { lines?: number }) {
  return <div className="skeleton" aria-label="Loading">{Array.from({ length: lines }, (_, index) => <span key={index} style={{ width: `${92 - index * 11}%` }} />)}</div>;
}
