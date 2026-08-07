"use client";

import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  Check,
  Database,
  FileText,
  GitBranch,
  Network,
  Radio,
  ScanSearch,
  ShieldCheck,
  UserRoundCheck,
} from "lucide-react";
import type { Run, RunEvent } from "@/lib/types";
import { formatDate } from "@/lib/format";

type Locale = "ar" | "en";
type StreamState = "connecting" | "open" | "closed" | "error";
type StepState = "complete" | "active" | "waiting" | "error" | "pending";

export const workflowSteps = [
  {
    id: "preflight",
    label: "Preflight",
    description: "Validate the café profile, date range, limits, and required connections.",
    icon: ScanSearch,
    stages: ["queued", "running", "preflight", "preflight_dataset", "guard_limits"],
  },
  {
    id: "ingestion",
    label: "Ingest & clean",
    description: "Parse registered files in parallel, reject invalid rows, and materialise versioned clean data.",
    icon: Database,
    stages: ["parse_source", "ingestion_fanin", "cleaning", "clean_and_materialise"],
  },
  {
    id: "analysis",
    label: "Analyst fan-out",
    description: "Run the eligible sales, margin, operations, customer, and anomaly analysts in parallel.",
    icon: GitBranch,
    stages: ["run_one_analyst", "analysis_fanin"],
  },
  {
    id: "cross_domain",
    label: "Cross-domain synthesis",
    description: "Relate grounded metrics from multiple analysts without inventing new measurements or bypassing evidence checks.",
    icon: Network,
    stages: ["cross_domain_synthesis"],
  },
  {
    id: "verification",
    label: "Critic verification",
    description: "Verify evidence, request targeted revisions when needed, then rank supported findings.",
    icon: ShieldCheck,
    stages: ["critic", "run_one_analyst_revision", "revision_fanin", "rank", "no_evidence", "findings"],
  },
  {
    id: "reporting",
    label: "Story & report",
    description: "Build grounded context, generate content, validate it, and assemble the decision report.",
    icon: FileText,
    stages: ["build_context", "content_agent", "validate_content", "increment_repair_attempts", "discard_invalid_content", "report_generator"],
  },
  {
    id: "approval",
    label: "Human gate",
    description: "Pause for manager and owner decisions, then deliver or stop and persist the outcome.",
    icon: UserRoundCheck,
    stages: ["human_gate", "manager_review", "owner_decision", "deliver", "stop_rejected", "persist_run", "completed"],
  },
] as const;

export function workflowIndexForStage(stage?: string, status?: string) {
  const normalized = (stage || "").toLowerCase();
  const matched = workflowSteps.findIndex((step) => step.stages.some((candidate) => normalized === candidate || normalized.includes(candidate)));
  if (matched >= 0) return matched;
  if (["waiting_review", "succeeded", "partial", "rejected"].includes(status || "")) return workflowSteps.length - 1;
  return 0;
}

export function workflowStepState(index: number, currentIndex: number, status?: string, stage?: string): StepState {
  const normalizedStatus = status || "";
  const terminalSuccess = ["succeeded", "partial", "completed", "saved"].includes(normalizedStatus) && stage === "completed";
  if (terminalSuccess) return "complete";
  if (index < currentIndex) return "complete";
  if (index > currentIndex) return "pending";
  if (["failed", "aborted", "rejected"].includes(normalizedStatus)) return "error";
  if (normalizedStatus === "waiting_review") return "waiting";
  return "active";
}

export function WorkflowMonitor({ run, locale }: { run: Run; locale: Locale }) {
  const queryClient = useQueryClient();
  const [eventLog, setEventLog] = useState<{ runId: string; items: RunEvent[] }>({ runId: run.id, items: [] });
  const [streamState, setStreamState] = useState<StreamState>("connecting");
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  useEffect(() => {
    const stream = new EventSource(`/api/runs/${encodeURIComponent(run.id)}/events`, { withCredentials: true });
    const onStatus = (event: MessageEvent) => {
      try {
        const next = JSON.parse(event.data) as RunEvent;
        setEventLog((current) => {
          const items = current.runId === run.id ? current.items : [];
          return items.some((item) => item.event_id && item.event_id === next.event_id) ? { runId: run.id, items } : { runId: run.id, items: [...items.slice(-39), next] };
        });
        setStreamState("open");
        queryClient.invalidateQueries({ queryKey: ["runs"] });
      } catch {
        setStreamState("error");
      }
    };
    const onEnd = () => {
      setStreamState("closed");
      stream.close();
      queryClient.invalidateQueries({ queryKey: ["runs"] });
    };
    stream.addEventListener("run.status", onStatus as EventListener);
    stream.addEventListener("end", onEnd);
    stream.onopen = () => setStreamState("open");
    stream.onerror = () => setStreamState(stream.readyState === EventSource.CLOSED ? "closed" : "error");
    return () => stream.close();
  }, [run.id, queryClient]);

  const events = eventLog.runId === run.id ? eventLog.items : [];
  const lastEvent = events.at(-1);
  const liveStage = lastEvent?.stage || run.stage || run.status;
  const liveStatus = lastEvent?.status || run.status;
  const currentIndex = workflowIndexForStage(liveStage, liveStatus);
  const inspectedIndex = selectedIndex ?? currentIndex;
  const inspected = workflowSteps[inspectedIndex];
  const eventForInspectedStep = [...events].reverse().find((event) => workflowIndexForStage(event.stage, event.status) === inspectedIndex);

  return <section className="workflow-monitor" aria-labelledby="workflow-title">
    <div className="workflow-head">
      <div>
        <span className="eyebrow"><Radio aria-hidden="true" /> SSE workflow</span>
        <h3 id="workflow-title">Live execution map</h3>
        <p>Choose a stage to inspect it. The highlighted stage follows real run events.</p>
      </div>
      <div className="workflow-live" aria-live="polite">
        <span className={`stream-dot stream-${streamState}`} />
        <span>{streamState === "open" ? "Live" : streamState === "connecting" ? "Connecting" : streamState === "closed" ? "Run stream closed" : "Stream disconnected"}</span>
      </div>
    </div>

    <ol className="workflow-track" aria-label="Run workflow stages">
      {workflowSteps.map((step, index) => {
        const state = workflowStepState(index, currentIndex, liveStatus, liveStage);
        const Icon = step.icon;
        return <li className={`workflow-step workflow-${state}`} key={step.id}>
          <button type="button" onClick={() => setSelectedIndex(index === currentIndex ? null : index)} aria-current={index === currentIndex ? "step" : undefined} aria-pressed={inspectedIndex === index}>
            <span className="workflow-node">{state === "complete" ? <Check aria-hidden="true" /> : <Icon aria-hidden="true" />}</span>
            <span><small>0{index + 1}</small><strong>{step.label}</strong><em>{state}</em></span>
          </button>
        </li>;
      })}
    </ol>

    <div className="workflow-detail" aria-live="polite">
      <div>
        <span className="technical-name">{inspected.id}</span>
        <h4>{inspected.label}</h4>
        <p>{inspected.description}</p>
      </div>
      <dl>
        <div><dt>Live backend stage</dt><dd><code>{eventForInspectedStep?.stage || (inspectedIndex === currentIndex ? liveStage : "Not reached")}</code></dd></div>
        <div><dt>Latest signal</dt><dd>{eventForInspectedStep?.message || (inspectedIndex === currentIndex ? `Run is ${liveStatus.replaceAll("_", " ")}.` : "No event received for this stage.")}</dd></div>
        <div><dt>Signal time</dt><dd>{eventForInspectedStep?.at ? formatDate(eventForInspectedStep.at, locale) : "—"}</dd></div>
      </dl>
      {selectedIndex !== null && selectedIndex !== currentIndex && <button className="text-link" type="button" onClick={() => setSelectedIndex(null)}><Radio />Follow live stage</button>}
    </div>
  </section>;
}
