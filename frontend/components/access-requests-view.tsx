"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Clock3, Mail, ShieldCheck, UserRoundCheck, X } from "lucide-react";
import { api } from "@/lib/api";
import { displayError, formatDate } from "@/lib/format";
import { useWorkspace } from "@/components/providers";
import { Skeleton, StatePanel, StatusBadge } from "@/components/ui";

export function AccessRequestsView() {
  const { user, locale } = useWorkspace();
  const queryClient = useQueryClient();
  const [status, setStatus] = useState<"pending" | "approved" | "rejected">("pending");
  const requests = useQuery({ queryKey: ["access-requests", status], queryFn: () => api.accessRequests(status), enabled: user?.role === "owner" });
  const decision = useMutation({
    mutationFn: ({ id, value }: { id: string; value: "approve" | "reject" }) => api.decideAccess(id, value),
    onSuccess() { queryClient.invalidateQueries({ queryKey: ["access-requests"] }); },
  });

  if (user?.role !== "owner") return <StatePanel kind="error" title="Owner access required" body="Only the owner can review account requests." />;
  return <div className="page-stack">
    <section className="page-heading"><div><span className="eyebrow">Owner control</span><h1>Access requests</h1><p>Approve or reject Manager and Employee sign-up requests.</p></div><span className="request-count"><UserRoundCheck />{requests.data?.length ?? 0} {status}</span></section>
    <div className="segmented request-tabs" aria-label="Request status">{(["pending", "approved", "rejected"] as const).map((value) => <button key={value} type="button" className={status === value ? "selected" : ""} onClick={() => setStatus(value)}>{value}</button>)}</div>
    {decision.isError && <StatePanel kind="error" title="Could not save decision" body={displayError(decision.error, "Try again.")} />}
    {requests.isLoading ? <Skeleton lines={5} /> : requests.isError ? <StatePanel kind="error" title="Could not load access requests" body={displayError(requests.error, "API error")} /> : requests.data?.length ? <section className="request-list">{requests.data.map((request) => <article className="request-card" key={request.id}><div className="request-avatar"><span>{request.display_name.slice(0, 1).toUpperCase()}</span></div><div className="request-identity"><div><h2>{request.display_name}</h2><StatusBadge status={request.status} label={request.status} /></div><p><Mail />{request.email}</p><p><ShieldCheck />Requested role: <strong>{request.requested_role}</strong></p><small><Clock3 />Sent {formatDate(request.created_at, locale)}</small></div>{status === "pending" && <div className="request-actions"><button className="secondary-button danger-button" type="button" disabled={decision.isPending} onClick={() => decision.mutate({ id: request.id, value: "reject" })}><X />Reject</button><button className="primary-button" type="button" disabled={decision.isPending} onClick={() => decision.mutate({ id: request.id, value: "approve" })}><Check />Approve</button></div>}</article>)}</section> : <StatePanel title={`No ${status} requests`} body={status === "pending" ? "New sign-up requests will appear here for your decision." : "There are no requests in this state."} />}
  </div>;
}
