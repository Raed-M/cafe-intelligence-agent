import type { AccessRequest, AiConnectionTest, AiSettings, AiSettingsPayload, BrowserFilePayload, Cafe, ChatAnswer, Conversation, DataProcessResult, Finding, Lineage, Report, Run, SourceSummary, User } from "@/lib/types";

export class ApiError extends Error {
  constructor(message: string, public readonly status: number, public readonly code = "request_failed", public readonly requestId?: string) { super(message); }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...init, credentials: "include", headers: { "Content-Type": "application/json", ...init?.headers } });
  if (response.status === 204) return undefined as T;
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = data?.error ?? data;
    throw new ApiError(error?.message || `Request failed (${response.status})`, response.status, error?.code, error?.request_id);
  }
  return data as T;
}

const query = (values: Record<string, string | number | undefined>) => {
  const params = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => { if (value !== undefined && value !== "") params.set(key, String(value)); });
  return params.size ? `?${params.toString()}` : "";
};

const items = <T>(payload: unknown): T[] => {
  if (Array.isArray(payload)) return payload as T[];
  if (payload && typeof payload === "object") {
    const value = (payload as Record<string, unknown>).items;
    return Array.isArray(value) ? (value as T[]) : [];
  }
  return [];
};

const member = <T>(payload: unknown, key: string): T => {
  if (payload && typeof payload === "object" && key in payload) return (payload as Record<string, unknown>)[key] as T;
  throw new ApiError(`API response is missing '${key}'.`, 502, "invalid_response");
};

export const api = {
  login: (email: string, password: string) => request<{ user: User }>("/api/auth/login", { method: "POST", body: JSON.stringify({ email, password }) }),
  signup: (payload: { display_name: string; email: string; password: string; requested_role: "manager" | "employee" }) => request<{ access_request: AccessRequest }>("/api/auth/signup", { method: "POST", body: JSON.stringify(payload) }),
  logout: () => request<void>("/api/auth/logout", { method: "POST" }),
  me: async () => (await request<{ user: User }>("/api/auth/me")).user,
  cafes: async () => items<Cafe>(await request("/api/cafes")),
  sources: async (cafeId: string) => items<SourceSummary>(await request(`/api/cafes/${encodeURIComponent(cafeId)}/sources`)),
  data: (cafeId: string, source: string, cursor?: string) => request<{ items: Array<Record<string, unknown>>; next_cursor?: string }>(`/api/cafes/${encodeURIComponent(cafeId)}/data/${encodeURIComponent(source)}${query({ limit: 50, cursor })}`),
  lineage: (cafeId: string, source: string, recordId: string) => request<Lineage>(`/api/cafes/${encodeURIComponent(cafeId)}/data/${encodeURIComponent(source)}/${encodeURIComponent(recordId)}/lineage`),
  processData: (cafeId: string, files: BrowserFilePayload[]) => request<DataProcessResult>(`/api/cafes/${encodeURIComponent(cafeId)}/data/process`, { method: "POST", body: JSON.stringify({ files }) }),
  runs: async (cafeId: string, limit = 10) => items<Run>(await request(`/api/runs${query({ cafe_id: cafeId, limit })}`)),
  startRun: async (cafeId: string) => member<Run>(await request("/api/runs", { method: "POST", body: JSON.stringify({ cafe_id: cafeId }) }), "run"),
  run: async (runId: string) => member<Run>(await request(`/api/runs/${encodeURIComponent(runId)}`), "run"),
  findings: async (runId: string) => items<Finding>(await request(`/api/runs/${encodeURIComponent(runId)}/findings`)),
  report: (runId: string) => request<Report>(`/api/runs/${encodeURIComponent(runId)}/report`),
  managerReview: (runId: string, decision: "submit" | "request_changes", comment: string) => request<Report>(`/api/runs/${encodeURIComponent(runId)}/manager-review`, { method: "POST", body: JSON.stringify({ decision, comment }) }),
  ownerDecision: (runId: string, decision: "approve" | "edit" | "reject", comment?: string) => request<Report>(`/api/runs/${encodeURIComponent(runId)}/decision`, { method: "POST", body: JSON.stringify({ decision, comment }) }),
  conversations: async () => items<Conversation>(await request("/api/conversations")),
  createConversation: async (cafeId: string, runId?: string) => member<Conversation>(await request("/api/conversations", { method: "POST", body: JSON.stringify({ cafe_id: cafeId, run_id: runId }) }), "conversation"),
  sendMessage: (conversationId: string, message: string) => request<ChatAnswer>(`/api/conversations/${encodeURIComponent(conversationId)}/messages`, { method: "POST", body: JSON.stringify({ message }) }),
  accessRequests: async (status: "pending" | "approved" | "rejected" = "pending") => items<AccessRequest>(await request(`/api/admin/access-requests${query({ status })}`)),
  decideAccess: (userId: string, decision: "approve" | "reject") => request<{ access_request: AccessRequest }>(`/api/admin/access-requests/${encodeURIComponent(userId)}/decision`, { method: "POST", body: JSON.stringify({ decision }) }),
  aiSettings: async () => (await request<{ settings: AiSettings }>("/api/admin/ai-settings")).settings,
  saveAiSettings: async (payload: AiSettingsPayload) => (await request<{ settings: AiSettings }>("/api/admin/ai-settings", { method: "POST", body: JSON.stringify(payload) })).settings,
  testAiSettings: async () => (await request<{ result: AiConnectionTest }>("/api/admin/ai-settings/test", { method: "POST" })).result,
  clearAiSettings: async () => (await request<{ settings: AiSettings }>("/api/admin/ai-settings", { method: "DELETE" })).settings,
};

export const apiInternals = { items, member, query };
