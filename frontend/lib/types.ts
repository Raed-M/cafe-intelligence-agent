export type Role = "owner" | "manager" | "employee" | "demo_visitor" | string;

export interface User { id: string; email: string; display_name: string; role: Role; cafe_ids: string[] }
export interface Cafe { id: string; name: string; city?: string; region?: string; country?: string; timezone?: string; currency?: string; seats?: number; branch_id?: string }
export interface SourceSummary { id: string; name: string; status: string; raw_rows?: number; accepted_rows?: number; rejected_rows?: number; schema_version?: string; last_run_id?: string }
export interface AgentState { id?: string; name?: string; status?: string; message?: string; attempts?: number; runtime_ms?: number; error?: string }
export interface Run { id: string; cafe_id: string; status: string; stage?: string; created_at?: string; updated_at?: string; analysis_period?: string | { start?: string; end?: string }; report_state?: string; findings_count?: number; error_count?: number; agents?: Record<string, AgentState> | AgentState[] }
export interface Evidence { id: string; metric_name: string; value: unknown; unit?: string; period_start?: string; period_end?: string; source_names?: string[]; artifact_id?: string }
export interface Finding { id: string; analyst: string; title: string; claim: string; type?: string; confidence?: number; approved?: boolean; evidence: Evidence[] }
export interface Report { run_id: string; state: string; format?: "html" | string; html?: string; whatsapp_summary?: string; generated_at?: string }
/** A Monday-start week the POS data covers. `covered_days` is how many of the
 *  seven days actually carry transactions, so a stub week is distinguishable
 *  from a whole one. */
export interface WeekOption { week_start: string; week_end: string; transactions: number; covered_days: number }
export interface AvailableWeeks { cafe_id: string; items: WeekOption[]; default_week: string | null }
/** Deployment switches the UI must respect. `owner_self_review` is off unless
 *  the API opted out of two-person review, so the gate can tell an owner why
 *  the manager step is not theirs to take. */
export interface Capabilities { owner_self_review: boolean; local_file_access: boolean }
/** Where a run's report files landed on the machine running the API. `available`
 *  is false until the files actually exist on disk; `can_reveal` is false when
 *  the API is not local/development, in which case `directory` is all the UI
 *  can offer. */
export interface ReportLocation { run_id: string; available: boolean; can_reveal: boolean; directory: string | null; files: Array<{ name: string; bytes: number }> }
export interface RunEvent { event_id?: string; run_id: string; type: string; at?: string; stage?: string; status?: string; message?: string }
export interface Lineage { raw?: Record<string, unknown> | null; cleaned?: Record<string, unknown> | null; changes?: Array<{ field?: string; before?: unknown; after?: unknown; reason?: string; rule?: string }> }
export interface Conversation { id: string; cafe_id?: string; title?: string; created_at?: string }
/** A source behind a chat answer. `evidence` citations name a pipeline finding
 *  and link to its in-app page; `web` citations carry an external URL. */
export interface Citation { kind?: "evidence" | "web"; url?: string; label?: string; finding_id?: string }
export interface ChatAnswer { answer?: string; message?: string; citations?: Array<string | Citation>; limitations?: string[]; confidence?: number }
export interface AccessRequest { id: string; email: string; display_name: string; requested_role: "manager" | "employee"; status: "pending" | "approved" | "rejected"; created_at?: string; reviewed_at?: string; cafe_ids: string[] }
export interface BrowserFilePayload { name: string; relative_path: string; media_type?: string; size: number; last_modified?: string; content_base64: string }
export interface ProcessedBrowserFile { name: string; relative_path?: string; source?: string; type?: string; size?: number; last_modified?: string; status: "accepted" | "rejected"; reason?: string; replaced_existing?: boolean }
export interface DataProcessResult { upload: { batch_id: string; accepted: ProcessedBrowserFile[]; rejected: ProcessedBrowserFile[]; processed_at: string }; run: Run }
export interface AiModelOption {
  id: string;
  name: string;
  summary: string;
  tier: string;
  status: "Current" | "Preview" | "Previous generation";
  // null for a model configured outside the built-in catalog: unknown,
  // which the UI must render as such rather than as free.
  input_price: number | null;
  cached_input_price: number | null;
  output_price: number | null;
  speed: string;
  speed_rank: number;
  speed_note: string;
  context_window: string;
  pricing_note: string | null;
  recommended_for: string;
}
export interface AiProviderOption { id: "openai" | "anthropic" | "gemini"; key_env: string; source_url: string; default_analysis_model: string; default_utility_model: string; models: AiModelOption[] }
export interface AiSettings { provider?: AiProviderOption["id"] | null; provider_configured: boolean; provider_fingerprint?: string | null; analysis_model?: string; utility_model?: string; tavily_configured: boolean; tavily_fingerprint?: string | null; langsmith_configured: boolean; langsmith_fingerprint?: string | null; persisted: boolean; persistence_available: boolean; catalog: AiProviderOption[] }
export interface AiSettingsPayload { provider: AiProviderOption["id"]; api_key?: string; analysis_model: string; utility_model?: string; tavily_key?: string; langsmith_key?: string; remember: boolean }
export interface AiConnectionTest { ok: boolean; latency_ms: number; message: string }
