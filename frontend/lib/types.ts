export type Role = "owner" | "manager" | "employee" | "demo_visitor" | string;

export interface User { id: string; email: string; display_name: string; role: Role; cafe_ids: string[] }
export interface Cafe { id: string; name: string; city?: string; region?: string; country?: string; timezone?: string; currency?: string; seats?: number; branch_id?: string }
export interface SourceSummary { id: string; name: string; status: string; raw_rows?: number; accepted_rows?: number; rejected_rows?: number; schema_version?: string; last_run_id?: string }
export interface AgentState { id?: string; name?: string; status?: string; message?: string; attempts?: number; runtime_ms?: number; error?: string }
export interface Run { id: string; cafe_id: string; status: string; stage?: string; created_at?: string; updated_at?: string; analysis_period?: string | { start?: string; end?: string }; report_state?: string; findings_count?: number; error_count?: number; agents?: Record<string, AgentState> | AgentState[] }
export interface Evidence { id: string; metric_name: string; value: unknown; unit?: string; period_start?: string; period_end?: string; source_names?: string[]; artifact_id?: string }
export interface Finding { id: string; analyst: string; title: string; claim: string; type?: string; confidence?: number; approved?: boolean; evidence: Evidence[] }
export interface Report { run_id: string; state: string; format?: "html" | string; html?: string; whatsapp_summary?: string; generated_at?: string }
export interface RunEvent { event_id?: string; run_id: string; type: string; at?: string; stage?: string; status?: string; message?: string }
export interface Lineage { raw?: Record<string, unknown> | null; cleaned?: Record<string, unknown> | null; changes?: Array<{ field?: string; before?: unknown; after?: unknown; reason?: string; rule?: string }> }
export interface Conversation { id: string; cafe_id?: string; title?: string; created_at?: string }
export interface ChatAnswer { answer?: string; message?: string; citations?: Array<string | { id?: string; label?: string }>; limitations?: string[]; confidence?: number }
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
  input_price: number;
  cached_input_price: number | null;
  output_price: number;
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
