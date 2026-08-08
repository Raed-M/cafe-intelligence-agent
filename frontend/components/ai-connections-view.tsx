"use client";

import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  BrainCircuit,
  CheckCircle2,
  Coins,
  ExternalLink,
  Eye,
  EyeOff,
  Info,
  KeyRound,
  LockKeyhole,
  PlugZap,
  Search,
  ShieldCheck,
  Timer,
  Trash2,
} from "lucide-react";
import { api } from "@/lib/api";
import { displayError } from "@/lib/format";
import type { AiModelOption, AiProviderOption, AiSettingsPayload } from "@/lib/types";
import { useWorkspace } from "@/components/providers";
import { Skeleton, StatePanel } from "@/components/ui";

const providerNames = {
  openai: "OpenAI",
  anthropic: "Anthropic",
  gemini: "Google Gemini",
} as const;

const price = (value: number | null) =>
  value === null || value === undefined ? "—" : `$${value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

function ModelCard({ model, selected, onSelect }: { model: AiModelOption; selected: boolean; onSelect: () => void }) {
  return <button type="button" className={`model-choice ${selected ? "selected" : ""}`} onClick={onSelect} aria-pressed={selected}>
    <span className="model-choice-head"><span><span className="model-status-row"><b>{model.tier}</b><em className={`model-status ${model.status === "Current" ? "current" : model.status === "Preview" ? "preview" : "previous"}`}>{model.status}</em></span><strong>{model.name}</strong><code>{model.id}</code></span>{selected && <CheckCircle2 aria-hidden="true" />}</span>
    <span className="model-summary">{model.summary}</span>
    <span className="model-price-label">Standard text price · per 1M tokens</span>
    <span className="model-prices">
      <span><small>Input</small><strong>{price(model.input_price)}</strong></span>
      <span><small>Output</small><strong>{price(model.output_price)}</strong></span>
    </span>
    <span className="model-speed"><Timer aria-hidden="true" /><span><small>Typical speed</small><strong>{model.speed}</strong></span><p>{model.speed_note}</p></span>
    <span className="model-context"><span>Context <strong>{model.context_window}</strong></span>{model.cached_input_price !== null && <span>Cached input <strong>{price(model.cached_input_price)}</strong></span>}</span>
    {model.pricing_note && <span className="model-pricing-note"><Info aria-hidden="true" />{model.pricing_note}</span>}
    <span className="model-best-for"><small>Best for</small><strong>{model.recommended_for}</strong></span>
  </button>;
}

function SecretField({ id, label, required, configured, fingerprint, value, onChange, hint }: { id: string; label: string; required?: boolean; configured?: boolean; fingerprint?: string | null; value: string; onChange: (value: string) => void; hint: string }) {
  const [visible, setVisible] = useState(false);
  return <div className="secret-field">
    <div className="secret-label"><label htmlFor={id}>{label}</label><span className={required ? "required-tag" : "optional-tag"}>{required ? "Required" : "Optional"}</span></div>
    <div className="secret-input-wrap">
      <KeyRound aria-hidden="true" />
      <input id={id} type={visible ? "text" : "password"} value={value} onChange={(event) => onChange(event.target.value)} autoComplete="off" spellCheck={false} placeholder={configured ? "Leave blank to keep the saved key" : "Paste key here"} aria-describedby={`${id}-hint`} />
      <button type="button" className="secret-reveal" onClick={() => setVisible((current) => !current)} aria-label={visible ? `Hide ${label}` : `Show ${label}`}>{visible ? <EyeOff /> : <Eye />}</button>
    </div>
    <div className="secret-hint" id={`${id}-hint`}><span>{hint}</span>{configured && <strong><CheckCircle2 />Configured · fingerprint {fingerprint}</strong>}</div>
  </div>;
}

export function AiConnectionsView() {
  const { user } = useWorkspace();
  const queryClient = useQueryClient();
  const settings = useQuery({ queryKey: ["ai-settings"], queryFn: api.aiSettings, enabled: user?.role === "owner", retry: false });
  const [selectedProvider, setSelectedProvider] = useState<AiProviderOption["id"] | null>(null);
  const [selectedAnalysisModel, setSelectedAnalysisModel] = useState("");
  const [selectedUtilityModel, setSelectedUtilityModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [tavilyKey, setTavilyKey] = useState("");
  const [langsmithKey, setLangsmithKey] = useState("");
  const [selectedRemember, setSelectedRemember] = useState<boolean | null>(null);
  const provider = selectedProvider ?? settings.data?.provider ?? "openai";
  const providerOption = useMemo(() => settings.data?.catalog.find((item) => item.id === provider), [provider, settings.data]);
  const savedProviderSelected = settings.data?.provider === provider;
  const analysisModel = selectedAnalysisModel || (savedProviderSelected ? settings.data?.analysis_model : "") || providerOption?.default_analysis_model || providerOption?.models[0]?.id || "";
  const utilityModel = selectedUtilityModel || (savedProviderSelected ? settings.data?.utility_model : "") || providerOption?.default_utility_model || providerOption?.models[0]?.id || "";
  const remember = selectedRemember ?? settings.data?.persisted ?? false;
  const switchProvider = (next: AiProviderOption) => {
    setSelectedProvider(next.id);
    setSelectedAnalysisModel(next.default_analysis_model);
    setSelectedUtilityModel(next.default_utility_model);
    setApiKey("");
  };
  const save = useMutation({
    mutationFn: (payload: AiSettingsPayload) => api.saveAiSettings(payload),
    onSuccess(data) {
      queryClient.setQueryData(["ai-settings"], data);
      setApiKey(""); setTavilyKey(""); setLangsmithKey("");
    },
  });
  const test = useMutation({ mutationFn: api.testAiSettings });
  const clear = useMutation({
    mutationFn: api.clearAiSettings,
    onSuccess(data) { queryClient.setQueryData(["ai-settings"], data); setApiKey(""); setTavilyKey(""); setLangsmithKey(""); setSelectedRemember(false); },
  });

  if (user?.role !== "owner") return <StatePanel kind="error" title="Owner access required" body="Only the owner can view or change AI provider credentials." />;
  if (settings.isLoading) return <Skeleton lines={8} />;
  if (settings.isError || !settings.data) return <StatePanel kind="error" title="AI settings are unavailable" body={displayError(settings.error, "The API did not return the provider configuration.")} />;

  const providerConfigured = settings.data.provider === provider && settings.data.provider_configured;
  const canSave = Boolean(provider && analysisModel && (apiKey || providerConfigured));
  const submit = () => save.mutate({ provider, api_key: apiKey || undefined, analysis_model: analysisModel, utility_model: utilityModel || analysisModel, tavily_key: tavilyKey || undefined, langsmith_key: langsmithKey || undefined, remember });
  const models = providerOption?.models ?? [];
  // Unpriced (unlisted) models cannot take part in a cheapest-model comparison.
  const pricedModels = models.filter((m) => m.input_price !== null && m.output_price !== null);
  const lowestCostModel = pricedModels.reduce<AiModelOption | undefined>((lowest, model) => !lowest || (model.input_price! + model.output_price!) < (lowest.input_price! + lowest.output_price!) ? model : lowest, undefined);
  const fastestModel = models.reduce<AiModelOption | undefined>((fastest, model) => !fastest || model.speed_rank > fastest.speed_rank ? model : fastest, undefined);

  return <div className="page-stack ai-settings-page">
    <section className="page-heading ai-heading">
      <div><span className="eyebrow">CONTROL ROOM · OWNER ONLY</span><h1>AI connections</h1><p>Choose who powers the analysis, control the quality–cost tradeoff, and keep every credential server-side.</p></div>
      <div className={`connection-state ${settings.data.provider_configured ? "ready" : "missing"}`}><span /><div><strong>{settings.data.provider_configured ? "Provider configured" : "Setup required"}</strong><small>{settings.data.provider_configured ? `${providerNames[settings.data.provider!]} · ${settings.data.analysis_model}` : "A provider key is required before a live run."}</small></div></div>
    </section>

    <section className="security-vault-note"><LockKeyhole aria-hidden="true" /><div><strong>Keys stay behind the API boundary</strong><p>The browser sends a key once over the local API and never receives it back. Session-only is the safest default. “Remember” encrypts the configuration with Windows DPAPI for this Windows account.</p></div><ShieldCheck aria-hidden="true" /></section>

    <section className="settings-section">
      <div className="settings-section-head"><span>01</span><div><h2>Intelligence provider</h2><p>Select one provider. You do not need keys for all three.</p></div><b className="required-tag">Required</b></div>
      <div className="provider-grid">{settings.data.catalog.map((option) => <button type="button" key={option.id} className={`provider-card ${provider === option.id ? "selected" : ""}`} onClick={() => switchProvider(option)} aria-pressed={provider === option.id}><span className="provider-sigil">{providerNames[option.id].slice(0, 1)}</span><span><strong>{providerNames[option.id]}</strong><small>{option.models.length} compatible model{option.models.length === 1 ? "" : "s"}</small></span>{provider === option.id && <CheckCircle2 />}</button>)}</div>
      <SecretField id="provider-key" label={`${providerNames[provider]} API key`} required configured={providerConfigured} fingerprint={providerConfigured ? settings.data.provider_fingerprint : null} value={apiKey} onChange={setApiKey} hint="Never stored in the browser, page source, logs, or Git." />
    </section>

    <section className="settings-section">
      <div className="settings-section-head"><span>02</span><div><h2>Analysis model</h2><p>Compare actual text-token prices and plain-language speed guidance within {providerNames[provider]}.</p></div><b className="required-tag">Required</b></div>
      <div className="model-comparison-summary">
        <div><Coins aria-hidden="true" /><span><small>Lowest token price</small><strong>{lowestCostModel?.name}</strong><em>{lowestCostModel ? `${price(lowestCostModel.input_price)} in · ${price(lowestCostModel.output_price)} out` : "—"}</em></span></div>
        <div><Timer aria-hidden="true" /><span><small>Fastest tier</small><strong>{fastestModel?.name}</strong><em>{fastestModel?.speed}</em></span></div>
        <div><BrainCircuit aria-hidden="true" /><span><small>Selected for analysis</small><strong>{models.find((model) => model.id === analysisModel)?.name}</strong><em>{models.find((model) => model.id === analysisModel)?.tier}</em></span></div>
        {providerOption && <a href={providerOption.source_url} target="_blank" rel="noreferrer">Official model &amp; pricing source <ExternalLink aria-hidden="true" /></a>}
      </div>
      <div className="model-grid">{models.map((model) => <ModelCard key={model.id} model={model} selected={analysisModel === model.id} onSelect={() => setSelectedAnalysisModel(model.id)} />)}</div>
      <div className="utility-select"><div><label htmlFor="utility-model">Utility model <span className="optional-tag">Optional</span></label><p>Used for email extraction and short report compression. A smaller model usually saves time and tokens here.</p></div><select id="utility-model" value={utilityModel} onChange={(event) => setSelectedUtilityModel(event.target.value)}>{providerOption?.models.map((model) => <option value={model.id} key={model.id}>{model.name}</option>)}</select></div>
    </section>

    <section className="settings-section">
      <div className="settings-section-head"><span>03</span><div><h2>Optional services</h2><p>The weekly pipeline still runs without these. Missing services are shown as degraded, not as success.</p></div><b className="optional-tag">Optional</b></div>
      <div className="optional-grid">
        <div className="optional-service"><Search /><div><h3>Tavily local search</h3><p>Adds local events and web context. Queries may leave this device.</p></div><SecretField id="tavily-key" label="Tavily API key" configured={settings.data.tavily_configured} fingerprint={settings.data.tavily_fingerprint} value={tavilyKey} onChange={setTavilyKey} hint="Optional context enrichment; cafe files are not uploaded by this field." /></div>
        <div className="optional-service"><Activity /><div><h3>LangSmith tracing</h3><p>Debugs model calls and timing. Trace content may include prompts and outputs.</p></div><SecretField id="langsmith-key" label="LangSmith API key" configured={settings.data.langsmith_configured} fingerprint={settings.data.langsmith_fingerprint} value={langsmithKey} onChange={setLangsmithKey} hint="Leave empty if you do not want external traces." /></div>
      </div>
    </section>

    <section className="settings-actions-panel">
      <label className="remember-choice"><input type="checkbox" checked={remember} onChange={(event) => setSelectedRemember(event.target.checked)} disabled={!settings.data.persistence_available} /><span><strong>Remember on this Windows account</strong><small>{settings.data.persistence_available ? "Encrypted with Windows DPAPI. Other Windows users cannot decrypt it." : "Persistent encrypted storage is unavailable on this operating system."}</small></span></label>
      <div className="settings-actions">
        <button className="secondary-button" type="button" disabled={!settings.data.provider_configured || test.isPending} onClick={() => test.mutate()}><PlugZap />{test.isPending ? "Testing…" : "Test connection"}</button>
        <button className="primary-button" type="button" disabled={!canSave || save.isPending} onClick={submit}><ShieldCheck />{save.isPending ? "Saving securely…" : "Save AI settings"}</button>
      </div>
      {(save.isSuccess || test.data) && <div className={`settings-feedback ${(test.data?.ok ?? true) ? "success" : "error"}`} role="status"><Info />{test.data ? `${test.data.message} (${test.data.latency_ms} ms)` : "Settings applied to new runs. Secret values were not returned to the browser."}</div>}
      {(save.isError || test.isError) && <div className="settings-feedback error" role="alert"><Info />{displayError(save.error || test.error, "The settings could not be applied.")}</div>}
      {settings.data.provider_configured && <button className="danger-text-button" type="button" disabled={clear.isPending} onClick={() => { if (window.confirm("Remove the saved AI configuration from this app?")) clear.mutate(); }}><Trash2 />Remove AI configuration</button>}
    </section>

    <section className="model-legend"><div><Coins /><span><strong>Prices are USD per 1M tokens</strong><small>Standard paid text rates are shown; free, batch, cache, long-context, and promotional rates can differ.</small></span></div><div><Timer /><span><strong>Speed is comparative, not guaranteed</strong><small>Network load, prompt size, reasoning effort, and tool calls change real response time.</small></span></div><div><BrainCircuit /><span><strong>Lower price does not mean fewer tokens</strong><small>Your input, output, and hidden reasoning usage determine the final bill.</small></span></div></section>
  </div>;
}
