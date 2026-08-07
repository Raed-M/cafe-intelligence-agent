"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowDown, ArrowLeft, ArrowRight, CheckCircle2, Database, FileText, FileUp, Filter, FolderOpen, GitCompareArrows, Play, Search, ShieldCheck, Trash2, TriangleAlert, X } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import { assessFiles, flattenRows, humanLabel, humanValue, visibleColumns } from "@/lib/data-display";
import { displayError } from "@/lib/format";
import { useWorkspace } from "@/components/providers";
import { Skeleton, StatePanel, StatusBadge } from "@/components/ui";

type SelectedFile = { file: File; relativePath: string };
const selectedPath = (file: File) => (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;
const formatBytes = (bytes: number) => bytes < 1024 ? `${bytes} B` : bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
const fileAsBase64 = (file: File) => new Promise<string>((resolve, reject) => {
  const reader = new FileReader();
  reader.onerror = () => reject(reader.error || new Error("Could not read file"));
  reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
  reader.readAsDataURL(file);
});

export function DataExplorer() {
  const { locale, cafeId, cafe, viewMode, user } = useWorkspace();
  const queryClient = useQueryClient();
  const params = useSearchParams();
  const sourceParam = params.get("source") ?? "";
  const recordParam = params.get("record") ?? "";
  const [sourceId, setSourceId] = useState("");
  const [cursor, setCursor] = useState<string>();
  const [cursorHistory, setCursorHistory] = useState<(string | undefined)[]>([]);
  const [query, setQuery] = useState("");
  const [selectedRecord, setSelectedRecord] = useState(recordParam);
  const [sort, setSort] = useState<{ field: string; direction: "asc" | "desc" }>();
  const [uploaderOpen, setUploaderOpen] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<SelectedFile[]>([]);
  const [duplicateChoice, setDuplicateChoice] = useState<"new_only" | "overwrite">("new_only");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const sources = useQuery({ queryKey: ["sources", cafeId], queryFn: () => api.sources(cafeId), enabled: Boolean(cafeId) });
  const needsEmailPreview = selectedFiles.some(({ file }) => file.name.toLowerCase().endsWith(".txt"));
  const emailPreview = useQuery({ queryKey: ["upload-email-preview", cafeId], queryFn: () => api.data(cafeId, "emails"), enabled: Boolean(cafeId && needsEmailPreview), retry: false });
  const assessmentSignature = selectedFiles.map(({ file, relativePath }) => `${relativePath}:${file.size}:${file.lastModified}`).join("|");
  const assessment = useQuery({
    queryKey: ["upload-assessment", cafeId, assessmentSignature, sources.data?.map((item) => `${item.id}:${item.accepted_rows ?? item.raw_rows ?? 0}`).join("|"), emailPreview.data?.items.length],
    queryFn: () => assessFiles(selectedFiles, sources.data ?? [], emailPreview.data?.items ?? []),
    enabled: Boolean(selectedFiles.length && sources.data && (!needsEmailPreview || emailPreview.data || emailPreview.isError)),
    staleTime: Infinity,
  });
  const uploadAssessment = assessment.data ?? [];
  const assessmentPending = Boolean(selectedFiles.length && (sources.isLoading || assessment.isLoading || (needsEmailPreview && emailPreview.isLoading)));
  const canProcess = user?.role === "owner" || user?.role === "manager";
  const filesToProcess = uploadAssessment.filter((item) => item.state === "new" || (duplicateChoice === "overwrite" && item.state === "existing"));
  const processFiles = useMutation({
    mutationFn: async () => api.processData(cafeId, await Promise.all(filesToProcess.map(async ({ file, relativePath }) => ({
      name: file.name,
      relative_path: relativePath,
      media_type: file.type || undefined,
      size: file.size,
      last_modified: new Date(file.lastModified).toISOString(),
      content_base64: await fileAsBase64(file),
    })))),
    onSuccess() {
      queryClient.invalidateQueries({ queryKey: ["sources", cafeId] });
      queryClient.invalidateQueries({ queryKey: ["runs", cafeId] });
    },
  });

  const addFiles = (list: FileList | null) => {
    if (!list) return;
    const additions = Array.from(list).map((file) => ({ file, relativePath: selectedPath(file) }));
    setSelectedFiles((current) => {
      const merged = new Map(current.map((item) => [`${item.relativePath}:${item.file.size}:${item.file.lastModified}`, item]));
      additions.forEach((item) => merged.set(`${item.relativePath}:${item.file.size}:${item.file.lastModified}`, item));
      return Array.from(merged.values()).slice(0, 50);
    });
    setDuplicateChoice("new_only");
    processFiles.reset();
  };

  useEffect(() => {
    folderInputRef.current?.setAttribute("webkitdirectory", "");
  }, []);

  useEffect(() => {
    if (!uploaderOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => { if (event.key === "Escape") setUploaderOpen(false); };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [uploaderOpen]);

  const requestedSource = sources.data?.find((source) => source.id === sourceParam || source.name === sourceParam);
  const effectiveSourceId = sourceId || requestedSource?.id || sources.data?.[0]?.id || "";

  const records = useQuery({ queryKey: ["data", cafeId, effectiveSourceId, cursor], queryFn: () => api.data(cafeId, effectiveSourceId, cursor), enabled: Boolean(cafeId && effectiveSourceId), retry: 1 });
  const lineage = useQuery({ queryKey: ["lineage", cafeId, effectiveSourceId, selectedRecord], queryFn: () => api.lineage(cafeId, effectiveSourceId, selectedRecord), enabled: Boolean(cafeId && effectiveSourceId && selectedRecord), retry: false });
  const source = sources.data?.find((candidate) => candidate.id === effectiveSourceId);
  const normalizedRows = useMemo(() => flattenRows(records.data?.items ?? []), [records.data]);
  const columns = useMemo(() => visibleColumns(normalizedRows, effectiveSourceId), [normalizedRows, effectiveSourceId]);
  const rows = useMemo(() => {
    const filtered = query ? normalizedRows.filter((row) => Object.values(row).some((value) => humanValue(value, locale).toLowerCase().includes(query.toLowerCase()))) : normalizedRows;
    if (!sort) return filtered;
    return [...filtered].sort((a, b) => String(a[sort.field] ?? "").localeCompare(String(b[sort.field] ?? ""), locale === "ar" ? "ar" : "en", { numeric: true }) * (sort.direction === "asc" ? 1 : -1));
  }, [normalizedRows, query, sort, locale]);
  const changeFields = new Set(lineage.data?.changes?.map((change) => change.field).filter(Boolean));
  const newFileCount = uploadAssessment.filter((item) => item.state === "new").length;
  const existingFileCount = uploadAssessment.filter((item) => item.state === "existing" || item.state === "identical").length;
  const ignoredFileCount = uploadAssessment.filter((item) => item.state === "duplicate_selection" || item.state === "unsupported").length;

  return <div className="page-stack">
    <section className="page-heading"><div><span className="eyebrow">{locale === "ar" ? "خام ثابت · تنظيف بإصدار · أثر قابل للتتبع" : "Immutable raw · versioned clean · traceable lineage"}</span><h1>{locale === "ar" ? "مستكشف البيانات" : "Data explorer"}</h1><p>{cafe?.name || "—"}</p></div><div className="heading-actions">{source && <StatusBadge status={source.status} label={source.status} />}<button className="primary-button" type="button" disabled={!canProcess} onClick={() => setUploaderOpen(true)}><FileUp />{locale === "ar" ? "إضافة فواتير أو بيانات" : "Add invoices or data"}</button></div></section>
    <section className="data-intake-callout" aria-labelledby="data-intake-title"><span className="data-intake-icon"><FolderOpen aria-hidden="true" /></span><div><span className="eyebrow">Browser ingestion</span><h2 id="data-intake-title">Drop in files, then process them</h2><p>Select individual source files or a whole folder. Review the detected name, type, size, and date before starting the real pipeline.</p></div><button className="secondary-button" type="button" disabled={!canProcess} onClick={() => setUploaderOpen(true)}>Choose data<ArrowRight /></button></section>
    <section className="data-toolbar"><div className="field"><label htmlFor="source-filter">{locale === "ar" ? "المصدر" : "Source"}</label><select id="source-filter" value={effectiveSourceId} onChange={(event) => { setSourceId(event.target.value); setCursor(undefined); setCursorHistory([]); setSelectedRecord(""); }}>{sources.data?.map((item) => <option key={item.id} value={item.id}>{humanLabel(item.name, locale)}</option>)}</select></div><div className="field search-field"><label htmlFor="record-search">{locale === "ar" ? "تصفية الصفحة الحالية" : "Filter current page"}</label><div><Search /><input id="record-search" type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder={locale === "ar" ? "ابحث في القيم…" : "Search values…"} /></div></div><div className="source-stats"><span><b>{source?.raw_rows ?? "—"}</b>{locale === "ar" ? " خام" : " raw"}</span><span><b>{source?.accepted_rows ?? "—"}</b>{locale === "ar" ? " مقبول" : " accepted"}</span><span><b>{source?.rejected_rows ?? "—"}</b>{locale === "ar" ? " مرفوض" : " rejected"}</span></div></section>
    {sources.isLoading ? <Skeleton lines={5} /> : sources.isError ? <StatePanel kind="error" title={locale === "ar" ? "تعذر تحميل المصادر" : "Could not load sources"} body={displayError(sources.error, "API error")} /> : !sources.data?.length ? <StatePanel title={locale === "ar" ? "لا توجد مصادر مسجلة" : "No sources registered"} body={locale === "ar" ? "اربط أو ارفع مصدرًا من إعدادات المقهى قبل الاستكشاف." : "Connect or upload a café source before exploring data."} /> : <section className={`explorer-layout ${selectedRecord ? "with-lineage" : ""}`}>
      <article className="table-panel"><div className="table-head"><span><Filter />{locale === "ar" ? `${rows.length} سجلًا في الصفحة` : `${rows.length} readable records`}</span><small>{locale === "ar" ? "حدد صفًا لعرض أثره" : "Internal IDs are hidden; select any row for its audit trail"}</small></div>{records.isLoading ? <Skeleton lines={8} /> : records.isError ? <StatePanel kind="error" title={locale === "ar" ? "تعذر تحميل السجلات" : "Could not load records"} body={displayError(records.error, "API error")} /> : rows.length ? <div className="table-scroll" tabIndex={0}><table><thead><tr>{columns.map((column) => <th key={column} aria-sort={sort?.field === column ? (sort.direction === "asc" ? "ascending" : "descending") : "none"}><button type="button" onClick={() => setSort((current) => ({ field: column, direction: current?.field === column && current.direction === "asc" ? "desc" : "asc" }))}>{humanLabel(column, locale)}<ArrowDown /></button></th>)}</tr></thead><tbody>{rows.map((row) => { const id = row.__recordId; return <tr key={id} className={selectedRecord === id ? "selected" : ""} onClick={() => setSelectedRecord(id)}>{columns.map((column) => <td key={column}><button type="button" title={humanValue(row[column], locale)} onClick={() => setSelectedRecord(id)}>{humanValue(row[column], locale)}</button></td>)}</tr>; })}</tbody></table></div> : <StatePanel title={query ? (locale === "ar" ? "لا توجد مطابقة في الصفحة" : "No match on this page") : (locale === "ar" ? "المصدر لا يحتوي سجلات مرئية" : "Source has no visible records")} body={query ? (locale === "ar" ? "امسح التصفية أو استخدم صفحة أخرى." : "Clear the filter or try another page.") : undefined} />}
        {(records.data?.next_cursor || cursorHistory.length > 0) && <div className="pagination"><button className="secondary-button" type="button" disabled={cursorHistory.length === 0} onClick={() => { const newHistory = [...cursorHistory]; const prevCursor = newHistory.pop(); setCursorHistory(newHistory); setCursor(prevCursor); }}><ArrowLeft />{locale === "ar" ? "الصفحة السابقة" : "Previous page"}</button>{records.data?.next_cursor && <button className="secondary-button" type="button" onClick={() => { setCursorHistory([...cursorHistory, cursor]); setCursor(records.data?.next_cursor); }}>{locale === "ar" ? "الصفحة التالية" : "Next page"}<ArrowRight /></button>}</div>}
      </article>
      {selectedRecord && <aside className="lineage-panel"><div className="lineage-head"><div><span className="eyebrow">Record details</span><h2>{locale === "ar" ? "الخام مقابل المنظف" : "Original vs cleaned"}</h2>{viewMode === "technical" && <code title="Internal record reference">{selectedRecord}</code>}</div><button className="icon-button" type="button" onClick={() => setSelectedRecord("")} aria-label={locale === "ar" ? "إغلاق المقارنة" : "Close comparison"}>×</button></div>{lineage.isLoading ? <Skeleton lines={8} /> : lineage.isError ? <StatePanel kind="error" title={locale === "ar" ? "لا يتوفر أثر لهذا المعرّف" : "Audit trail unavailable"} body={displayError(lineage.error, locale === "ar" ? "قد يكون معرّف أثر لا معرّف سجل." : "This record does not expose a raw-to-cleaned comparison.")} /> : lineage.data ? <><div className="lineage-columns"><div><span className="column-label"><Database />ORIGINAL</span>{Object.entries(lineage.data.raw ?? {}).map(([key, value]) => <div className={changeFields.has(key) ? "changed-field" : ""} key={key}><b>{humanLabel(key, locale)}</b><span>{humanValue(value, locale)}</span></div>)}</div><div><span className="column-label"><ShieldCheck />CLEANED</span>{Object.entries(lineage.data.cleaned ?? {}).map(([key, value]) => <div className={changeFields.has(key) ? "changed-field" : ""} key={key}><b>{humanLabel(key, locale)}</b><span>{humanValue(value, locale)}</span></div>)}</div></div><div className="change-ledger"><span className="column-label"><GitCompareArrows />{locale === "ar" ? "أسباب التغيير" : "What changed and why"}</span>{lineage.data.changes?.length ? lineage.data.changes.map((change, index) => <div key={`${change.field}-${index}`}><strong>{humanLabel(change.field || "Field", locale)}</strong><span>{change.reason || change.rule || (locale === "ar" ? "لم يذكر الخادم سببًا" : "No reason returned")}</span></div>) : <p>{locale === "ar" ? "لم يسجل الخادم تغييرات لهذا السجل." : "No cleaning changes were needed for this record."}</p>}</div>{viewMode === "technical" && <pre className="json-block">{JSON.stringify(lineage.data, null, 2)}</pre>}</> : null}</aside>}
    </section>}
    {uploaderOpen && <div className="modal-layer" role="dialog" aria-modal="true" aria-labelledby="data-uploader-title">
      <button className="modal-scrim" type="button" onClick={() => setUploaderOpen(false)} aria-label="Close data uploader" />
      <div className="decision-dialog data-guide-dialog data-uploader-dialog">
        <div className="data-guide-head"><div><span className="eyebrow">Select · inspect · process</span><h2 id="data-uploader-title">Add invoices or source data</h2></div><button autoFocus className="icon-button" type="button" onClick={() => setUploaderOpen(false)} aria-label="Close data uploader"><X /></button></div>
        <div className="data-guide-notice"><ShieldCheck /><p><strong>Real local processing.</strong> Accepted files replace their registered café source with a recoverable backup, then start a new run.</p></div>
        <input ref={fileInputRef} className="sr-only" type="file" multiple accept=".csv,.xlsx,.json,.txt" onChange={(event) => addFiles(event.target.files)} />
        <input ref={folderInputRef} className="sr-only" type="file" multiple onChange={(event) => addFiles(event.target.files)} />
        <div className="upload-pickers"><button className="upload-picker" type="button" onClick={() => fileInputRef.current?.click()}><FileUp /><span><strong>Choose files</strong><small>CSV, XLSX, JSON, or TXT</small></span></button><button className="upload-picker" type="button" onClick={() => folderInputRef.current?.click()}><FolderOpen /><span><strong>Choose a folder</strong><small>Keep relative folder names</small></span></button></div>
        {selectedFiles.length ? <div className="upload-preview"><div className="upload-preview-head"><div><strong>{selectedFiles.length} selected</strong><small>{formatBytes(selectedFiles.reduce((total, item) => total + item.file.size, 0))} total</small></div><button className="text-link" type="button" onClick={() => { setSelectedFiles([]); processFiles.reset(); }}><Trash2 />Clear all</button></div><div className="upload-list" role="list">{selectedFiles.map(({ file, relativePath }, index) => <article role="listitem" key={`${relativePath}-${file.lastModified}`}><span className="file-type-icon"><FileText /></span><div><strong>{file.name}</strong><small title={relativePath}>{relativePath}</small></div><dl><div><dt>Type</dt><dd>{file.type || file.name.split(".").at(-1)?.toUpperCase() || "Unknown"}</dd></div><div><dt>Size</dt><dd>{formatBytes(file.size)}</dd></div><div><dt>Modified</dt><dd>{new Intl.DateTimeFormat(locale === "ar" ? "ar-SA" : "en-GB", { dateStyle: "medium" }).format(file.lastModified)}</dd></div></dl><button className="icon-button" type="button" onClick={() => { setSelectedFiles((current) => current.filter((_, itemIndex) => itemIndex !== index)); processFiles.reset(); }} aria-label={`Remove ${file.name}`}><X /></button></article>)}</div></div> : <StatePanel title="No files selected" body="Choose files or a folder. Nothing is uploaded until you click Process data." />}
        {selectedFiles.length > 0 && <section className="duplicate-review" aria-live="polite" aria-busy={assessmentPending}>
          <div className="duplicate-summary"><div><span>New</span><strong>{assessmentPending ? "…" : newFileCount}</strong><small>will be added</small></div><div><span>Already there</span><strong>{assessmentPending ? "…" : existingFileCount}</strong><small>need your choice</small></div><div><span>Ignored</span><strong>{assessmentPending ? "…" : ignoredFileCount}</strong><small>unsupported or repeated</small></div></div>
          {!assessmentPending && existingFileCount > 0 && <fieldset><legend>What should happen to existing data?</legend><label className={duplicateChoice === "new_only" ? "selected" : ""}><input type="radio" name="duplicate-choice" checked={duplicateChoice === "new_only"} onChange={() => setDuplicateChoice("new_only")} /><span><strong>Keep existing, add new only</strong><small>Safest option. Existing files stay exactly as they are.</small></span></label><label className={duplicateChoice === "overwrite" ? "selected" : ""}><input type="radio" name="duplicate-choice" checked={duplicateChoice === "overwrite"} onChange={() => setDuplicateChoice("overwrite")} /><span><strong>Replace existing and add new</strong><small>Changed registered files are overwritten; the normal recovery backup is kept.</small></span></label></fieldset>}
          {!assessmentPending && uploadAssessment.length > 0 && <details className="file-assessment"><summary>See file-by-file result</summary>{uploadAssessment.map((item) => <div key={`${item.relativePath}-${item.file.lastModified}`}><span className={`assessment-dot assessment-${item.state}`} /><div><strong>{item.file.name}</strong><small>{item.detail}</small></div><b>{item.state.replace(/_/g, " ")}</b></div>)}</details>}
        </section>}
        {processFiles.isError && <StatePanel kind="error" title="Processing could not start" body={displayError(processFiles.error, "Check the file names, formats, and size limits.")} />}
        {processFiles.data && <div className="upload-result" aria-live="polite"><div><CheckCircle2 /><strong>{processFiles.data.upload.accepted.length} accepted</strong></div>{processFiles.data.upload.rejected.length > 0 && <div><TriangleAlert /><strong>{processFiles.data.upload.rejected.length} rejected</strong></div>}<p>The accepted files were saved and run <code>{processFiles.data.run.id}</code> was created.</p>{processFiles.data.upload.rejected.map((item) => <small key={`${item.name}-${item.reason}`}>{item.name}: {item.reason}</small>)}</div>}
        <div className="dialog-actions"><button className="secondary-button" type="button" onClick={() => setUploaderOpen(false)}>Close</button>{processFiles.data ? <Link className="primary-button" href={`/runs/${processFiles.data.run.id}`}><Play />Open live run</Link> : <button className="primary-button" type="button" disabled={assessmentPending || !filesToProcess.length || processFiles.isPending} onClick={() => processFiles.mutate()}><Play />{processFiles.isPending ? "Processing…" : duplicateChoice === "overwrite" ? `Process ${filesToProcess.length} files` : `Add ${filesToProcess.length} new files`}</button>}</div>
      </div>
    </div>}
  </div>;
}
