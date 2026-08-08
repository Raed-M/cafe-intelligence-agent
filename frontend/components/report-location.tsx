"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Copy, FolderOpen } from "lucide-react";
import { useState } from "react";
import { api } from "@/lib/api";
import { displayError } from "@/lib/format";
import { livePolling } from "@/lib/run-state";
import { useWorkspace } from "@/components/providers";

/** "Open reports folder" for a finished run.
 *
 *  Renders nothing at all until the API confirms the files exist on disk, so
 *  the control cannot appear for a run whose report was never saved. When the
 *  API is not local (`can_reveal` false) it degrades to showing/copying the
 *  path, because a browser cannot open a folder on a different machine. */
export function ReportLocationButton({ runId, status }: { runId: string; status?: string }) {
  const { locale } = useWorkspace();
  const queryClient = useQueryClient();
  const [copied, setCopied] = useState(false);

  const location = useQuery({
    queryKey: ["report-location", runId],
    queryFn: () => api.reportLocation(runId),
    retry: false,
    refetchInterval: () => livePolling(status),
  });

  const reveal = useMutation({
    mutationFn: () => api.revealReport(runId),
    onError() { queryClient.invalidateQueries({ queryKey: ["report-location", runId] }); },
  });

  if (!location.data?.available || !location.data.directory) return null;
  const directory = location.data.directory;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(directory);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2_000);
    } catch {
      setCopied(false);
    }
  };

  return <div className="report-location">
    <div className="report-location-head">
      <FolderOpen aria-hidden="true" />
      <div>
        <strong>{locale === "ar" ? "التقرير محفوظ محليًا" : "Report saved locally"}</strong>
        <small>{location.data.files.map((file) => file.name).join(" · ")}</small>
      </div>
    </div>
    <code className="report-location-path" title={directory}>{directory}</code>
    <div className="report-location-actions">
      {location.data.can_reveal && <button type="button" className="secondary-button" onClick={() => reveal.mutate()} disabled={reveal.isPending}>
        <FolderOpen />{reveal.isPending ? (locale === "ar" ? "جارٍ الفتح…" : "Opening…") : (locale === "ar" ? "افتح المجلد" : "Open folder")}
      </button>}
      <button type="button" className="secondary-button" onClick={copy}>
        {copied ? <Check /> : <Copy />}{copied ? (locale === "ar" ? "تم النسخ" : "Copied") : (locale === "ar" ? "انسخ المسار" : "Copy path")}
      </button>
    </div>
    {!location.data.can_reveal && <small className="report-location-note">{locale === "ar" ? "فتح المجلد متاح فقط عندما تعمل الواجهة الخلفية على نفس الجهاز." : "Opening the folder is only available when the API runs on this machine."}</small>}
    {reveal.isError && <small className="report-location-note">{displayError(reveal.error, locale === "ar" ? "تعذر فتح المجلد." : "Could not open the folder.")}</small>}
  </div>;
}
