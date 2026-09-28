"use client";

import clsx from "clsx";
import { Download, FileDown } from "lucide-react";
import { useMemo, useState } from "react";

import { useAuth } from "@/components/AuthProvider";
import { ChatTab } from "@/components/report/ChatTab";
import { PapersTab } from "@/components/report/PapersTab";
import { ReportTab } from "@/components/report/ReportTab";
import { VerificationTab } from "@/components/report/VerificationTab";
import { Alert, Button, Stat } from "@/components/ui";
import { formatCost, formatDuration, formatPercent } from "@/lib/format";
import type { ExportFormat, ResearchReport } from "@/lib/types";

type Tab = "report" | "papers" | "verification" | "chat";

function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

/** A finished report: header with metrics and export buttons, then the four tabs. */
export function ReportView({ sessionId, report }: { sessionId: string; report: ResearchReport }) {
  const { api } = useAuth();
  const [tab, setTab] = useState<Tab>("report");
  const [exporting, setExporting] = useState<ExportFormat | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const metrics = report.metrics;

  const titles = useMemo(() => new Map(report.references.map((reference) => [reference.number, reference.title])), [report]);
  const referenceNumbers = useMemo(
    () => new Map(report.references.map((reference) => [reference.paper_id, reference.number])),
    [report],
  );

  async function download(format: ExportFormat) {
    setExporting(format);
    setExportError(null);
    try {
      const file = await api.downloadExport(sessionId, format);
      saveBlob(file.blob, file.filename);
    } catch (caught) {
      setExportError(caught instanceof Error ? caught.message : "Export failed");
    } finally {
      setExporting(null);
    }
  }

  const tabs: { id: Tab; label: string }[] = [
    { id: "report", label: "Report" },
    { id: "papers", label: `Papers (${report.papers.length})` },
    { id: "verification", label: `Verification (${report.verification.length})` },
    { id: "chat", label: "Chat" },
  ];

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-6">
      <header className="space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0 flex-1 space-y-1">
            <h1 className="text-2xl font-semibold text-slate-900 dark:text-white">{report.title}</h1>
            <p className="text-sm text-slate-500 dark:text-slate-400">{report.question}</p>
          </div>
          <div className="flex gap-2">
            <Button variant="secondary" onClick={() => void download("markdown")} loading={exporting === "markdown"} icon={<FileDown className="h-4 w-4" aria-hidden />}>
              Markdown
            </Button>
            <Button variant="secondary" onClick={() => void download("pdf")} loading={exporting === "pdf"} icon={<Download className="h-4 w-4" aria-hidden />}>
              PDF
            </Button>
          </div>
        </div>
        {exportError && <Alert tone="error">{exportError}</Alert>}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
          <Stat label="Papers" value={metrics.papers_found} hint={`${metrics.papers_full_text} read in full text`} />
          <Stat label="Claims checked" value={metrics.claims_total || "—"} />
          <Stat
            label="Supported"
            value={metrics.critic_enabled ? formatPercent(metrics.faithfulness) : "off"}
            hint="Share of claims the Critic found fully supported by the sources"
          />
          <Stat label="Time" value={formatDuration(metrics.total_seconds)} hint={`${metrics.revisions} revision(s)`} />
          <Stat label="Cost" value={formatCost(metrics.total_cost_usd)} hint={`${metrics.llm_model}, ${metrics.total_input_tokens.toLocaleString()} input tokens`} />
        </div>
      </header>

      <div className="border-b border-slate-200 dark:border-slate-800" role="tablist" aria-label="Report sections">
        <div className="-mb-px flex gap-1 overflow-x-auto">
          {tabs.map((item) => (
            <button
              key={item.id}
              role="tab"
              aria-selected={tab === item.id}
              onClick={() => setTab(item.id)}
              className={clsx(
                "whitespace-nowrap border-b-2 px-4 py-2.5 text-sm font-medium",
                tab === item.id
                  ? "border-brand-600 text-brand-700 dark:border-brand-400 dark:text-brand-300"
                  : "border-transparent text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200",
              )}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <div role="tabpanel" className="animate-fade-in">
        {tab === "report" && <ReportTab report={report} titles={titles} />}
        {tab === "papers" && <PapersTab papers={report.papers} referenceNumbers={referenceNumbers} />}
        {tab === "verification" && (
          <VerificationTab claims={report.verification} criticEnabled={metrics.critic_enabled} titles={titles} />
        )}
        {tab === "chat" && <ChatTab sessionId={sessionId} />}
      </div>
    </div>
  );
}
