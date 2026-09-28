import { ExternalLink, FileText } from "lucide-react";

import { Badge } from "@/components/ui";
import { formatAuthors } from "@/lib/format";
import type { PaperDigest } from "@/lib/types";

function List({ label, items }: { label: string; items: string[] }) {
  if (items.length === 0) return null;
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">{label}</p>
      <ul className="mt-1 list-disc space-y-0.5 pl-5 text-sm text-slate-700 dark:text-slate-300">
        {items.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

/** Every paper the agents read, with its ranking and the Reader's structured summary. */
export function PapersTab({ papers, referenceNumbers }: { papers: PaperDigest[]; referenceNumbers: Map<string, number> }) {
  return (
    <div className="space-y-4">
      {papers.map((digest, index) => {
        const { paper, summary } = digest;
        const reference = referenceNumbers.get(paper.paper_id);
        return (
          <div key={paper.paper_id} className="rounded-xl border border-slate-200 p-5 dark:border-slate-800">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  #{index + 1} · score {paper.score.toFixed(2)}
                  {reference !== undefined && ` · cited as [${reference}]`}
                </p>
                <h3 className="mt-0.5 font-semibold text-slate-900 dark:text-white">{paper.title}</h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  {formatAuthors(paper.authors)} · {paper.year ?? "n.d."}
                  {paper.venue && ` · ${paper.venue}`}
                </p>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {digest.full_text ? <Badge tone="green">Full text</Badge> : <Badge tone="amber">Abstract only</Badge>}
                <Badge>{paper.citation_count.toLocaleString()} citations</Badge>
                {paper.sources.map((source) => (
                  <Badge key={source} tone="blue">
                    {source === "semantic_scholar" ? "Semantic Scholar" : "arXiv"}
                  </Badge>
                ))}
              </div>
            </div>

            {summary ? (
              <div className="mt-4 grid gap-4 md:grid-cols-2">
                <div className="space-y-3 text-sm text-slate-700 dark:text-slate-300">
                  <p>
                    <span className="font-semibold">Problem: </span>
                    {summary.problem}
                  </p>
                  <p>
                    <span className="font-semibold">Method: </span>
                    {summary.method}
                  </p>
                  {(summary.datasets.length > 0 || summary.metrics.length > 0) && (
                    <div className="flex flex-wrap gap-1.5">
                      {summary.datasets.map((dataset) => (
                        <Badge key={`d-${dataset}`} tone="brand">
                          {dataset}
                        </Badge>
                      ))}
                      {summary.metrics.map((metric) => (
                        <Badge key={`m-${metric}`}>{metric}</Badge>
                      ))}
                    </div>
                  )}
                </div>
                <div className="space-y-3">
                  <List label="Key results" items={summary.key_results} />
                  <List label="Limitations" items={summary.limitations} />
                </div>
              </div>
            ) : (
              <p className="mt-3 text-sm text-slate-600 dark:text-slate-400">{paper.abstract}</p>
            )}

            <div className="mt-4 flex flex-wrap items-center gap-4 text-sm">
              {paper.url && (
                <a href={paper.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-brand-600 hover:underline dark:text-brand-400">
                  <ExternalLink className="h-3.5 w-3.5" aria-hidden /> Paper page
                </a>
              )}
              {paper.pdf_url && (
                <a href={paper.pdf_url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-brand-600 hover:underline dark:text-brand-400">
                  <FileText className="h-3.5 w-3.5" aria-hidden /> PDF
                </a>
              )}
              {digest.note && <span className="text-xs text-amber-700 dark:text-amber-400">{digest.note}</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}
