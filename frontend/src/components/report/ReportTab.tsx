import { AlertTriangle } from "lucide-react";

import { CitationText } from "@/components/report/CitationText";
import { Badge } from "@/components/ui";
import { formatAuthors } from "@/lib/format";
import type { ResearchReport, SectionTheme } from "@/lib/types";

const THEME_LABELS: Record<SectionTheme, string> = {
  methods: "Methods",
  datasets: "Datasets",
  results: "Results",
  open_problems: "Open problems",
  other: "Background",
};

/** The literature review itself: summary, themed sections, comparison table, references. */
export function ReportTab({ report, titles }: { report: ResearchReport; titles: Map<number, string> }) {
  return (
    <article className="space-y-8">
      <section className="rounded-xl border border-brand-200 bg-brand-50/60 p-5 dark:border-brand-900 dark:bg-brand-950/30">
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-brand-700 dark:text-brand-300">Executive summary</h2>
        <p className="leading-7 text-slate-800 dark:text-slate-200">
          <CitationText text={report.executive_summary} titles={titles} />
        </p>
      </section>

      {report.sections.map((section) => (
        <section key={section.heading} className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-lg font-semibold text-slate-900 dark:text-white">{section.heading}</h2>
            <Badge tone="brand">{THEME_LABELS[section.theme]}</Badge>
          </div>
          {section.paragraphs.map((paragraph, index) => (
            <p key={index} className="leading-7 text-slate-700 dark:text-slate-300">
              <CitationText text={paragraph} titles={titles} />
            </p>
          ))}
        </section>
      ))}

      {report.comparison_table.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Comparison of key papers</h2>
          <div className="overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800">
            <table className="min-w-full divide-y divide-slate-200 text-sm dark:divide-slate-800">
              <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500 dark:bg-slate-900 dark:text-slate-400">
                <tr>
                  {["Ref", "Paper", "Year", "Method", "Dataset", "Key result"].map((heading) => (
                    <th key={heading} scope="col" className="px-3 py-2 font-semibold">
                      {heading}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {report.comparison_table.map((row) => (
                  <tr key={row.reference} className="align-top">
                    <td className="px-3 py-2">
                      <a href={`#ref-${row.reference}`} className="font-medium text-brand-600 dark:text-brand-400">
                        [{row.reference}]
                      </a>
                    </td>
                    <td className="max-w-xs px-3 py-2 font-medium text-slate-800 dark:text-slate-200">{row.title}</td>
                    <td className="px-3 py-2 text-slate-600 dark:text-slate-400">{row.year ?? "n.d."}</td>
                    <td className="px-3 py-2 text-slate-700 dark:text-slate-300">{row.method}</td>
                    <td className="px-3 py-2 text-slate-700 dark:text-slate-300">{row.dataset}</td>
                    <td className="px-3 py-2 text-slate-700 dark:text-slate-300">{row.key_result}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {report.references.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white">References</h2>
          <ol className="space-y-2 text-sm">
            {report.references.map((reference) => (
              <li
                key={reference.number}
                id={`ref-${reference.number}`}
                className="scroll-mt-20 rounded-lg p-2 text-slate-700 target:bg-brand-50 dark:text-slate-300 dark:target:bg-brand-950/50"
              >
                <span className="mr-2 font-medium text-slate-500">[{reference.number}]</span>
                {formatAuthors(reference.authors)} ({reference.year ?? "n.d."}).{" "}
                {reference.url ? (
                  <a href={reference.url} target="_blank" rel="noreferrer" className="font-medium text-brand-600 hover:underline dark:text-brand-400">
                    {reference.title}
                  </a>
                ) : (
                  <span className="font-medium">{reference.title}</span>
                )}
                {reference.venue && <span className="italic">. {reference.venue}</span>}
              </li>
            ))}
          </ol>
        </section>
      )}

      {report.warnings.length > 0 && (
        <details className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
          <summary className="flex cursor-pointer items-center gap-2 font-medium">
            <AlertTriangle className="h-4 w-4" aria-hidden /> {report.warnings.length} note(s) from the pipeline
          </summary>
          <ul className="mt-2 list-disc space-y-1 pl-6">
            {report.warnings.map((warning, index) => (
              <li key={index}>{warning}</li>
            ))}
          </ul>
        </details>
      )}
    </article>
  );
}
