"use client";

import clsx from "clsx";
import { ShieldOff } from "lucide-react";
import { useState } from "react";

import { CitationText } from "@/components/report/CitationText";
import { Badge, EmptyState } from "@/components/ui";
import { formatPercent } from "@/lib/format";
import type { SupportLabel, VerifiedClaim } from "@/lib/types";

const LABELS: Record<SupportLabel, { text: string; tone: "green" | "amber" | "red"; bar: string }> = {
  supported: { text: "Supported", tone: "green", bar: "bg-emerald-500" },
  partial: { text: "Partially supported", tone: "amber", bar: "bg-amber-500" },
  unsupported: { text: "Unsupported", tone: "red", bar: "bg-red-500" },
};

type Filter = "all" | SupportLabel;

/** Every claim the Critic checked, its verdict, and the source passage it was checked against. */
export function VerificationTab({
  claims,
  criticEnabled,
  titles,
}: {
  claims: VerifiedClaim[];
  criticEnabled: boolean;
  titles: Map<number, string>;
}) {
  const [filter, setFilter] = useState<Filter>("all");

  if (claims.length === 0) {
    return (
      <EmptyState icon={<ShieldOff className="h-10 w-10" />} title={criticEnabled ? "No claims were verified" : "Fact-checking was turned off"}>
        {criticEnabled
          ? "The Critic did not return any verdicts for this report (see the pipeline notes on the Report tab)."
          : "This run was made without the Critic agent, so its claims were not checked against the sources."}
      </EmptyState>
    );
  }

  const counts = { supported: 0, partial: 0, unsupported: 0 };
  for (const claim of claims) counts[claim.label] += 1;
  const shown = filter === "all" ? claims : claims.filter((claim) => claim.label === filter);

  return (
    <div className="space-y-5">
      <div className="space-y-2">
        <div className="flex h-3 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800" aria-hidden>
          {(Object.keys(LABELS) as SupportLabel[]).map((label) => (
            <div key={label} className={LABELS[label].bar} style={{ width: `${(counts[label] / claims.length) * 100}%` }} />
          ))}
        </div>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          {claims.length} claims checked · {formatPercent(counts.supported / claims.length)} supported by the source passages
        </p>
      </div>

      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Filter claims">
        {(["all", "supported", "partial", "unsupported"] as Filter[]).map((option) => (
          <button
            key={option}
            role="tab"
            aria-selected={filter === option}
            onClick={() => setFilter(option)}
            className={clsx(
              "rounded-full border px-3 py-1 text-sm",
              filter === option
                ? "border-brand-500 bg-brand-50 text-brand-700 dark:bg-brand-950/50 dark:text-brand-300"
                : "border-slate-200 text-slate-600 hover:border-slate-300 dark:border-slate-700 dark:text-slate-300",
            )}
          >
            {option === "all" ? `All (${claims.length})` : `${LABELS[option].text} (${counts[option]})`}
          </button>
        ))}
      </div>

      <ul className="space-y-3">
        {shown.map((claim) => (
          <li key={claim.claim_id} className="rounded-xl border border-slate-200 p-4 dark:border-slate-800">
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone={LABELS[claim.label].tone}>{LABELS[claim.label].text}</Badge>
              {claim.action === "removed" && <Badge tone="red">removed from report</Badge>}
              {claim.action === "flagged" && <Badge tone="amber">flagged in report</Badge>}
              <span className="text-xs text-slate-400">{claim.section}</span>
            </div>
            <p className="mt-2 font-medium text-slate-900 dark:text-slate-100">{claim.text}</p>
            {claim.sentence && claim.sentence !== claim.text && (
              <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                In the report: “<CitationText text={claim.sentence} titles={titles} />”
              </p>
            )}
            <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">{claim.rationale}</p>
            {claim.evidence.map((evidence) => (
              <blockquote
                key={evidence.chunk_id}
                className="mt-3 border-l-4 border-slate-300 bg-slate-50 px-3 py-2 text-sm dark:border-slate-600 dark:bg-slate-900"
              >
                <p className="text-slate-700 dark:text-slate-300">“{evidence.text}”</p>
                <footer className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                  {evidence.title} · {evidence.page > 0 ? `${evidence.section}, p. ${evidence.page}` : "abstract"}
                </footer>
              </blockquote>
            ))}
          </li>
        ))}
      </ul>
    </div>
  );
}
